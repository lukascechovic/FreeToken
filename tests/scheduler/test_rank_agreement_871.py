"""Patch 0022 (llm-server #871), bullet 1: the ranks agree before a per-rank failure is acted on.

`_attach_mm_embeds`'s `except` is the only early return in that path whose outcome can differ per
rank -- it depends on that rank's card at that instant, not on the request. When it differs, the
refusing rank returns to idle while its peer enters the forward and blocks in the embedding
all-reduce; NCCL's watchdog takes the whole process down 60 s later, with the OOM already scrolled
off the tail of the log (#871).

`any_rank_failed` is the agreement that closes it: every rank puts its OWN outcome in, and every
rank takes the SAME answer out. What these tests hold down:

1. At TP=1 the collective is not merely a no-op -- it is never reached. A single-card row must not
   pay, or risk, one instruction of distributed machinery for a failure it cannot disagree about.
2. A peer's failure binds this rank even when this rank succeeded. That is the whole fix: the rank
   whose encode worked must NOT enter the forward alone.
3. The flag that crosses is a CPU int64. `tp_cpu_group` is gloo in BOTH branches of
   `engine._init_communication`, so the agreement can never touch the NCCL path -- but only if the
   tensor it hands over stays on the host.
"""
from __future__ import annotations

import logging

import pytest
import torch

from freetoken.scheduler.rank_agreement import any_rank_failed


class _Reducer:
    """A stand-in for `torch.distributed.all_reduce(..., op=MAX, group=tp_cpu_group)`.

    Records what it was handed -- the served collective mutates its tensor in place, so the test
    that the peer's value wins and the test that this rank's value is PUBLISHED are the same call.
    """

    def __init__(self, peer_failed: bool = False):
        self.peer_failed = peer_failed
        self.calls: list[torch.Tensor] = []

    def __call__(self, flag: torch.Tensor) -> None:
        self.calls.append(flag.clone())
        if self.peer_failed:
            flag.fill_(1)  # MAX of a 0/1 flag with a failing peer is 1, whatever this rank put in


def _never(flag: torch.Tensor) -> None:
    raise AssertionError("TP=1 must not reach the collective at all")


@pytest.mark.parametrize("local_failed", [False, True])
def test_tp1_never_reaches_the_collective(local_failed):
    """One rank cannot disagree with itself, so it must not pay for the agreement."""
    assert any_rank_failed(local_failed, tp_size=1, reduce_max=_never) is local_failed


def test_a_peers_failure_binds_a_rank_whose_own_encode_succeeded():
    """#871 itself: rank 0 OOMs, rank 1 does not. Rank 1 must refuse anyway."""
    reducer = _Reducer(peer_failed=True)
    assert any_rank_failed(False, tp_size=2, reduce_max=reducer) is True


def test_this_ranks_failure_is_published_to_its_peer():
    """The mirror: the rank that DID fail has to put a 1 into the collective, or the peer that
    succeeded sees a clean group and enters the forward alone -- the same desync, inverted."""
    reducer = _Reducer(peer_failed=False)
    assert any_rank_failed(True, tp_size=2, reduce_max=reducer) is True
    assert int(reducer.calls[0].max().item()) == 1


def test_a_healthy_group_is_not_turned_into_a_refusal():
    """The healthy path is every image request on the row; it must come back False."""
    reducer = _Reducer(peer_failed=False)
    assert any_rank_failed(False, tp_size=2, reduce_max=reducer) is False
    assert int(reducer.calls[0].max().item()) == 0


def test_the_flag_that_crosses_is_a_cpu_int64():
    """gloo takes host tensors. A device tensor here would put the agreement on the card and
    reintroduce exactly the coupling the fix exists to remove."""
    reducer = _Reducer()
    any_rank_failed(True, tp_size=2, reduce_max=reducer)
    (flag,) = reducer.calls
    assert flag.dtype is torch.int64
    assert flag.device.type == "cpu"
    assert flag.numel() == 1


def test_the_collective_runs_exactly_once_per_call():
    """One small all-reduce per multimodal request is the whole budget this fix asked for."""
    reducer = _Reducer()
    any_rank_failed(False, tp_size=4, reduce_max=reducer)
    assert len(reducer.calls) == 1


# ---------------------------------------------------------------------------------------------
# Bullet 2: `_attach_mm_embeds` acts on the AGREED outcome, never on its own.
# ---------------------------------------------------------------------------------------------

D, MERGE, HID = 8, 2, 4


def _pixels(patch_counts):
    total = sum(patch_counts)
    return torch.rand(total, D), torch.zeros(total, 2, dtype=torch.int64), list(patch_counts)


class _Tower:
    """Stands in for the replicated vision tower: `[1, P, D]` in, `[P // MERGE**2, HID]` out."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def __call__(self, pixels, position_ids):
        self.calls += 1
        if self.fail:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.20 GiB")
        return torch.zeros(pixels.shape[1] // (MERGE**2), HID)


def _msg(uid=1, patch_counts=(8,)):
    from types import SimpleNamespace

    pixel_values, position_ids, counts = _pixels(patch_counts)
    return SimpleNamespace(
        uid=uid,
        pixel_values=pixel_values,
        image_position_ids=position_ids,
        image_patch_counts=counts,
        mm_embeds=None,
    )


def _sched(tower, tp_size, reducer, *, has_tower=True):
    """The `__new__` + namespace rig `test_skip_cached_encode_892.py` uses, plus `tp_info`."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sch = Scheduler.__new__(Scheduler)
    sch.engine = SimpleNamespace(
        model=SimpleNamespace(**({"encode_images": tower} if has_tower else {}))
    )
    sch.config = SimpleNamespace(tp_info=SimpleNamespace(size=tp_size, rank=0))
    sch.device = "cpu"
    sch._reduce_failure_flag = reducer
    return sch


def _attach(sch, msg):
    from freetoken.scheduler.scheduler import Scheduler

    return Scheduler._attach_mm_embeds(sch, msg)


def test_a_rank_whose_encode_succeeded_still_refuses_when_its_peer_failed():
    """⭐⭐ THE FIX. This is the rank that used to enter the forward alone and block in the
    embedding all-reduce until NCCL's watchdog took the process down 60 s later."""
    from freetoken.message.tokenizer import ErrorReplyMsg

    tower = _Tower(fail=False)
    sch = _sched(tower, tp_size=2, reducer=_Reducer(peer_failed=True))
    msg = _msg()

    error = _attach(sch, msg)

    assert isinstance(error, ErrorReplyMsg) and error.uid == msg.uid
    assert tower.calls == 1, "it did the work; it just must not act on it alone"


def test_the_refusing_rank_drops_the_embeddings_it_computed():
    """The tower ran and its output is on the card. The request is refused, so nothing downstream
    will ever free it -- there being no `Req` yet -- and the next image needs that room."""
    sch = _sched(_Tower(fail=False), tp_size=2, reducer=_Reducer(peer_failed=True))
    msg = _msg()

    _attach(sch, msg)

    assert msg.mm_embeds is None


def test_a_rank_that_failed_reports_its_own_error_not_the_peers():
    """The refusal the client reads should name what actually went wrong on the rank that broke."""
    from freetoken.message.tokenizer import ErrorReplyMsg

    sch = _sched(_Tower(fail=True), tp_size=2, reducer=_Reducer(peer_failed=False))

    error = _attach(sch, _msg())

    assert isinstance(error, ErrorReplyMsg)
    assert "out of memory" in error.error.lower()


def test_the_healthy_path_still_attaches_and_admits():
    """Every image request on the row takes this line; it must come back None with embeddings."""
    sch = _sched(_Tower(fail=False), tp_size=2, reducer=_Reducer(peer_failed=False))
    msg = _msg(patch_counts=(8, 12))

    assert _attach(sch, msg) is None
    assert msg.mm_embeds is not None and msg.mm_embeds.shape == (5, HID)


def test_the_agreement_is_reached_on_the_SUCCESS_path_too():
    """⛔⛆ The one way this fix could be worse than the bug. If only a FAILING rank called the
    collective, a healthy peer would never arrive and the failing rank would block in gloo --
    a desync with the same shape, moved one function earlier. Both outcomes must reduce."""
    ok, bad = _Reducer(peer_failed=False), _Reducer(peer_failed=False)
    _attach(_sched(_Tower(fail=False), tp_size=2, reducer=ok), _msg())
    _attach(_sched(_Tower(fail=True), tp_size=2, reducer=bad), _msg())
    assert len(ok.calls) == 1 and len(bad.calls) == 1


def test_tp1_refuses_without_ever_reaching_the_collective():
    """Both deployed single-card vision rows take this path and behave exactly as they do today."""
    from freetoken.message.tokenizer import ErrorReplyMsg

    sch = _sched(_Tower(fail=True), tp_size=1, reducer=_never)

    assert isinstance(_attach(sch, _msg()), ErrorReplyMsg)


def test_a_model_with_no_vision_tower_refuses_before_any_agreement():
    """Deterministic in the model config, so identical on every rank -- and it must stay ABOVE the
    agreement, because a rank refusing here has no peer waiting to hear about it."""
    from freetoken.message.tokenizer import ErrorReplyMsg

    sch = _sched(None, tp_size=2, reducer=_never, has_tower=False)

    assert isinstance(_attach(sch, _msg()), ErrorReplyMsg)


# ---------------------------------------------------------------------------------------------
# Bullet 3: a failure on a NON-PRIMARY rank has to be visible.
#
# `logger.warning_rank0` prints on rank 0 only (`utils/logger.py`'s `_call_rank0`). The encode
# OOM is the one event here that happens on ONE card, so rank-0 gating means a rank-1 OOM is
# logged NOWHERE -- and before bullet 2 that was the case that killed the row two minutes later
# with no cause on record anywhere. #871 was rank 0, which is the only reason it was diagnosable.
#
# These assert the report is not rank-gated, without mutating the process-wide `_TP_INFO`
# (`set_tp_info` refuses a second call, and `_call_rank0` caches what it reads). Bullet 4's
# two-process test makes the same assertion on a REAL rank 1.
# ---------------------------------------------------------------------------------------------


class _RecordingLogger:
    """Records which logger method carried each message -- the rank-gated ones are `*_rank0`."""

    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def _record(self, which):
        def call(msg, *args, **kwargs):
            self.lines.append((which, msg % args if args else msg))

        return call

    def __getattr__(self, which):
        if not hasattr(logging.Logger, which.removesuffix("_rank0")):
            # ⛔ Otherwise a misspelled call (`errr`) records happily and the assertion still
            # passes -- the recorder would be asserting about a method the code never has.
            raise AttributeError(which)
        return self._record(which)

    def reaching_every_rank(self) -> list[str]:
        return [text for which, text in self.lines if not which.endswith("_rank0")]

    def rank0_only(self) -> list[str]:
        return [text for which, text in self.lines if which.endswith("_rank0")]


@pytest.fixture
def recorded(monkeypatch):
    from freetoken.scheduler import scheduler as scheduler_module

    rec = _RecordingLogger()
    monkeypatch.setattr(scheduler_module, "logger", rec)
    return rec


def test_an_encode_failure_is_reported_on_every_rank_not_just_rank_0(recorded):
    """⛔ The diagnostic hole: rank-gated, a card-1 OOM leaves NO line anywhere."""
    sch = _sched(_Tower(fail=True), tp_size=2, reducer=_Reducer(peer_failed=False))

    _attach(sch, _msg(uid=29))

    reported = " ".join(recorded.reaching_every_rank())
    assert "29" in reported and "out of memory" in reported.lower()
    assert recorded.rank0_only() == [], "nothing about this failure may be rank-0 gated"


def test_the_rank_that_refuses_in_agreement_records_why(recorded):
    """Its own encode worked, so without a line this rank's log shows a refusal with no cause --
    and reading the two ranks' logs side by side is how the desync gets diagnosed at all."""
    sch = _sched(_Tower(fail=False), tp_size=2, reducer=_Reducer(peer_failed=True))

    _attach(sch, _msg(uid=30))

    reported = " ".join(recorded.reaching_every_rank())
    assert "30" in reported and "rank" in reported.lower()


def test_the_healthy_path_stays_silent(recorded):
    """Every image request on the row takes it; it must not add a line per image."""
    _attach(_sched(_Tower(fail=False), tp_size=2, reducer=_Reducer(peer_failed=False)), _msg())

    assert recorded.lines == []


# ---------------------------------------------------------------------------------------------
# Bullet 5: the agreement carries its OWN deadline, and a peer that never answers is a logged
# refusal rather than an exception in the scheduler loop.
#
# ⛔ The group is NOT armed with the row's timeout. On the served TP=2 path
# `engine._init_communication` builds it with `new_group(backend="gloo")` and no `timeout=`, so
# it takes torch's `default_pg_timeout` -- 30 MINUTES on the pinned torch (2.11.0+rocm7.14.0) --
# not the `distributed_timeout: float = 60.0` the process group beside it was initialised with.
#
# ⚠ Belt-and-braces, not correctness: after bullets 1-4 a rank can be left unmet only if its peer
# died hard, and the backend supervisor already reports that. What these hold down is that the
# fix cannot itself become a new way to lose the row.
# ---------------------------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402 -- bullet 5's section reads with its own imports

DEADLINE_S = 60.0


class _Work:
    """`torch.distributed`'s async handle. Records the deadline it was waited on."""

    def __init__(self, raises: BaseException | None = None):
        self.raises = raises
        self.waited: list = []

    def wait(self, timeout=None):
        self.waited.append(timeout)
        if self.raises is not None:
            raise self.raises
        return True


def _reduce_sched(*, timeout_s=DEADLINE_S):
    """A `Scheduler` carrying only what `_reduce_failure_flag` reads: the group and the deadline."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sch = Scheduler.__new__(Scheduler)
    sch.config = SimpleNamespace(distributed_timeout=timeout_s, tp_info=SimpleNamespace(size=2))
    sch.tp_cpu_group = object()  # the real group; never touched, all_reduce is stubbed
    return sch


@pytest.fixture
def collective(monkeypatch):
    """Stub `torch.distributed.all_reduce`, returning a handle the test controls."""

    def install(work, raises: BaseException | None = None):
        calls = []

        def all_reduce(tensor, op=None, group=None, async_op=False):
            calls.append({"tensor": tensor, "op": op, "group": group, "async_op": async_op})
            if raises is not None:
                # gloo surfacing an already-dead peer at ISSUE time, before there is a handle.
                raise raises
            return work

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        return calls

    return install


def _reduce(sch, flag):
    from freetoken.scheduler.scheduler import Scheduler

    Scheduler._reduce_failure_flag(sch, flag)


def test_the_agreement_waits_on_its_own_deadline_not_gloos_default(collective):
    """⭐⭐ The bullet. Inheriting the group's default would park a rank for HALF AN HOUR."""
    work = _Work()
    calls = collective(work)
    flag = torch.tensor([0], dtype=torch.int64)

    _reduce(_reduce_sched(), flag)

    assert calls[0]["async_op"] is True, "a synchronous all_reduce cannot carry a deadline at all"
    assert work.waited == [timedelta(seconds=DEADLINE_S)]
    assert work.waited[0] < timedelta(minutes=30), "this is gloo's default; the point is not to"


def test_the_deadline_is_the_rows_own_distributed_timeout(collective):
    """One field, not a second number to keep in step: `distributed_timeout` is what NCCL's
    watchdog is armed with, so a peer this rank gives up on is one the watchdog gave up on too."""
    work = _Work()
    collective(work)

    _reduce(_reduce_sched(timeout_s=12.5), torch.tensor([0], dtype=torch.int64))

    assert work.waited == [timedelta(seconds=12.5)]


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Operation timed out!"),
        RuntimeError("[gloo/transport/tcp/pair.cc:547] Connection closed by peer"),
    ],
    ids=["hung-peer", "dead-peer-at-the-wait"],
)
def test_a_peer_that_never_answers_refuses_instead_of_raising(collective, exc):
    """⛔ `run_forever` catches `KeyboardInterrupt` and nothing else, so an escaping gloo error
    kills this rank on a traceback that reads as a distributed bug. Both shapes are seen: a gone
    peer raises at once, a hung one at the deadline."""
    work = _Work(raises=exc)
    collective(work)
    flag = torch.tensor([0], dtype=torch.int64)

    _reduce(_reduce_sched(), flag)  # must not raise

    assert int(flag.item()) == 1, "fail CLOSED -- the safe answer is the one nobody forwards on"


def test_a_peer_already_gone_refuses_when_gloo_raises_at_the_enqueue(recorded, collective):
    """⛔⛆ The `dead-peer-at-the-wait` arm above raises from `work.wait()`, but the docstring's
    "raises immediately" case is gloo raising from `all_reduce` ITSELF -- there is no handle to
    wait on yet. Guarding only the wait leaves that one escaping `_process_one_msg` and then
    `run_forever`, which catches `KeyboardInterrupt` and nothing else: 0022 would itself become
    a new way to lose the row. ⭐ Red with the enqueue outside the `try` (checked by mutation).
    """
    gone = RuntimeError("[gloo/transport/tcp/pair.cc:547] Connection closed by peer")
    collective(_Work(), raises=gone)
    flag = torch.tensor([0], dtype=torch.int64)

    _reduce(_reduce_sched(), flag)  # must not raise

    assert int(flag.item()) == 1, "fail CLOSED -- the safe answer is the one nobody forwards on"
    assert "peer" in " ".join(recorded.reaching_every_rank()).lower(), (
        "a rank that refused because its peer was gone must say so on its own log"
    )


def test_a_lost_peer_is_reported_on_whichever_rank_lost_it(recorded, collective):
    """Which rank lost its peer is the whole content of the line, so it must not be rank-0 gated:
    rank 1 losing rank 0 would otherwise be logged nowhere -- bullet 3's principle, again."""
    collective(_Work(raises=RuntimeError("Operation timed out!")))

    _reduce(_reduce_sched(), torch.tensor([0], dtype=torch.int64))

    reported = " ".join(recorded.reaching_every_rank())
    assert "peer" in reported.lower() and "timed out" in reported.lower()
    assert recorded.rank0_only() == []


def test_a_healthy_collective_is_left_exactly_as_the_group_returned_it(collective):
    """The path every image request on the row takes: the deadline must not perturb the value."""
    work = _Work()
    collective(work)
    flag = torch.tensor([0], dtype=torch.int64)

    _reduce(_reduce_sched(), flag)

    assert int(flag.item()) == 0
