"""#871 end to end: two real processes, one real gloo group, one rank's encode out of memory.

`tests/scheduler/test_rank_agreement_871.py` pins the pieces with an injected reducer. This is the
served shape: two `mp.Process` ranks, `torch.distributed` on gloo, `Scheduler._reduce_failure_flag`
doing a real all-reduce, and the request travelling through `_process_one_msg` exactly as it does
on the row -- minus the engine.

⭐ It is also the wedge detector. If the agreement were ever reached on only one of the two
outcomes, the rank that reached it would block in gloo forever; `faulthandler.dump_traceback_later`
turns that into a failed test with both stacks printed, instead of a hung suite.

⭐ Bullet 5 rides the same harness: `test_a_peer_that_never_arrives_*` starves the agreement of
one rank -- once by killing it, once by leaving it alive and never letting it arrive -- and asks
that the survivor refuse on its own deadline instead of parking in gloo for the default HALF HOUR.

⛔ What this CANNOT show is the kill itself: that needs the language model's embedding all-reduce
and NCCL's watchdog, so it needs the card. What it shows is the property the kill hangs off --
whether the two ranks take the same branch -- and `test_without_agreement_the_ranks_take_different
_branches` reproduces the pre-fix divergence deterministically, the way #795's harness reproduces
the dropped first request.
"""
from __future__ import annotations

import dataclasses
import logging
import socket

import pytest
import torch.multiprocessing as mp

JOIN_TIMEOUT_S = 60
# The row's `distributed_timeout` (60.0 served) shrunk to keep the suite quick: the property under
# test is that the wait is bounded by THIS number rather than by gloo's 30-minute default, and the
# number itself is the config field, not a constant in the code.
AGREEMENT_DEADLINE_S = 3.0
PATCH_D, HID, MERGE = 8, 4, 2
IMG_ID, PATCHES = 101, 8


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One spawned pair, described once. ⛔ `None` is NOBODY -- no rank OOMs, no rank goes absent;
    a rank index is the whole domain otherwise, so a sentinel index would collide with rank 0 the
    day this harness grows a `size=1` case. Frozen so it can key `_RUNS` directly."""

    failing_rank: int | None = None
    agree: bool = True
    size: int = 2
    absent_rank: int | None = None
    absent_stays_alive: bool = False


class _Recorder(logging.Handler):
    def __init__(self, out: list):
        super().__init__()
        self.out = out

    def emit(self, record):
        self.out.append(record.getMessage())


class _StubPrefill:
    """Records admission. `reserve_prefix` returning None is the no-prefix-hit path (patch 0019)."""

    image_token_id = IMG_ID

    def __init__(self):
        self.admitted = []
        self.released = 0

    def reserve_prefix(self, msg):
        return None

    def release_reservation(self, reservation):
        self.released += 1

    def add_one_req(self, msg, reservation=None):
        self.admitted.append(msg.uid)


def _msg(uid: int):
    import torch

    from freetoken.core import SamplingParams
    from freetoken.message.backend import UserMsg

    soft = PATCHES // (MERGE**2)
    prompt = torch.tensor([1, 2] + [IMG_ID] * soft + [3], dtype=torch.int32)
    pixels = torch.rand(PATCHES, PATCH_D, generator=torch.Generator().manual_seed(uid))
    pos = torch.zeros(PATCHES, 2, dtype=torch.int64)
    assert int((prompt == IMG_ID).sum()) == soft
    return UserMsg(
        uid=uid,
        input_ids=prompt,
        sampling_params=SamplingParams(max_tokens=4),
        pixel_values=pixels,
        image_position_ids=pos,
        image_patch_counts=[PATCHES],
    )


def _tower(fail: bool):
    import torch

    def encode(pixels, position_ids):
        if fail:
            raise torch.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2.21 GiB. GPU 0 has a total capacity of "
                "31.86 GiB of which 476.00 MiB is free."
            )
        return torch.zeros(pixels.shape[1] // (MERGE**2), HID)

    return encode


def _rank_main(rank: int, port: int, result, sc: Scenario):
    """One rank's scheduler process: real gloo group, real `_attach_mm_embeds`."""
    import faulthandler
    import os
    import sys
    import time

    import torch.distributed as dist

    from freetoken.distributed.info import set_tp_info

    faulthandler.dump_traceback_later(30, exit=True, file=sys.stderr)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=sc.size)
    set_tp_info(rank, sc.size)
    try:
        if sc.absent_rank is not None:
            # Both ranks are through init before either can vanish, so what the survivor meets is
            # a peer lost at the AGREEMENT and not a rendezvous that never completed.
            dist.barrier()
        if rank == sc.absent_rank:
            result.put((rank, {"never_reached_the_agreement": True}))
            if sc.absent_stays_alive:
                # Alive, connected, and never arriving: the shape that used to park the survivor
                # in gloo for the default 30 minutes. ⛔ Outlast the deadline by a MARGIN -- exiting
                # just after it would hand the survivor a `Connection closed by peer` instead, and
                # the test would pass with no deadline in the code at all (it did).
                time.sleep(AGREEMENT_DEADLINE_S * 5)
            return
        from types import SimpleNamespace

        from freetoken.message.tokenizer import ErrorReplyMsg
        from freetoken.scheduler.scheduler import Scheduler

        lines: list[str] = []
        handler = _Recorder(lines)
        logging.getLogger("freetoken.scheduler.scheduler").addHandler(handler)
        if not sc.agree:
            # The pre-fix engine: each rank acts on its OWN outcome. ⛔ Patch the SOURCE module --
            # `_attach_mm_embeds` imports `any_rank_failed` function-locally, so rebinding the
            # name on `scheduler` is a no-op that reads as the fix having been disabled when it
            # has not (it cost this file one full spawned run to find).
            from freetoken.scheduler import rank_agreement

            rank_agreement.any_rank_failed = lambda local, tp_size, reduce_max: local

        pm = _StubPrefill()
        sch = Scheduler.__new__(Scheduler)
        sch.engine = SimpleNamespace(
            max_seq_len=4096,
            model=SimpleNamespace(encode_images=_tower(fail=rank == sc.failing_rank)),
        )
        sch.config = SimpleNamespace(
            image_soft_token_limit=lambda: None,
            multimodal_prompt_limit=lambda: None,
            tp_info=SimpleNamespace(size=sc.size, rank=rank),
            distributed_timeout=AGREEMENT_DEADLINE_S,
        )
        sch.device = "cpu"
        sch.prefill_manager = pm
        sch.tp_cpu_group = dist.group.WORLD
        sch.sent = []
        sch.send_result = sch.sent.extend
        sch._abort_tombstones = {}

        msg = _msg(29)
        started = time.monotonic()
        Scheduler._process_one_msg(sch, msg)
        elapsed = time.monotonic() - started

        result.put((rank, {
            "elapsed": elapsed,
            "refused": [isinstance(m, ErrorReplyMsg) for m in sch.sent],
            "error": sch.sent[0].error if sch.sent else None,
            "admitted": pm.admitted,
            "mm_embeds_is_none": msg.mm_embeds is None,
            "log": lines,
        }))
    finally:
        result.close()
        result.join_thread()
        if sc.absent_rank is not None:
            # ⛔ No barrier and no `destroy_process_group` here: both are collectives, and the
            # whole scenario is that one rank is not there to meet them.
            os._exit(0)
        dist.barrier()
        dist.destroy_process_group()


_RUNS: dict[Scenario, dict] = {}


def _run(sc: Scenario):
    """Cached per scenario: a spawned rank pays a cold torch import, and the cases below ask only
    a handful of distinct questions of the engine. The payloads are read-only."""
    if sc not in _RUNS:
        _RUNS[sc] = _spawn(sc)
    return _RUNS[sc]


def _spawn(sc: Scenario):
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = [
        ctx.Process(
            target=_rank_main,
            args=(r, port, result, sc),
        )
        for r in range(sc.size)
    ]
    for p in procs:
        p.start()
    out = {}
    for _ in procs:
        rank, payload = result.get(timeout=JOIN_TIMEOUT_S)
        out[rank] = payload
    for p in procs:
        p.join(JOIN_TIMEOUT_S)
    alive = [p.pid for p in procs if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.kill()
    assert not alive, f"ranks still running after {JOIN_TIMEOUT_S} s (a wedge): {alive}"
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    return out


@pytest.mark.parametrize("failing_rank", [0, 1], ids=["rank0-oom", "rank1-oom"])
def test_one_ranks_oom_refuses_on_both_and_admits_on_neither(failing_rank):
    """⭐⭐ #871 CLOSED, in the shape the row runs it. Either card can be the one that runs out --
    #871 saw rank 0 because rank 0 carries the replicated ViT transient on top of an even memory
    split, but the fix must not depend on which."""
    out = _run(Scenario(failing_rank=failing_rank))

    assert out[0]["refused"] == [True] and out[1]["refused"] == [True]
    assert out[0]["admitted"] == [] and out[1]["admitted"] == [], (
        "the rank that encoded successfully entered the forward alone -- this is #871"
    )


@pytest.mark.parametrize("failing_rank", [0, 1], ids=["rank0-oom", "rank1-oom"])
def test_the_rank_that_succeeded_drops_its_embeddings(failing_rank):
    """Its tower ran and the output is live; the request never becomes a `Req`, so nothing
    downstream would ever free it."""
    out = _run(Scenario(failing_rank=failing_rank))
    survivor = 1 - failing_rank

    assert out[survivor]["mm_embeds_is_none"], "the successful rank kept embeddings it will not use"


def test_a_rank_1_oom_is_in_rank_1s_own_log():
    """⭐ Bullet 3 on a REAL rank 1, where `_TP_INFO` is genuinely set and `warning_rank0` would
    print nothing. Before this, a card-1 OOM left no line anywhere on the box."""
    out = _run(Scenario(failing_rank=1))

    assert any("out of memory" in line.lower() for line in out[1]["log"]), out[1]["log"]
    assert any("29" in line for line in out[1]["log"])


def test_the_refusing_peer_says_why_in_its_own_log():
    """Rank 0 encoded fine. Without its line, its log shows a refusal with no cause on it."""
    out = _run(Scenario(failing_rank=1))

    assert any("another rank" in line.lower() for line in out[0]["log"]), out[0]["log"]


def test_a_healthy_group_still_admits_on_both_ranks():
    """The path every image request on the row takes. It must cross the real collective and come
    out the other side admitting, on both ranks."""
    out = _run(Scenario())

    assert out[0]["admitted"] == [29] and out[1]["admitted"] == [29]
    assert out[0]["refused"] == [] and out[1]["refused"] == []


def test_without_agreement_the_ranks_take_different_branches():
    """⛔ The pre-fix engine, made deterministic -- the divergence NCCL's watchdog turns into a
    dead row 60 s later. Rank 0 refuses and rank 1 admits: from here, rank 1 enters the forward
    and blocks in the embedding all-reduce with nobody coming.

    If this ever passes with the ranks AGREEING, the fix has been made unreachable and every
    test above it is vacuous."""
    out = _run(Scenario(failing_rank=0, agree=False))

    assert out[0]["refused"] == [True] and out[0]["admitted"] == []
    assert out[1]["refused"] == [] and out[1]["admitted"] == [29]


# ---------------------------------------------------------------------------------------------
# Bullet 5: the survivor of a lost peer refuses on its own deadline.
# ---------------------------------------------------------------------------------------------


def test_a_peer_that_never_arrives_alive_is_a_refusal_at_the_deadline():
    """⭐⭐ The bullet, in the shape that used to be a HALF-HOUR park. Rank 1 is up, connected and
    simply never reaches the agreement; rank 0's own encode succeeded, so before this it would sit
    in gloo on `new_group(backend="gloo")`'s default `default_pg_timeout` -- 30 minutes -- and the
    request would be neither served nor refused for the whole of it."""
    out = _run(Scenario(absent_rank=1, absent_stays_alive=True))

    assert out[1] == {"never_reached_the_agreement": True}
    assert out[0]["refused"] == [True] and out[0]["admitted"] == []
    assert out[0]["elapsed"] >= AGREEMENT_DEADLINE_S, "it did not actually wait for its peer"
    # ⭐ The peer is still there when this fires: it stays alive for 5x the deadline, so a rank
    # that came back inside 2x can only have come back on a deadline of its own.
    assert out[0]["elapsed"] < AGREEMENT_DEADLINE_S * 2, (
        f"it waited on the GROUP's timeout, not its own: {out[0]['elapsed']:.1f}s"
    )


def test_a_peer_that_died_is_a_refusal_and_not_a_traceback():
    """The likelier half: gloo notices the closed socket and raises AT ONCE. ⛔ Uncaught that
    leaves `_process_one_msg`, and `run_forever` catches `KeyboardInterrupt` and nothing else --
    so the fix for #871 would have become its own way to kill a rank, on a traceback reading as a
    distributed bug rather than as the peer's death."""
    out = _run(Scenario(absent_rank=1))

    assert out[1] == {"never_reached_the_agreement": True}
    assert out[0]["refused"] == [True] and out[0]["admitted"] == []
    assert out[0]["elapsed"] < 30


@pytest.mark.parametrize("stays_alive", [True, False], ids=["hung-peer", "dead-peer"])
def test_the_survivor_records_that_it_lost_its_peer(stays_alive):
    """A refusal with no cause on it is the hole bullet 3 closed; a lost peer must not reopen it."""
    out = _run(Scenario(absent_rank=1, absent_stays_alive=stays_alive))

    assert any("peer" in line.lower() for line in out[0]["log"]), out[0]["log"]


def test_the_survivor_keeps_no_embeddings_when_its_peer_is_gone():
    """Its tower ran, and the refusal means nothing downstream will ever free the result."""
    out = _run(Scenario(absent_rank=1, absent_stays_alive=True))

    assert out[0]["mm_embeds_is_none"]
