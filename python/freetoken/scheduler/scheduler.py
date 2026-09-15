from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.multimodal import images_too_large_message, prompt_too_long_message
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `scheduler/scheduler.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 76293fe8d4b6e1cfd1cba38e1bb8092b,
# 1208 lines) BIND-MOUNTED over the installed package, plus the edits marked `#801 bullet 8`
# below. ⛔ It is NOT a patch in the Dockerfile ladder and is in NO image. ⚠ A `-v` that silently
# does not take leaves the row running the IMAGE's scheduler while every log line looks like the
# arm we think we launched (#866) -- hence a marker, on stderr, before logging is up.
import os as _ft801_os
import sys as _ft801_sys

print(
    "[#801] overlay ACTIVE: scheduler/scheduler.py bind-mounted from the repo "
    f"(pid {_ft801_os.getpid()}, base md5 76293fe8d4b6e1cfd1cba38e1bb8092b, "
    f"FREETOKEN_MTP801_VERIFY={_ft801_os.getenv('FREETOKEN_MTP801_VERIFY', '<unset>')})",
    file=_ft801_sys.stderr,
    flush=True,
)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


def _soft_token_count(msg: UserMsg, merge_size: int) -> int | None:
    """How many soft tokens this request's images expand to, or None if it carries none.

    Two shapes reach the scheduler and both are counted here, because the cap has to bind
    whichever one arrived (the contract ``UserMsg`` states for the prompt ceiling):

      * the ONLINE path sends preprocessed pixels. Since #890 those are PACKED,
        ``[sum(P), 2]`` with no padding rows, so every row counts; the offline tray
        ``[N, P, 2]`` right-padded with ``(-1, -1)`` also still reaches here, and the same
        ``>= 0`` test reads both. Counting padding would refuse a legal request whose batch
        happens to contain one large image -- which is why the test is on the ids, not the
        shape.
      * the OFFLINE path attaches ``mm_embeds`` directly, one row per soft token already.

    ⛔ Counted BEFORE ``_attach_mm_embeds`` runs the tower: rejecting after the vision pass
       would pay the very VRAM transient the cap exists to bound.
    """
    embeds = msg.mm_embeds
    if embeds is not None:
        return int(embeds.shape[0])
    pos = msg.image_position_ids
    if pos is None:
        return None
    patches = int((pos[..., 0] >= 0).sum())
    return patches // (merge_size * merge_size)


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            # The placeholder id a multimodal prompt's soft tokens scatter over; the prefill
            # adder needs it to hand each chunk its own rows. None on a text-only model. Read
            # with getattr only because the scheduler tests build model_config as a bare
            # SimpleNamespace; a real ModelConfig always declares the field, and the adder
            # asserts on a multimodal prompt that reaches it as None.
            image_token_id=getattr(config.model_config, "image_token_id", None),
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        # ── #801 r6 b9am: flush the last request's α row before anything can raise ──────
        try:
            from freetoken.models.qwen4_exp.spec import drain_alpha

            drain_alpha(getattr(self.engine, "model", None))
        except Exception:  # pragma: no cover - measurement must never kill a served row
            pass
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        # #801 bullet 8: per-request token lists, once for the batch -- see the loop below.
        _ft801_published = _ft801_published_tokens(batch, next_tokens_cpu)
        # ── #801 r6 b9bh: the LENGTH verdict, per PUBLISHED token ──────────────────────
        # ⛔⛆ `hit_length = not req.can_decode` read the request's LIVE `device_len`, and
        #   `spec.commit_verify` advanced that by the WHOLE accepted pair before this drain
        #   ran. A pair landing on the cap therefore tripped *length* on its FIRST token, the
        #   loop below broke, and the second token was appended to `input_ids` by the tail
        #   loop but never shipped -- load 24 measured the client one token short on seven of
        #   nine cells. `publish_plan` judges token k against the `device_len` a one-token-per-
        #   step row would have had at the same generated index.
        # ⭐ MODEL-AGNOSTIC and hoisted: one `getattr` per DRAIN, not per request. A model
        #   without the hook -- and a row with the verify dial off -- keeps `not req.can_decode`
        #   byte for byte, which for a one-token commit is the same verdict by arithmetic.
        _ft801_lengths_of = getattr(self.engine.model, "mtp_publish_lengths", None)
        # ── #801 r6 b9bn: THE DRAINED FORWARD'S OWN POST-COMMIT `cached_len` ───────────────
        # ⛔⛆ 9bh fixed the FORMULA and load 25 showed the INPUT was wrong. `req.cached_len` read
        #   HERE is the NEXT forward's post-commit value -- `overlap_loop` forwards batch N
        #   before draining batch N-1, so N's commit is already in the number on every row (36 of
        #   36, load 25). Judged against it the verify arm finishes one generated token early and
        #   the following drain hits `already_finished` and ships nothing.
        # ⭐ `engine/engine.py::forward_batch` records the value on the batch after BOTH advance
        #   paths, and this is the batch that forward produced -- so the tuple below belongs to
        #   the forward being drained, whatever ran after it.
        # ⛔ ABSENT -> the image's own `not req.can_decode`, byte for byte, exactly as a model
        #   without `mtp_publish_lengths` already degrades. ⛔⛆ It must NOT fall back to the live
        #   `req.cached_len`: that is the defect this line exists to remove, and it would come
        #   back wearing the costume of a rare flake instead of an unmounted overlay.
        _ft801_own = getattr(batch, "ft801_post_commit", None)
        # ── #801 r6 b9bd: THE PUBLISH LEDGER, AND IT EXISTS BECAUSE LOAD 22 MEASURED THE
        #   LEAK AND THE NEXT LINK IS STILL DERIVED ──────────────────────────────────────────
        # ⛔⛆ Load 22's retire-time ledger caught the leaking retire in the act -- `ids_numel`
        #   127 against `cached_len` 128, both frees EMPTY, `freed_total` 0 -- with 9az's drain
        #   fix MOUNTED and md5-verified. ⇒ the missing id never reaches `_ft801_published[i]`
        #   at all, and appending "the rest of the published list" cannot recover a token that
        #   was never published. This records, per DRAINED FORWARD and per request, the two
        #   sides of that hand-off: what `spec.commit_verify` was called with
        #   (`staged.committed`, at `plan.cu_seqlens`), what `published_tokens` actually SLICED
        #   out of `next_tokens_cpu`, and what reached `req.input_ids`.
        # ⛔ MEASURE IT, DO NOT DERIVE IT. This round has now read this path's source and been
        #   wrong twice (9ay's `insert:dedup`, 9az's fix), and load 21 proved the desk sweep's
        #   `drain_step` does not describe the deployed drain.
        # ⭐ MEASUREMENT-SAFE, like 9bb's ledger and unlike CHECKROW/GDNCHECK: no second
        #   forward, no rewind, no device read, nothing that makes `can_use_cuda_graph` decline
        #   the verify capture. It reports numbers this drain already holds. ⇒ a decode figure
        #   off a `PUBCHECK` load MAY be banked as the arm's.
        # ⭐ OFF BY DEFAULT, and the budget is read ONCE per Scheduler: a row that never sets the
        #   dial pays one `getattr` per drain. `FREETOKEN_MTP801_PUBCHECK=N` counts DRAINED
        #   FORWARDS (not retires, not steps), set by `arm_mtp_801.sh` from `FT801_PUBCHECK` --
        #   round 5 bullet 7c's rule: a dial the launcher cannot set is off.
        # ⛔ MODEL-AGNOSTIC, like every other hook this file carries: it reads the step the model
        #   staged on the batch (`ft801_verify_step`), never an import from `models/qwen4_exp/`.
        _ft801_pub = None
        _ft801_left = getattr(self, "_ft801_pubcheck_left", None)
        if _ft801_left is None:
            _ft801_left = int(_ft801_os.getenv("FREETOKEN_MTP801_PUBCHECK", "0") or 0)
        if _ft801_left > 0:
            self._ft801_pubcheck_left = _ft801_left - 1
            _ft801_step = getattr(batch, "ft801_verify_step", None)
            # ⚠ Recorded as -1 rather than raised when the staged step is short: an instrument
            #   may not kill the row it is measuring, and a -1 in the record is louder in the
            #   log than an IndexError three frames deep.
            _ft801_committed = list(getattr(_ft801_step, "committed", ()) or ())
            _ft801_cu = (
                [int(c) for c in _ft801_step.plan.cu_seqlens]
                if _ft801_step is not None
                else []
            )
            _ft801_pub = [
                {
                    "uid": getattr(req, "uid", None),
                    # what `spec.commit_verify` advanced this request by, this forward
                    "committed": (
                        1
                        if _ft801_step is None
                        else (int(_ft801_committed[i]) if i < len(_ft801_committed) else -1)
                    ),
                    "cu": (
                        i
                        if _ft801_step is None
                        else (int(_ft801_cu[i]) if i < len(_ft801_cu) else -1)
                    ),
                    # what `published_tokens` actually sliced out -- SHORT when the slice ran
                    # past the end of `next_tokens_cpu`, which python does silently
                    "published": int(_ft801_published[i].numel()),
                    "cached_len": int(req.cached_len),
                    "device_len": int(getattr(req, "device_len", -1)),
                    "ids_before": int(req.input_ids.numel()),
                    "appended": 0,
                    "tail": 0,
                }
                for i, req in enumerate(batch.reqs)
            ]
        else:
            self._ft801_pubcheck_left = _ft801_left
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                # ── #801 bullet 8: one OR TWO tokens per forward ─────────────────
                # ⛔⛆ The image reads `next_tokens_cpu[i]` with `i` the REQUEST's position. On a
                # verify step that tensor is one row per TOKEN, so `[i]` is the wrong element for
                # every request after the first and an accepted second token is never shipped.
                # `_ft801_published_tokens` returns exactly what this forward COMMITTED for each
                # request -- a plain decode batch is one token each, spelled as a one-element
                # tensor, which is today's behaviour and today's cost.
                # ⛔ `finished` is read AFTER this loop by the free/cache block below, so the
                # loop `break`s on the first terminal token: an accepted pair whose FIRST token
                # is an EOS must not ship the second.
                finished = False
                _ft801_appended = 0
                # ⭐ ONE call per request per drain, off integers this drain already holds.
                # ⛔⛆ #801 r6 b9bn: the FIRST argument is `_ft801_own[i]`, the drained forward's
                # OWN post-commit `cached_len` -- NEVER `req.cached_len`, which by now carries the
                # next forward's commit. See the hoist above.
                _ft801_lengths = (
                    None
                    if _ft801_lengths_of is None
                    or _ft801_own is None
                    or i >= len(_ft801_own)
                    else _ft801_lengths_of(
                        int(_ft801_own[i]),
                        int(_ft801_published[i].numel()),
                        int(req.max_device_len),
                    )
                )
                for _ft801_token in _ft801_published[i]:
                    req.append_host(_ft801_token.unsqueeze(0))
                    _ft801_appended += 1
                    next_token = int(_ft801_token.item())
                    # EOS / stop-string -> "stop", output budget exhausted -> "length";
                    # EOS and stop strings win over length.
                    hit_length = (
                        (not req.can_decode)
                        if _ft801_lengths is None
                        else bool(_ft801_lengths[_ft801_appended - 1])
                    )
                    hit_eos = (
                        not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                    )
                    matched_stop = (
                        self._match_stop_str(req)
                        if not hit_eos and req.sampling_params.stop_strs
                        else None
                    )
                    finished = hit_length or hit_eos or matched_stop is not None
                    finish_reason = (
                        ("stop" if (hit_eos or matched_stop is not None) else "length")
                        if finished
                        else None
                    )
                    if (
                        next_token == self.toolcall_anchor_id
                        and req.toolcall_anchor_len is None
                        and not finished
                    ):
                        req.toolcall_anchor_len = req.input_ids.numel()
                    reply.append(
                        DetokenizeMsg(
                            uid=req.uid,
                            next_token=next_token,
                            finished=finished,
                            finish_reason=finish_reason,
                            matched_stop=matched_stop,
                            stop_strs=req.sampling_params.stop_strs or None,
                        )
                    )
                    if finished:
                        break
                # ⛔⛆ #801 bullet 9az: EVERY COMMITTED TOKEN OWES `append_host` AN ID, INCLUDING
                # the ones the loop above broke before -- and the break above is REACHABLE ON A
                # PAIR'S FIRST TOKEN. `hit_length` is `not req.can_decode`, i.e.
                # `device_len >= max_device_len`, and `spec.commit_verify` advanced `device_len`
                # by the WHOLE accepted pair before this drain ran, so a pair landing on the
                # output cap trips *length* on token 1 and token 2 was never appended.
                # ⛔⛆ THAT IS A PAGE LEAK, NOT A COSMETIC SHORT REPLY. `_cache_req_hybrid`'s
                # finish-donate keys the tree on `_cache_ids(req)` -- and
                # `kvcache/hybrid_radix_cache.py:96` IGNORES the length the caller sliced to:
                # `insert_len = align_down(len(input_ids), page_size)`. 127 ids against 128 page
                # indices seats `align_down(127, 64) = 64`, returns `prefix_len = 64`, and BOTH
                # the dedup free (`page_indices[64:max(64,64)]`) and `_padded_tail` (`[128:128]`)
                # are EMPTY. The tail page is charged, in the row, and owned by nobody:
                # `free_pages(4094) + cache_pages(1) != num_pages(4096)` -- load 20's signature,
                # reproduced at the desk by `repro_page_ledger_801.py` (prompt 65, max_new 64,
                # alt-RA: free(14) + cache(1) != 16 on the 16-page pool).
                # ⭐ INVARIANT: `req.input_ids.numel() == req.cached_len` at retire.
                # ⭐ A NON-SPECULATING ROW IS UNCHANGED BYTE FOR BYTE -- it commits one token per
                # forward, so the loop above never breaks early and this slice is always empty.
                # ⚠ 9az left `hit_length` alone deliberately, so this loop ALSO carried the
                # client's short reply. #801 r6 b9bh fixed the verdict above, so on a *length*
                # finish this slice is now empty -- the pair's second token ships. It still
                # fires on an EOS or a stop string landing on a pair's FIRST token, which is
                # the case that must NOT ship the second, and the ids are still owed.
                _ft801_tail = 0
                for _ft801_token in _ft801_published[i][_ft801_appended:]:
                    req.append_host(_ft801_token.unsqueeze(0))
                    _ft801_tail += 1
                # ── #801 r6 b9bd: what this block APPENDED, counted where it appends ─────
                # ⛔⛆ Both loops, and counted rather than re-derived from
                # `ids_after - ids_before`: a derivation agrees with itself even when a future
                # edit appends somewhere else in this block, which is exactly the miss 9bb's own
                # gate caught in `cache.py` on its first run. A no-op when the dial is off.
                if _ft801_pub is not None:
                    _ft801_pub[i]["appended"] = _ft801_appended
                    _ft801_pub[i]["tail"] = _ft801_tail

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        # ── #801 r6 b9bd: emit the publish ledger ───────────────────────────────────────
        # ⛔ HERE, not inside the loop: `skip` has to be read with `self.finished_reqs` still
        #   holding what the loop itself tested, and a request the loop `continue`d never
        #   reaches an in-loop emit -- which is the whole point, since a skipped request is a
        #   forward whose committed tokens were dropped in silence.
        # ⚠ `ids_after` / `cached_len_after` are read here rather than per request: nothing in
        #   this drain touches another request's `input_ids`, and the retire (`_free_req_resources`
        #   -> `_cache_req_hybrid`, 9bb's ledger) has already run by now, so the two instruments
        #   describe the SAME moment and can be read against each other.
        if _ft801_pub is not None:
            import json as _ft801_json

            for _ft801_i, _ft801_req in enumerate(batch.reqs):
                _ft801_rec = _ft801_pub[_ft801_i]
                _ft801_rec["skip"] = (
                    "chunked"
                    if isinstance(_ft801_req, ChunkedReq)
                    else "aborted"
                    if _ft801_req.aborted
                    else "already_finished"
                    if _ft801_req in self.finished_reqs
                    else None
                )
                _ft801_rec["ids_after"] = int(_ft801_req.input_ids.numel())
                _ft801_rec["cached_len_after"] = int(_ft801_req.cached_len)
                # ⭐ The number this whole instrument exists to print, greppable in a 14-minute
                #   log: what the commit advanced by, minus what reached the host ids.
                _ft801_rec["deficit"] = _ft801_rec["committed"] - (
                    _ft801_rec["appended"] + _ft801_rec["tail"]
                )
            print(
                "[#801] pubcheck: "
                + _ft801_json.dumps(
                    {
                        "left": int(self._ft801_pubcheck_left),
                        "staged": getattr(batch, "ft801_verify_step", None) is not None,
                        "is_prefill": bool(batch.is_prefill),
                        "rows": list(next_tokens_cpu.shape),
                        "reqs": _ft801_pub,
                    }
                ),
                file=_ft801_sys.stderr,
                flush=True,
            )
        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            # ── #801 round 7: what this forward COMMITTED, so the log line is tokens/s ───
            # ⭐ MODEL-AGNOSTIC, like every other #801 hook in this file: on an unstaged batch
            #   -- every prefill, every flag-off row, every no-draft decode step --
            #   `_ft801_published_tokens` returns one one-element tensor per request, so this
            #   sum IS `len(batch.reqs)` and the reporter logs exactly what the image logs.
            # ⚠ Counts every row of the forward, including one the loop below then skips as
            #   aborted or already-finished. That is the IMAGE's own treatment (`len(batch.reqs)`
            #   counts those rows too), kept deliberately so the only thing this edit changes is
            #   steps -> tokens.
            generated_tokens=sum(int(_ft801_p.numel()) for _ft801_p in _ft801_published),
        )
        self.send_result(reply)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _vision_merge_size(self) -> int:
        """The model's ``spatial_merge_size`` -- how many patches collapse into one soft token.

        Read from the loaded model rather than assumed, because it is what the tower actually
        applies (`models/qwen4_exp/vision.py`'s ``merge_unit``). Falls back to 2, the value
        every Qwen2VL-style checkpoint this engine serves uses, so a model that does not
        publish one cannot turn the cap into a crash.
        """
        visual = getattr(self.engine.model, "visual", None)
        config = getattr(visual, "_vc", None) or getattr(visual, "config", None)
        size = getattr(config, "spatial_merge_size", None)
        return int(size) if isinstance(size, int) and size > 0 else 2

    def _multimodal_ceiling_error(self, msg: UserMsg) -> ErrorReplyMsg | None:
        """Refuse an image request over either of the operator's two caps.

        ``--max-image-soft-tokens`` bounds what the IMAGES expand to; it is checked first and
        rejects before ``_attach_mm_embeds`` runs the tower, which is the point of it -- that
        pass is the VRAM transient the cap exists to bound. ``--max-multimodal-prompt-tokens``
        bounds the WHOLE prompt of an image request. ⚠ They are separate policies and #841
        split them for a measured reason: one number cannot both keep a picture small and let
        it arrive in a long conversation.

        Policy only: the engine chunks a multimodal prompt across prefill passes like a text
        prompt, so with neither cap set an image prompt is bounded by the context check above
        and nothing else. Every request carrying vision input passes through here, whether it
        arrived as pixels from the tokenizer worker or as embeddings the in-process offline
        API attached itself.
        """
        soft_limit = self.config.image_soft_token_limit()
        if soft_limit is not None:
            soft = _soft_token_count(msg, self._vision_merge_size())
            if soft is not None and soft > soft_limit:
                return ErrorReplyMsg(
                    uid=msg.uid,
                    error=images_too_large_message(soft, soft_limit),
                    code="context_length_exceeded",
                )
        limit = self.config.multimodal_prompt_limit()
        input_len = len(msg.input_ids)
        if limit is not None and input_len > limit:
            return ErrorReplyMsg(
                uid=msg.uid,
                error=prompt_too_long_message(input_len, limit),
                code="context_length_exceeded",
            )
        return None

    def _attach_mm_embeds(self, msg: UserMsg, skip_images: int = 0) -> ErrorReplyMsg | None:
        """Run the vision tower for an admitted image prompt; returns the client's error, or None.

        This is where the pixels the tokenizer worker sent become soft-token embeddings, on the
        device that holds the model. A model with no vision tower is the client's error too:
        a dropped image returning 200 is the exact failure this whole path exists to end.

        ⭐⭐ #890 (patch 0018): ONE IMAGE AT A TIME. The whole ``[N, P_max, D]`` batch used to
        cross to the device in a single ``.to()`` -- #871 measured 2.21 GiB of float32 landing
        before the tower could split anything, which ``FREETOKEN_VIT_GROUP=1`` cannot reach
        because it is downstream of the copy. See ``scheduler/mm_encode.py``.

        ⭐⭐ #871 (patch 0022): the ``except`` below is the only early return here whose outcome
        can differ per rank -- it depends on that rank's card at that instant, not on the request.
        It no longer changes control flow on its own: every rank all-reduces its own outcome and
        acts on the agreed one, so a per-rank OOM is a refused request on a live row, which is
        what the ``noqa`` comment always intended. 0018 made it far less likely to be REACHED;
        this is what makes reaching it survivable.

        ⭐ #892 (patch 0019): ``skip_images`` is the LEADING run of images the prefix match taken
        at admission already holds. The caller passes the reservation's count; k == N never
        arrives here (the reservation answers ``holds_every_image`` and the encode is skipped
        outright), because the tower cannot be asked for a zero-row result of its own hidden size.
        """
        encode = getattr(self.engine.model, "encode_images", None)
        if encode is None:
            return ErrorReplyMsg(
                uid=msg.uid,
                error=(
                    "this model does not accept images; retry without them "
                    "(the server was started without a vision tower)"
                ),
                code="invalid_request_error",
            )
        from freetoken.scheduler.mm_encode import encode_one_at_a_time
        from freetoken.scheduler.rank_agreement import any_rank_failed

        local_error: ErrorReplyMsg | None = None
        try:
            msg.mm_embeds = encode_one_at_a_time(
                encode,
                msg.pixel_values,
                msg.image_position_ids,
                self.device,
                msg.image_patch_counts,
                skip_images=skip_images,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad image must not take the engine down
            # ⛔ NOT `warning_rank0`. This is the one event in this path that happens on ONE card,
            # so rank-0 gating means a rank-1 OOM is logged NOWHERE -- and that is precisely the
            # case whose cause would otherwise be unrecoverable. #871 was rank 0, which is the
            # only reason it was diagnosable at all. The formatter already carries `rank=N`.
            logger.warning("vision encode failed for request %d: %r", msg.uid, exc)
            local_error = ErrorReplyMsg(
                uid=msg.uid,
                error=f"could not encode the image: {exc}",
                code="invalid_request_error",
            )
        # ⭐⭐ #871 (patch 0022): AGREE, THEN ACT. Every branch above this line is decided by the
        # request and is therefore identical on every rank; the `except` is not, and a rank that
        # refuses alone leaves its peer blocked in the embedding all-reduce until NCCL's watchdog
        # takes the process down 60 s later. ⛔ Both outcomes reduce -- a collective reached only
        # on failure would be the same desync moved one function earlier.
        if not any_rank_failed(
            local_error is not None, self.config.tp_info.size, self._reduce_failure_flag
        ):
            return None
        if local_error is not None:
            return local_error
        # This rank's encode succeeded and a peer's did not. Say so: without this line, this
        # rank's log shows a refusal with no cause on it, and reading the two ranks side by side
        # is how a desync gets diagnosed at all.
        logger.warning(
            "request %d refused in agreement: this rank encoded its images, another rank did not",
            msg.uid,
        )
        # Drop the embeddings before refusing:
        # the tower's output is on the card, the request never becomes a `Req`, and so nothing
        # downstream will ever free it -- while the retry this refusal invites needs that room.
        msg.mm_embeds = None
        return ErrorReplyMsg(
            uid=msg.uid,
            error=(
                "could not encode the image: another tensor-parallel rank failed to encode it "
                "(most often that card was momentarily out of memory); retry"
            ),
            code="invalid_request_error",
        )

    def _reduce_failure_flag(self, flag: torch.Tensor) -> None:
        """MAX all-reduce the agreement flag, in place, over the TP CPU group — with a deadline.

        ⭐ `tp_cpu_group` is gloo in BOTH branches of `engine._init_communication`, so this never
        touches the NCCL path. `scheduler/io.py`'s receive loop already broadcasts over the same
        group every iteration, which is what prices this as cheap.

        ⭐⭐ #871 (patch 0022), bullet 5. The group this runs on does NOT carry the row's own
        timeout. On the served TP=2 path `engine._init_communication` builds it with
        `new_group(backend="gloo")` and no `timeout=`, so it takes torch's default —
        `default_pg_timeout`, measured at **30 minutes** on the pinned torch (2.11.0+rocm7.14.0),
        not the `distributed_timeout: float = 60.0` that the process group beside it was
        initialised with (`engine/config.py`; ⛔ that field has no CLI flag). The wait below
        carries that 60 s explicitly. It is the same field NCCL's watchdog is armed with, so a
        peer this rank gives up on is one the watchdog would have given up on too — and 60 s is
        three orders of magnitude more than a small host all-reduce needs, while the encode that
        can delay a peer's arrival runs one image at a time (patch 0018) on a soft-token-capped
        row.

        ⭐ Both failure shapes end the same way — a logged refusal:

        - a peer that is **gone** raises immediately (gloo: `Connection closed by peer`) -- at the
          enqueue or at the wait, so BOTH are inside the guard -- which
          without this `except` would leave `_process_one_msg` and then `run_forever` — which
          catches `KeyboardInterrupt` and nothing else — killing this rank on a traceback that
          reads as a distributed bug rather than as the peer's death;
        - a peer that is **hung** raises `Operation timed out!` at the deadline instead of
          parking this rank for the half hour.

        ⛔ Fail CLOSED. On any error this rank marks the request failed, which is the only safe
        answer: an all-reduce is symmetric, so a peer that did not answer this one is not
        entering the forward on the strength of it either, and refusing can never leave a rank
        in the forward alone — which is the whole of #871.

        ⚠ This is belt-and-braces, not correctness. After bullets 1–4 a rank can reach the
        agreement and fail to be met only if its peer died hard, and the backend supervisor
        already reports that (`Backend supervisor: backend worker exited`). ⛔ Nor does a fired
        deadline repair the group: the abandoned all-reduce stays queued in gloo, so a peer that
        arrived late would match THIS op and leave every later collective one behind. That is
        acceptable only because the deadline is unreachable unless the peer is already gone.
        This converts a silent park into a logged refusal on the way down; it is not a recovery.
        """
        deadline = self.config.distributed_timeout
        try:
            # ⛔ The ENQUEUE is inside the guard, not just the wait. A peer that is already gone
            # can surface from `all_reduce` ITSELF, before there is a handle to wait on, and an
            # error escaping here reaches `run_forever` exactly as an escaping wait would --
            # making 0022 the new way to lose the row that this `except` exists to prevent.
            work = torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group, async_op=True
            )
            work.wait(timedelta(seconds=deadline))
        except Exception as exc:  # noqa: BLE001 -- a peer that never answers must not raise here
            # ⛔ NOT `warning_rank0`: which rank lost its peer is the whole content of the line,
            # and rank 1 losing rank 0 would otherwise be logged nowhere (bullet 3's principle).
            logger.error(
                "no answer from a tensor-parallel peer on the vision failure agreement "
                "(deadline %.1fs): %r -- refusing the request",
                deadline,
                exc,
            )
            flag.fill_(1)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            reservation = None
            if msg.pixel_values is not None or msg.mm_embeds is not None:
                # The ceiling binds both arrivals; only the pixels still need the tower run.
                # ⚠ It refuses BEFORE anything below reserves, so a refused request never holds
                # a lock to give back.
                error = self._multimodal_ceiling_error(msg)
                if error is None and msg.pixel_values is not None:
                    # The prefix-cache key stream: real ids with each picture's first placeholders
                    # replaced by pixel-hash markers, so a session with a picture in it keeps its
                    # prefix and a different picture can never hit the first one's KV. Derived on
                    # every rank from the same pixels; None keeps the old bypass (mm_key.py).
                    # ⭐ #892 (patch 0019): derived HERE, above the tower run, because the match
                    # keys on it and the whole point of the patch is to know the match BEFORE
                    # the tower runs. It reads only the pixels, so hoisting it changes nothing
                    # about the key itself.
                    from freetoken.scheduler.mm_key import image_cache_key_ids

                    msg.cache_key_ids = image_cache_key_ids(
                        msg.input_ids, self.prefill_manager.image_token_id,
                        msg.pixel_values, msg.image_position_ids,
                        msg.image_patch_counts,
                    )
                    if msg.cache_key_ids is None:
                        logger.warning_rank0(
                            "request %d: image prompt without a derivable cache key "
                            "(placeholder runs != images); prefix cache bypassed", msg.uid,
                        )
                    # ⭐⭐ #892: match the prefix and LOCK it, then run the tower on what the
                    # match does not already hold. llama.cpp has skipped this encode since it
                    # began keying chunks by hash; #843 priced FreeToken's re-encode at 0.27 s
                    # per image per turn, linear in N.
                    reservation = self.prefill_manager.reserve_prefix(msg)
                    if reservation is not None and reservation.holds_every_image:
                        # ⛔ k == N: the tower has nothing to run, and must not be asked for a
                        # zero-row result of its own hidden size. Sound because no placeholder
                        # falls in the extend region -- no chunk asks for a row -- and the
                        # request still has a KEY, so this is not the #791 cache bypass.
                        msg.mm_embeds = None
                    else:
                        error = self._attach_mm_embeds(
                            msg,
                            skip_images=0 if reservation is None else reservation.skip_images,
                        )
                if error is not None:
                    # #892: the request never becomes one, so its reservation gives the lock
                    # back here -- nothing downstream will, there being no Req yet.
                    self.prefill_manager.release_reservation(reservation)
                    self.send_result([error])
                    return
                if msg.pixel_values is not None:
                    # ⭐ #890: the pixels have done both jobs they came for -- the tower has run
                    # and the cache key is derived -- and `add_one_req` never carries them into
                    # `PendingReq`. Dropping the reference here bounds the SCHEDULER's host peak
                    # to one request's pixels even when a batch admits several image requests,
                    # instead of holding every one of them until the batch loop ends.
                    # ⚠ This is the scheduler's decoded copy only. #876/#883's residue is the
                    # TOKENIZER WORKER's, and no request-lifecycle event returns that one -- see
                    # `multimodal._pack_batch`.
                    msg.pixel_values = None
                    msg.image_position_ids = None
                    msg.image_patch_counts = None
            self.prefill_manager.add_one_req(msg, reservation)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # ── #801 bullet 8: stage the verify step ─────────────────────────────────────────
        # ⛔⛆ FIRST, before `pad_batch`. The staging advances each request's `device_len` to
        # `cached_len + T`, and EVERY consumer downstream derives the verify shape from that one
        # field: `_make_positions` and `_make_input_tuple` here, `attention/linear.py::
        # build_fla_metadata`, `attention/qsa_sparse.py`, and `engine/graph.py::_uniform_width`
        # -- which `pad_batch` itself asks, through `can_use_cuda_graph`. Staged after it, a
        # verify batch would be routed as a T=1 step and REPLAY THE T=1 GRAPH, which admits it
        # and does not error.
        # ⛔ MODEL-AGNOSTIC, like every other #801 hook in an engine file: a model without
        # `mtp_stage_verify` gets this method exactly as the image ships it, and so does a model
        # whose verify dial is off -- the hook answers `None` and touches nothing.
        _ft801_stage = getattr(self.engine.model, "mtp_stage_verify", None)
        if _ft801_stage is not None:
            _ft801_stage(batch, self.token_pool)
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. Each req carries the
        rows for ITS OWN CHUNK only (``PrefillAdder`` slices them per chunk; a chunk with no
        placeholder carries an empty tensor), so the concatenation matches the batch's ids
        one row per slot. ``req.mm_embeds`` is kept (not cleared) so the cache manager can
        recognize multimodal requests and keep them out of the shared prefix cache (image
        placeholders share a token id but carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _ft801_published_tokens(batch: Batch, next_tokens_cpu: torch.Tensor) -> list:
    """#801 bullet 8: per request, the tokens this drained forward actually committed.

    ⛔ MODEL-AGNOSTIC: it reads the step the model staged on the batch, the same carrier
    `models/qwen4_exp/gdn.py` uses for `batch.linear_snapshots`. An unstaged batch -- every
    prefill, every flag-off row, every no-draft decode step -- is one token per request, which is
    what the image does and what it costs.
    """
    staged = getattr(batch, "ft801_verify_step", None)
    if staged is None:
        return [next_tokens_cpu[i : i + 1] for i in range(len(batch.reqs))]
    from freetoken.models.qwen4_exp.spec import published_tokens

    return published_tokens(batch, next_tokens_cpu)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # ── #801 bullet 8: one row per TOKEN when a verify step is staged ────────────────────
    # ⛔⛆ The one builder that does NOT generalise for free. `_make_positions` and
    # `_make_input_tuple` are written in terms of `Req.extend_len` and produce the verify step's
    # layout unmodified; this one emits one row per REQUEST at `device_len`, which on a T=2 step
    # is one past the BONUS row's input. Kept as-is, a verify step writes the verify row's token
    # over the bonus slot and drops the bonus entirely -- and at bs == 1 the scatter BROADCASTS
    # rather than raising, so the row serves, slightly wrong, in silence.
    _ft801_staged = getattr(batch, "ft801_verify_step", None)
    if _ft801_staged is not None:
        _ft801_plan = _ft801_staged.plan
        _ft801_rows = [batch.reqs[r] for r in _ft801_plan.token_to_req]
        mapping_host = torch.tensor(
            [req.table_idx for req in _ft801_rows], dtype=torch.int64, pin_memory=True
        )
        write_host = torch.tensor(
            [
                (position if req.can_decode else -1)
                for req, position in zip(_ft801_rows, _ft801_plan.write_positions)
            ],
            dtype=torch.int64,
            pin_memory=True,
        )
        return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
