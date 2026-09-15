# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `engine/graph.py` from image `llm-server/freetoken-gfx1201:2026-09-09-agree-0022`
# (md5 8bc02cd8e10dfbf5239bae56292c50d0, 217 lines) BIND-MOUNTED over the installed package, plus
# this block, a `width` on `GraphCaptureBuffer`, `_uniform_width`, a second graph set and the
# routing that picks between them. ⛔ In no image and in no Dockerfile ladder -- it must be in
# `arm_mtp_801.sh`'s OVERLAY and ORIGS lists or the row runs the image's copy silently (#866).
#
# ⛔⛆ WHY IT EXISTS (round 6 bullet 6). `can_use_cuda_graph` is `batch.is_decode and batch.size <=
#   max_graph_bs`, and a #801 verify batch satisfies it -- so a T=2 step would REPLAY THE T=1
#   GRAPH. Three more things fall out of the graph at T=2: this file's buffers are `[bs]`, its
#   `set_batch` builds its OWN `FLAMetadata` with a constant `arange(bs + 1)` (so bullet 5's
#   `attention/linear.py` overlay never reaches the captured path), and `attention/qsa_sparse.py`
#   refuses the step outright. Without a capture, #701's eager tax lands on every step and round
#   5's projection is void.
#
# ⛔ MODEL-AGNOSTIC, DELIBERATELY. Everything #801 asks of this file arrives through three
#   DUCK-TYPED attributes on the model -- `mtp_verify_width`, `mtp_verify_graph_max_bs` and
#   `mtp_reserve_verify_buffers` -- the same hook shape `engine.py`'s own
#   `hasattr(self.model, "mtp_shadow_step")` uses. A model without them gets today's runner
#   exactly: no second graph set, no second buffer, no widened anything.
#
# ⛔⛆ THE GUARD, AND IT IS NOT A SHORTCUT (upstream #173). A captured graph BAKES ITS ROW COUNT,
#   so every padded request must forward the same T. `pad_batch` pads with the one-token
#   `dummy_req`, which would make any padded verify batch RAGGED. So a verify step is NEVER
#   PADDED -- `padded_size == size` -- and it uses a graph only when its own size is a captured
#   one. A ragged batch (one row speculating beside one that is not, which `spec.plan_verify`
#   calls normal) runs EAGER. That is a real limitation; `test_graph_capture_801.py
#   ::TestRoutingRefusesWhatItCannotCapture` pins every refusal rather than leaving it to be
#   discovered as a tok/s number.

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


def _uniform_width(batch: Batch) -> int:
    """#801: rows this batch forwards per request, or ``0`` when they differ.

    ⚠ Read off `Req.extend_len`, the same field every other builder this round touched is written
    in terms of (`scheduler.py::_make_positions`, `attention/linear.py::build_fla_metadata`). ⛔
    `padded_reqs` when it exists and `reqs` otherwise: `scheduler._prepare_batch` calls
    `pad_batch` FIRST, and `pad_batch` itself asks `can_use_cuda_graph` -- at which point
    `padded_reqs` is an unset `field(init=False)` and reaching for it is an `AttributeError` on
    every decode step.
    """
    reqs = getattr(batch, "padded_reqs", None) or batch.reqs
    widths = {req.extend_len for req in reqs}
    return widths.pop() if len(widths) == 1 else 0


def _widen_dummy(dummy: Req, width: int) -> Req:
    """#801: the placeholder a VERIFY capture runs on -- `dummy_req` with ``extend_len == width``.

    ⛔ A COPY. The one-token graphs pad with the original on every replay, and a mutated
    `device_len` there would make each padded row two positions long and break the T=1 row count.
    ⚠ `_alloc_ids_buf` already sized `_ids_buf` to `max_device_len`; the capture never reads the
    ids (``set_batch`` replaces `batch.input_ids` with the static buffer), only `cached_len` /
    `device_len`, which is what the position and page-table builders walk.
    """
    req = copy.copy(dummy)
    req.cached_len = 0
    req.max_device_len = max(dummy.max_device_len, width)
    req.device_len = width
    return req


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr; a constant per captured (bs, width), filled once.
    # #801: `arange(bs+1) * width` -- `arange(bs+1)` IS its one-token special case.
    fla_cu_seqlens: torch.Tensor
    #: #801: rows per request this buffer was built for. 1 = today's decode step.
    width: int = 1

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, width: int = 1
    ) -> GraphCaptureBuffer:
        # ⛔ FLAT tensors are per TOKEN, `table_idx` and the indptr per REQUEST. Getting that
        #    backwards is how a T=2 step reads request 1's recurrent state for request 0's
        #    second token -- silently, because the shapes still broadcast at bs == 1.
        rows = bs * width
        return GraphCaptureBuffer(
            input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            out_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            positions=torch.zeros(rows, dtype=torch.int32, device=device),
            logits=torch.empty(rows, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device) * width,
            width=width,
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        bs = batch.padded_size
        _slice = slice(bs * self.width)
        _reqs = slice(bs)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_reqs]
        # Decode GDN metadata reads the persistent cu_seqlens (constant per width) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_reqs]
        )

    def copy_from(self, batch: Batch) -> None:
        bs = batch.padded_size
        rows = bs * self.width
        # ⛔ #801: without this the mismatch surfaces as a torch shape error naming neither the
        #    width nor the graph -- or, at width 1 with a 1-row buffer, not at all.
        assert batch.input_ids.numel() == rows, (
            f"graph buffer width {self.width} expects {rows} rows for {bs} requests, "
            f"but this step forwards {batch.input_ids.numel()}"
        )
        _slice = slice(rows)
        _reqs = slice(bs)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_reqs] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        # ── #801 round 6 bullet 6: the verify graph set ───────────────────────────────────────
        # ⛔ VRAM, and it is not small. `models/qwen4_exp/gdn.py`'s intermediate-states buffer is
        #   ~1.5 MiB per (request, step) per rank across 36 GDN layers, so a verify set reaching
        #   the T=1 set's own `max_graph_bs` (160 by default) would be tens of GiB. The MODEL
        #   names the cap, because the model is what knows the row it serves.
        self.verify_width = int(getattr(model, "mtp_verify_width", 1) or 1)
        verify_max_bs = int(getattr(model, "mtp_verify_graph_max_bs", 0) or 0)
        self.verify_graph_bs: List[int] = (
            [bs for bs in self.graph_bs_list if bs <= verify_max_bs]
            if self.verify_width > 1
            else []
        )
        self._verify_bs: frozenset = frozenset(self.verify_graph_bs)
        self.verify_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_buffer: GraphCaptureBuffer | None = None
        self.verify_dummy_req: Req | None = None
        # ── #801 round 6 bullet 9s: the memos a captured forward recorded ────────────────────
        # ⛔⛆ `replay()` does NOT call the model's python forward -- it refills the static
        #   buffers and calls `g.replay()` -- so the assignments a verify forward makes onto the
        #   batch in python happen exactly once, at capture, and a replayed step reaches its
        #   commit carrying none of them (bullet 9q, three loads). The DATA is fine: every field
        #   is a static graph-pool tensor the replay refills. Only the binding is lost, so it is
        #   recorded once per captured size here and re-bound on each replay.
        # ⭐ Keyed by bs, exactly like `verify_graph_map`: each size has its own views.
        # ⛔ MODEL-AGNOSTIC, like `mtp_verify_width` above -- the MODEL names the memos.
        self.verify_memos: Dict[int, dict] = {}
        self._restore_memos = getattr(model, "mtp_restore_memos", None)
        # ── #801 round 6 bullet 9x: the oracle's steps have to run EAGER ──────────────────────
        # ⛔⛆ 9q's lesson, one bullet later: `replay()` never calls the model's python forward, so
        #   `models/qwen4_exp/gdn.py`'s instrument is silent on exactly the steps that matter --
        #   `can_use_cuda_graph` admits a verify batch from the very first one. While the budget
        #   remains, the router declines and the step runs eager so the python actually runs.
        # ⛔ MODEL-AGNOSTIC, like the five names above, and the same absent-safe `getattr`.
        # ⛔ A CALLABLE, not a value: the budget empties mid-load and the capture must come back.
        self._gdncheck_pending = getattr(model, "mtp_gdncheck_pending", None)
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # ⛔ #801: the backend sizes its per-TOKEN static scratch by `max_bs * verify_width`, and
        #   `init_capture_graph` is called ONCE for both sets -- so the width has to be on the
        #   backend before it. Duck-typed for the same reason the model hooks are: no other
        #   backend in this engine knows what a verify step is.
        if self.verify_width > 1:
            self.attn_backend.verify_width = self.verify_width
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pool = self._capture_set(
            self.graph_bs_list, self.buffer, self.graph_map, model, self.dummy_req, None
        )

        if self.verify_graph_bs:
            # ⛔⛆ BEFORE any verify capture, never lazily inside one. An allocation made DURING
            #   `torch.cuda.graph(...)` comes out of THAT graph's private pool -- right on the
            #   eager warm-up, wrong on every replay. `engine.py`'s `mtp_reserve_graph_buffers`
            #   comment makes the same point about relying on capture order visiting the largest
            #   bs first; this is the reservation instead of the reliance.
            reserve = getattr(model, "mtp_reserve_verify_buffers", None)
            if reserve is not None:
                reserve(max(self.verify_graph_bs), self.verify_width)
            self.verify_dummy_req = _widen_dummy(self.dummy_req, self.verify_width)
            self.verify_buffer = GraphCaptureBuffer.init(
                max(self.verify_graph_bs), vocab_size, self.device, self.verify_width
            )
            pool = self._capture_set(
                self.verify_graph_bs,
                self.verify_buffer,
                self.verify_graph_map,
                model,
                self.verify_dummy_req,
                pool,
            )

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_set(
        self,
        bs_list: List[int],
        buffer: GraphCaptureBuffer,
        graph_map: Dict[int, torch.cuda.CUDAGraph],
        model: BaseLLMModel,
        dummy_req: Req,
        pool,
    ):
        """One capture pass over ``bs_list`` into ``graph_map``. #801: extracted verbatim from the
        loop this file already had, so the one-token pass runs the same statements in the same
        order; the verify pass differs only in its buffer, its dummy and its graph map."""
        width = buffer.width
        label = "decode" if width == 1 else f"verify T={width}"
        pbar = tqdm(
            sorted(bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = (
                f"Capturing graphs ({label}): bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            )
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (dummy_req.linear_slot_idx
                          if dummy_req.linear_slot_idx is not None
                          else dummy_req.table_idx)
            buffer.table_idx[:bs].fill_(dummy_slot)
            rows = bs * width
            with get_global_ctx().forward_batch(batch):
                buffer.logits[:rows] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    buffer.logits[:rows] = model.forward()
                self._reset_moe_offload_cache()
            if width > 1:
                # ⛔⛆ #801 bullet 9s, and the ORDER is load-bearing: AFTER the `torch.cuda.graph`
                #   block, so what is recorded is the CAPTURED run's tensors. The eager warm-up
                #   above allocates from the ordinary allocator and no replay ever writes those
                #   addresses again -- binding them would commit the DUMMY request's state on
                #   every step. ⚠ The same `batch` object serves both forwards, so the captured
                #   run has already overwritten the warm-up's entries by the time this runs.
                record = getattr(model, "mtp_capture_memos", None)
                if record is not None:
                    self.verify_memos[bs] = record(batch)
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            graph_map[bs] = graph
        return pool

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if not batch.is_decode:
            return False
        width = _uniform_width(batch)
        if width == 1:
            return batch.size <= self.max_graph_bs
        # ⛔⛆ #801 round 6 bullet 9x: the GDN-output oracle's budgeted steps run EAGER, or the
        #   instrument reports on nothing. A DIAGNOSTIC decline and a real cost -- an instrumented
        #   load measures the eager verify path -- so it is stated here rather than discovered as
        #   a tok/s number. ⛔ BELOW the `width == 1` return: a plain decode step has nothing for
        #   the oracle to read, and declining one would drop every non-speculating step in the
        #   load onto the eager path (#701's tax) for no report at all.
        if self._gdncheck_pending is not None and self._gdncheck_pending():
            return False
        # ⛔ #801 + upstream #173: a verify step is never padded, so ``batch.size`` ITSELF has to
        #   be a captured size. A ragged batch has width 0 and lands here too -- eager.
        return width == self.verify_width and batch.size in self._verify_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        width = _uniform_width(batch)
        buffer, graph_map = (
            (self.buffer, self.graph_map)
            if width == 1
            else (self.verify_buffer, self.verify_graph_map)
        )
        buffer.copy_from(batch)
        g = graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        if width > 1 and self._restore_memos is not None:
            # ⛔⛆ #801 bullet 9s. The kernels this graph just ran wrote their state into static
            #   tensors; `verify_step`'s commit needs the python objects that say WHERE. They
            #   were recorded at capture and the replay refilled exactly those tensors.
            # ⛔ A `.get(...)` default here would be the `or {}` bullet 9r just removed one level
            #   down: `can_use_cuda_graph` admits a verify batch only at a size captured EXACTLY,
            #   so a missing row means the capture pass skipped one and nothing else will say so.
            memos = self.verify_memos.get(batch.padded_size)
            if memos is None:
                raise RuntimeError(
                    f"#801: replaying the verify graph for bs={batch.padded_size} with no memo "
                    "set recorded for it. The captured forward's snapshots are the only thing "
                    "that tells this step's commit where its GDN and PLE state was written, and "
                    "without them the commit reaches the row empty. Captured sizes: "
                    f"{sorted(self.verify_graph_map)}; memo sizes: {sorted(self.verify_memos)}."
                )
            self._restore_memos(batch, memos)
        # #801: ONE ROW PER TOKEN, not per request. Trimming a verify step to `batch.size` would
        # keep the verify row and drop the bonus row (or the other way round, depending on the
        # layout) -- `engine.py::forward_batch` still does exactly that, and that is bullet 8.
        return buffer.logits[: batch.size * width]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        # ⛔ #801 + upstream #173. A captured graph BAKES its row count, so a verify step can only
        #   replay UNPADDED -- padding it with the one-token `dummy_req` would make the batch
        #   RAGGED, which is the one shape a captured graph cannot serve. `can_use_cuda_graph`
        #   above is what enforces it, by admitting a verify batch only at a size that was
        #   captured EXACTLY; this line is the LOCK on that reasoning rather than a second copy of
        #   it. ⚠ It cannot fire while the router holds -- which is the point: if a later change
        #   loosens the router, this fails loudly instead of replaying a graph built for another
        #   shape. `test_graph_capture_801.py` stubs the router True to prove it can.
        assert padded_size == batch.size or _uniform_width(batch) == 1, (
            f"#801: a verify step of {batch.size} requests was padded to {padded_size}; a "
            "captured verify graph is only valid unpadded (upstream #173)."
        )

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        # #801: the verify set shares the same mempool, so a map left behind pins all of it.
        self.graph_map = {}
        self.buffer = None
        self.verify_graph_map = {}
        self.verify_buffer = None
        # #801 bullet 9s: the memos hold GRAPH-POOL tensors, so a map left behind pins the
        # mempool exactly as a graph map does. Same reason, same line.
        self.verify_memos = {}
        gc.collect()
