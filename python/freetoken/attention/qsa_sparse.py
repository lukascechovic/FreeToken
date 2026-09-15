"""Qwen3.8-Flash-Next QSA compressed-block sparse attention backend.

Serves ``AttnType.QSA`` over ``kvcache/qsa_pool.py``: paged GQA K/V for the 12 full-attention
layers, a compressed index-key slab holding one key per ``index_ratio`` tokens, and a
per-request pending ring for the group a forward leaves open. The 36 GDN layers never reach
this backend, and the model has no dense attention layer, so :meth:`forward` is not served --
the only entry point is :meth:`qsa_forward` (``models/qwen4_exp/attention.py``).

One QSA layer's forward, all ragged over ``[T, ...]`` metadata:

1. store K/V at ``batch.out_loc``;
2. pool each row's closing group (members at positions >= ``cached_len`` come from this
   forward's raw index keys, the older ones from the pending ring), zero-centered rmsnorm it
   and rope it at the group's first position, then scatter it into the slab row
   ``out_loc // index_ratio`` (rows whose group does not close land on the request's scratch
   row and are never read);
3. store this forward's last ``ring_capacity`` raw index keys per request into the ring;
4. norm+rope the indexer queries at their own positions;
5. score every COMPLETE visible block (``sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)``,
   clamped to ``kvlen // index_ratio`` -- slab rows are never cleared, so stale rows must stay
   unreachable), take the top ``index_budget // index_ratio`` blocks, expand them to token
   indices plus the causal tail of the open group;
6. attend to exactly those tokens.

Addressing: the engine pins ``page_size == 64`` (this backend's ``page_sizes``), so a group of
``index_ratio`` tokens never straddles a page and ``block_table[req, p] = page_table[req, p *
64] // 64`` names both the K/V page and, viewed as ``page_size // index_ratio`` compressed
rows, the block's slab page. Decode stages that table plus the live lengths and table_idx into
static buffers (``prepare_for_replay``) so the whole path is CUDA-graph capturable.
"""

# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `attention/qsa_sparse.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 71c68ce977081fb55528f60b065c0375,
# 507 lines) BIND-MOUNTED over the installed package, plus this block and the four #801-marked
# edits below. ⛔ It is NOT a patch in the Dockerfile ladder and NOT in any image, so it must be
# in `arm_mtp_801.sh`'s OVERLAY manifest (and its ORIGS list) or the row runs the image's copy
# with nothing erroring (#866).
#
# ⛔⛆ WHY IT EXISTS. This backend DEFERS decode addressing and then rebuilds it as
#   ``token_to_req = arange(bs)`` / ``cu_seqlens = arange(bs + 1)`` -- one query row per request,
#   hard-coded. #801's verify step forwards TWO rows for a speculating request, and every QSA
#   kernel (`kernel/triton/qsa/{compress,score,expand,attend}.py`) indexes its per-request
#   tensors THROUGH that map. At bs >= 2 the stale map is a length mismatch and torch raises; at
#   bs == 1 every stale tensor has length 1 and broadcasts CLEANLY against the two-token ones, so
#   the step completes with the wrong ring mask and nothing anywhere errors -- and bs == 1 is the
#   single-stream shape this round measures. Gated in `test_qsa_metadata_801.py`
#   (`./check_qsa_801.sh`), against this file's own `.orig` rather than a re-derivation.
#
# ⚠ NO MARKER PRINT, and no behaviour change on the one-token step: `prepare_metadata` builds the
#   ragged host map ONLY when some request forwards more than one token, and `_snapshot_decode`
#   still takes the two aranges otherwise. The differential gate is what asserts the mount took.
#
# ⭐ ROUND 6 BULLET 6 WIDENED THE CAPTURED PATH. The per-TOKEN buffers are `[max_bs * width]` and
#   `_stage_decode` stages a uniform T-token step instead of refusing it; the per-REQUEST ones
#   (`block_table`, `kvlen`, `table_idx`) stay `[max_bs]` because every kernel reaches them
#   THROUGH `token_to_req`. ⛔ The refusal did not go away, it MOVED: `_stage_width` still fails
#   loudly on a ragged step or a width that was not captured, because `_stage_decode` is reached
#   through `prepare_for_replay` as well as through the router.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Block-score transient budget (vLLM's number): the fp32 [rows, n_blocks] logits tile is
# 256 KB per row at a 1M-token context, so a long prefill must be scored in row chunks.
_LOGITS_WORKSPACE_BYTES = 128 << 20


TORCH_TOPK_ENV = "FREETOKEN_QSA_TORCH_TOPK"


def _resolve_block_topk() -> Callable | None:
    """The in-repo Triton block top-k, or None to fall back on torch.topk."""
    if os.getenv(TORCH_TOPK_ENV, "0") == "1":
        logger.info(f"qsa_sparse block top-k: torch.topk ({TORCH_TOPK_ENV}=1)")
        return None
    try:
        from freetoken.kernel.triton.qsa import qsa_block_topk
    except Exception as exc:
        logger.info(f"qsa_sparse block top-k: torch.topk (triton unavailable: {exc})")
        return None
    logger.info("qsa_sparse block top-k: triton qsa_block_topk")
    return qsa_block_topk


def _token_to_req(bs: int, seqlens_q: List[int]) -> torch.Tensor:
    """``[T] int32`` query row -> request, pinned: the ragged map every QSA kernel indexes by.

    #801 bullet 4: lifted out of `prepare_metadata`'s prefill branch, which is where it was
    already written, because a verify DECODE step needs the same map -- ``arange(bs)`` is only
    its one-token special case.
    """
    return torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32),
        torch.tensor(seqlens_q, dtype=torch.int32),
    ).pin_memory()


@dataclass
class QSASparseMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:        bool
    last_indices:     torch.Tensor  # gpu
    qo_indptr_cpu:    torch.Tensor  # cpu pinned int32 [bs+1]
    kv_len_cpu:       torch.Tensor  # cpu pinned int32 [bs]
    # #801: T, this forward's query rows. Equals bs on a one-token decode step; a verify step
    # forwards two rows for every speculating request, and that is the whole difference.
    num_tokens:       int
    # Ragged per-token / per-request addressing. Decode defers these to the static graph
    # buffers (prepare_for_replay) or to a lazy eager snapshot at the first QSA layer.
    token_to_req:     torch.Tensor | None = None  # [T] int32
    # #801: the host side of the above, built only when T != bs. `is None` IS the decode fast
    # path's discriminator -- see `_snapshot_decode`.
    token_to_req_cpu: torch.Tensor | None = None  # cpu pinned int32 [T]
    cu_seqlens:       torch.Tensor | None = None  # [bs+1] int32
    seq_lens:         torch.Tensor | None = None  # [bs] int32, device_len
    ring_slots:       torch.Tensor | None = None  # [bs] int32, Req.table_idx
    block_table:      torch.Tensor | None = None  # [bs, W//page_size] int32, physical page ids
    # Per-forward scatter plans, built once by the first QSA layer and reused by the rest.
    # positions is bound here (not in prepare_metadata) because a capture batch has none yet.
    cmp_rows:         torch.Tensor | None = None  # [T] int32, compressed slab destination
    ring_rows:        torch.Tensor | None = None  # [T] int32, flat ring row or -1
    positions:        torch.Tensor | None = None  # [T] int32, logical query positions
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        args = config.qwen4_args
        assert args is not None, "qsa_sparse backend needs ModelConfig.qwen4_args"
        self.head_dim = config.head_dim
        self.index_heads = args.index_n_heads
        self.token_topk = args.index_budget
        self.kvcache = get_global_ctx().kv_cache
        assert isinstance(self.kvcache, QSAKVCache), (
            f"qsa_sparse backend needs a QSA pool, got {type(self.kvcache).__name__}"
        )
        self.device = self.kvcache.device
        self.dtype = self.kvcache.dtype
        self.index_head_dim = self.kvcache.index_head_dim
        self.ratio = self.kvcache.index_ratio
        self.ring_capacity = self.kvcache.ring_capacity
        self.page_size = get_global_ctx().page_size
        assert self.page_size % self.ratio == 0, (
            f"QSA needs page_size ({self.page_size}) divisible by index_ratio ({self.ratio})"
        )
        self.cmp_page_size = self.page_size // self.ratio
        self.block_topk = self.token_topk // self.ratio
        self.select_width = self.token_topk + self.ratio - 1
        assert self.token_topk % self.ratio == 0, "QSA budget must be a whole number of blocks"
        # The sparse attend kernel bakes 1/sqrt(head_dim) into its exp2 scale.
        assert config.attn_sm_scale in (None, self.head_dim**-0.5), (
            "qsa_sparse serves the default 1/sqrt(head_dim) attention scale only"
        )
        # QSA layer -> index slab slot, in sparse-layer order (the pool's own convention).
        group = self._qsa_group(config)
        self._idx_slot = {lid: i for i, lid in enumerate(group.layer_ids)}
        self.rotary_config = group.rotary_config
        self._index_cos_sin: torch.Tensor | None = None

        self._block_topk_kernel = _resolve_block_topk()
        # decode staging (static buffers under CUDA graphs; eager decode snapshots per step)
        self._graph: dict[str, torch.Tensor] = {}
        self.capture_bs: List[int] = []

    @staticmethod
    def _qsa_group(config: ModelConfig):
        from freetoken.models.config import FullAttentionGroupConfig

        groups = [
            g
            for g in config.attention_groups
            if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1
        ]
        assert len(groups) == 1, f"expected one QSA attention group, got {len(groups)}"
        return groups[0]

    # ----- slab views ---------------------------------------------------------------------
    def _cmp_pages(self, slot: int) -> torch.Tensor:
        """The compressed slab as ``[pages, page_size // ratio, 1, dim]``, the score kernel's
        paged layout. The scratch rows past ``cmp_scratch_base`` stay out of the view."""
        rows = self.kvcache.cmp_k_cache(slot)[: self.kvcache.cmp_scratch_base]
        return rows.view(-1, self.cmp_page_size, 1, self.index_head_dim)

    def _index_rope_cache(self) -> torch.Tensor:
        """cos/sin table of the indexer rope: same rotary_dim and frequencies as the main
        attention, ``head_size`` 128 instead of 256, so it is a separate get_rope instance.

        The table itself (not RotaryEmbedding.forward) because the indexer's norm+rope is one
        fused kernel and the compressed keys rope at their group's position, not the query's."""
        if self._index_cos_sin is None:
            from freetoken.layers.rotary import get_rope

            rotary = self.rotary_config
            with torch.device(self.device):
                rope = get_rope(
                    head_dim=self.index_head_dim,
                    rotary_dim=rotary.rotary_dim,
                    max_position=rotary.max_position,
                    base=rotary.base,
                    rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
                )
            self._index_cos_sin = rope._cos_sin_cache.to(self.device)
        return self._index_cos_sin

    # ----- metadata -----------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        md = QSASparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
            num_tokens=sum(seqlens_q),  # #801
        )
        batch.attn_metadata = md
        if not is_decode:
            table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
            token_to_req = _token_to_req(len(reqs), seqlens_q)  # #801: was inline here
            md.cu_seqlens = qo_indptr.to(self.device, non_blocking=True)
            md.token_to_req = token_to_req.to(self.device, non_blocking=True)
            md.seq_lens = kv_len.to(self.device, non_blocking=True)
            md.ring_slots = table_idx.to(self.device, non_blocking=True)
            md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        elif md.num_tokens != len(reqs):
            # #801: a DECODE step forwarding more than one token for some request -- the verify
            # step. The deferred one-token path below reads `arange(bs)`, which is not this
            # step's map; build the ragged one here, on the host, where the query lengths
            # already exist. Guarded on the count so the every-served-row step pays nothing.
            md.token_to_req_cpu = _token_to_req(len(reqs), seqlens_q)
        # Decode addressing is DEFERRED: a graph-bound step stages it into the static
        # buffers (prepare_for_replay), an eager step snapshots at the first QSA layer.

    def _block_base_view(self) -> torch.Tensor:
        """Every-``page_size``-th column of the page table: the per-page base slots. A strided
        VIEW, so gathering rows through it materializes only [bs, W/page_size]."""
        return get_global_ctx().page_table[:, :: self.page_size]

    def _block_table(self, table_idx: torch.Tensor) -> torch.Tensor:
        return (self._block_base_view().index_select(0, table_idx) // self.page_size).to(
            torch.int32
        )

    def _stage_decode(self, md: QSASparseMetadata, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the metadata
        at them (restage-per-replay, m3/dsa precedent).

        ``table_idx`` is one page-table row per REQUEST, or this step's per-ROW tensor
        (`Batch.active_table_idx`) to take them from -- see `_request_rows`.

        ⭐ #801 round 6 bullet 6: the token->request map and the query indptr are CONSTANT for a
        given width (uniform T ⇒ ``arange(rows) // T`` and ``arange(bs + 1) * T``), so both
        widths are filled once in `init_capture_graph` and neither is ever copied here. The
        per-REQUEST buffers below are the only per-step writes, exactly as before.
        """
        width = self._stage_width(md, bs)
        table_idx = self._request_rows(table_idx, bs, width)
        self._graph["block_table"][:bs].copy_(
            self._block_base_view().index_select(0, table_idx) // self.page_size
        )
        self._graph["kvlen"][:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        self._graph["table_idx"][:bs].copy_(table_idx)
        md.block_table = self._graph["block_table"][:bs]
        md.seq_lens = self._graph["kvlen"][:bs]
        md.ring_slots = self._graph["table_idx"][:bs]
        if width == 1:
            md.token_to_req = self._graph["token_to_req"][:bs]
            md.cu_seqlens = self._graph["cu_seqlens"][: bs + 1]
        else:
            md.token_to_req = self._graph["token_to_req_verify"][: bs * width]
            md.cu_seqlens = self._graph["cu_seqlens_verify"][: bs + 1]

    def _request_rows(self, table_idx: torch.Tensor, bs: int, width: int) -> torch.Tensor:
        """One page-table row per REQUEST, from a tensor that may carry one per ROW.

        ⛔⛆ #801 round 6 bullet 9d -- THE FIRST LOAD'S OWN FINDING, and the one thing no desk
        gate could have caught, because none of them called `prepare_for_replay`. The captured
        verify replay reaches here through `prepare_for_replay`, which passes
        `Batch.active_table_idx`; `scheduler._make_input_tuple` fills that with ``req.table_idx``
        once per TOKEN (``extend_len`` copies each). At width 1 rows and requests are the same
        count -- every row this box has ever served -- so the mismatch first exists on a verify
        step, where it died in the `copy_` below with *"output with shape [1, 4096] doesn't match
        the broadcast shape [2, 4096]"*: one request, two rows.

        ⭐ `init_capture_graph` states the rule being violated IN ITS OWN COMMENT -- block_table /
        kvlen / table_idx are per REQUEST and must NOT grow with the width, because every QSA
        kernel reaches them THROUGH `token_to_req`. Bullet 6 wrote it for the buffers and did not
        apply it to the one argument that crosses the boundary into them.

        ⭐ A request's rows all carry ITS table_idx, so column 0 IS ``[r.table_idx for r in
        padded_reqs]`` -- the spelling the eager path (`_snapshot_decode`, right all along) builds
        on the host. ⛔ At width 1 the tensor is returned UNTOUCHED, not rebuilt: the served decode
        step must not pay one extra op for a path it never takes.
        """
        if table_idx.numel() == bs:
            return table_idx
        assert table_idx.numel() == bs * width, (
            f"qsa_sparse: staging {bs} requests at width {width} wants {bs} or {bs * width} "
            f"page-table rows, not {table_idx.numel()} (#801 r6 b9d)."
        )
        return table_idx.view(bs, width)[:, 0]

    def _stage_width(self, md: QSASparseMetadata, bs: int) -> int:
        """#801: rows per request this step stages, checked against what was CAPTURED.

        ⛔ A captured graph bakes its row count and its indptr, so only a UNIFORM step whose width
        is the captured one can be staged. `engine/graph.py::GraphRunner.can_use_cuda_graph`
        already refuses everything else -- this is the second lock, because `_stage_decode` is
        also reached through `prepare_for_replay`, and bullet 4 put an assertion here precisely
        because the alternative is a replay of the wrong graph that reads as a working row.
        """
        if md.token_to_req_cpu is None:
            return 1
        width = int(getattr(self, "verify_width", 1) or 1)
        assert width > 1 and md.num_tokens == bs * width, (
            f"qsa_sparse: this decode step forwards {md.num_tokens} tokens for {bs} requests, "
            f"and the static graph buffers were built for width {width}. Only a uniform "
            "step of the captured width can replay; anything else must run eager (#801 r6 b6)."
        )
        assert md.qo_indptr_cpu.tolist() == [i * width for i in range(bs + 1)], (
            f"qsa_sparse: this decode step is ragged ({md.qo_indptr_cpu.tolist()}) and a captured "
            "graph bakes one indptr. A ragged verify batch must run eager (#801 r6 b6)."
        )
        return width

    def _snapshot_decode(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Eager decode (not graph-staged): this step's rows, once per forward. The live
        page-table row may mutate for the next batch while this one runs, so gather now."""
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        bs = len(reqs)
        table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
        md.ring_slots = table_idx.to(self.device, non_blocking=True)
        md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        md.seq_lens = md.kv_len_cpu.to(self.device, non_blocking=True)
        # #801: the one-token step -- every served row today -- keeps the two aranges verbatim.
        # A verify step took the ragged branch in `prepare_metadata` and moves that map instead;
        # its `cu_seqlens` is `qo_indptr`, which for T == 1 IS `arange(bs + 1)`.
        if md.token_to_req_cpu is None:
            md.token_to_req = torch.arange(bs, dtype=torch.int32, device=self.device)
            md.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
        else:
            md.token_to_req = md.token_to_req_cpu.to(self.device, non_blocking=True)
            md.cu_seqlens = md.qo_indptr_cpu.to(self.device, non_blocking=True)

    def draft_metadata(
        self, md: QSASparseMetadata, kv_lens: List[int]
    ) -> QSASparseMetadata:
        """A T=1 metadata for the MTP head's own forward: ONE query row per request.

        ⛔⛆ **#801 round 6 bullet 9e, and load 4 died on it.** `spec.shadow_step` forwards the head
        over one row per REQUEST (bullet 8c's `draft_rows`) while handing it the BACKBONE's batch.
        On a verify step that batch is one row per TOKEN, and the head's KV store raised
        ``store.cu:88, Size mismatch for L(shape#0): expected 1 but got 2``. ⛔ The store is only
        where it surfaced: `qsa_forward` reaches `token_to_req`, `cu_seqlens`, `seq_lens` and the
        scatter plan through this object, and every one of them is the backbone's width.

        ⭐ **Per-REQUEST fields are the backbone's own, taken not rebuilt.** `block_table` and
        `ring_slots` already carry one row per request (`init_capture_graph` says so in its own
        comment, and bullet 9d's crash came from breaking that rule) and the head only READS them.
        Rebuilding them would re-gather a page table the next batch's `allocate_paged` may already
        have moved -- the exact staleness `_snapshot_decode` exists to avoid.

        ⭐ **The scatter plan is deliberately left None.** `qsa_forward` rebuilds it when
        ``slot == 0 or md.cmp_rows is None``; the head is a single layer and need not be slot 0, so
        a carried plan would compress the head's K/V into the slab row of a token it never
        forwarded.

        ⛔⛆ **``kv_lens`` IS THE α TAX, and the caller owes it one length per REQUEST.** The
        backbone ran at ``device_len``; on a REJECTED step only ``cached_len + 1`` of that is real,
        **and the head's own KV layer was never written at the rejected slot** -- it writes one row
        per step, at the position it drafts from. `seq_lens` feeds `qsa_mqa_paged`'s visible-block
        count and `expand_qsa_block_indices`, so an inflated length lets a slot holding whatever
        the pool last left there compete for the head's top-k budget. Nothing raises; α reads low.
        `spec.draft_view` passes ``plan.positions[draft_row] + 1``, which is `device_len` exactly
        on a plain step and one short of it on a rejected one.

        ⛔ A NEW length tensor, never an edit in place: `_stage_decode` points `md.seq_lens` at the
        shared ``self._graph["kvlen"]`` buffer, and correcting that would be right for this step
        and wrong for the next replay.
        """
        bs = len(kv_lens)
        assert md.ring_slots is not None and md.block_table is not None, (
            "qsa_sparse: the head drafts AFTER the backbone's forward, which binds the "
            "per-request tensors (#801 r6 b9e)"
        )
        assert md.ring_slots.shape[0] >= bs, (
            f"qsa_sparse: one kv length per request, got {bs} for a step with "
            f"{md.ring_slots.shape[0]} request(s) (#801 r6 b9e)."
        )
        # The spelling `prepare_metadata` uses for the same two host buffers, at width 1.
        qo_indptr = torch.tensor([0] + [1] * bs, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(list(kv_lens), **_CPU_PINNED)
        draft = QSASparseMetadata(
            is_decode=True,
            last_indices=(qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True),
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
            num_tokens=bs,
        )
        # ⭐ `token_to_req_cpu` stays None: T == bs IS the one-token fast path's own discriminator
        #   (`_snapshot_decode`'s comment), and these two aranges are what it would have built.
        draft.token_to_req = torch.arange(bs, dtype=torch.int32, device=self.device)
        draft.cu_seqlens = qo_indptr.to(self.device, non_blocking=True)
        draft.seq_lens = kv_len.to(self.device, non_blocking=True)
        draft.ring_slots = md.ring_slots[:bs]
        draft.block_table = md.block_table[:bs]
        return draft

    # ----- dense layers -------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "qsa_sparse serves QSA layers only (Qwen3.8-Flash-Next has no dense attention "
            "layer); the QSA layer calls qsa_forward"
        )

    # ----- QSA layers ---------------------------------------------------------------------
    def qsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index,  # models.qwen4_exp.attention.QSAIndexerInputs
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        slot = self._idx_slot[layer_id]
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if md.block_table is None:
            self._snapshot_decode(md, batch)
        if slot == 0 or md.cmp_rows is None:
            # Rebuilt at the first QSA layer of every forward, not cached on the metadata: a
            # capture batch runs its warmup and its capture through ONE metadata object, and a
            # cached plan would bake the warmup's addresses into the graph.
            self._plan_index_writes(md, batch)

        self._update_index_cache(index, md, slot)
        indices = self._select(index, md, slot)
        return qsa_sparse_paged_attention(
            q,
            self.kvcache.k_cache(layer_id),
            self.kvcache.v_cache(layer_id),
            indices,
            md.block_table,
            md.token_to_req,
            torch.empty_like(q),
        )

    def _plan_index_writes(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Per-token slab row and ring row for this forward; the other QSA layers reuse it
        (it is layer-invariant). Pure device arithmetic: no host sync, graph-capturable."""
        md.positions = batch.positions
        out_loc = batch.out_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        rows = torch.arange(out_loc.numel(), device=self.device)
        req = md.token_to_req.to(torch.int64)
        slots = md.ring_slots.to(torch.int64).index_select(0, req)
        # out_loc % page_size == position % page_size and index_ratio divides page_size, so a
        # group closes exactly on out_loc % index_ratio == index_ratio - 1.
        closing = out_loc % self.ratio == self.ratio - 1
        scratch = self.kvcache.cmp_scratch_base + slots
        md.cmp_rows = torch.where(closing, out_loc // self.ratio, scratch).to(torch.int32)
        # Only the last ring_capacity rows of a request survive to the next forward; the rest
        # are masked off instead of dumped somewhere (vLLM rule).
        ends = md.cu_seqlens.to(torch.int64).index_select(0, req + 1)
        keep = rows >= ends - self.ring_capacity
        ring_row = slots * self.ring_capacity + positions % self.ring_capacity
        md.ring_rows = torch.where(keep, ring_row, torch.full_like(ring_row, -1)).to(
            torch.int32
        )

    def _update_index_cache(self, index, md: QSASparseMetadata, slot: int) -> None:
        """Compress each closing group into the slab, then refresh the pending ring."""
        from freetoken.kernel.triton.qsa import (
            qsa_compress_groups,
            qsa_index_norm_rope,
            qsa_store_rows,
        )

        rows = index.k.shape[0]
        ring = self.kvcache.pending_ring(slot)
        pooled = self._scratch("pooled", rows, self.index_head_dim, dtype=self.dtype)
        first = self._scratch("first_pos", rows, dtype=torch.int32)
        qsa_compress_groups(
            index.k,
            ring,
            md.ring_slots,
            md.token_to_req,
            md.cu_seqlens,
            md.positions,
            self.ratio,
            pooled,
            first,
        )
        qsa_index_norm_rope(
            pooled,
            first,
            self._index_rope_cache(),
            index.k_norm_weight,
            index.eps,
            self.kvcache.cmp_k_cache(slot),
            dest_rows=md.cmp_rows,
        )
        # After the compression read: the ring rows this forward overwrites are exactly the
        # ones a straddling group just consumed.
        qsa_store_rows(ring, md.ring_rows, index.k)

    def _select(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Score complete visible blocks, take the top-k, expand them to token indices."""
        from freetoken.kernel.triton.qsa import (
            expand_qsa_block_indices,
            qsa_index_norm_rope,
            qsa_mqa_paged,
        )

        rows = index.q.shape[0]
        positions = md.positions
        q_index = self._scratch(
            "q_index", rows, self.index_heads, self.index_head_dim, dtype=self.dtype
        )
        qsa_index_norm_rope(
            index.q.view(-1, self.index_head_dim),
            positions,
            self._index_rope_cache(),
            index.q_norm_weight,
            index.eps,
            q_index.view(-1, self.index_head_dim),
            heads=self.index_heads,
        )
        cmp_pages = self._cmp_pages(slot)
        columns = md.block_table.shape[1] * self.cmp_page_size
        indices = self._scratch("indices", rows, self.select_width, dtype=torch.int32)
        rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = slice(start, end)
            logits = self._scratch("logits", end - start, columns, dtype=torch.float32)
            visible = self._scratch("visible", end - start, dtype=torch.int32)
            qsa_mqa_paged(
                q_index[chunk],
                cmp_pages,
                md.block_table,
                md.token_to_req[chunk],
                positions[chunk],
                md.seq_lens,
                self.ratio,
                logits,
                visible,
            )
            blocks = self._scratch("blocks", end - start, self.block_topk, dtype=torch.int32)
            self._top_blocks(logits, visible, blocks)
            expand_qsa_block_indices(
                blocks,
                positions[chunk],
                md.seq_lens,
                md.token_to_req[chunk],
                self.ratio,
                self.token_topk,
                indices[chunk],
            )
        return indices

    def _top_blocks(
        self,
        logits: torch.Tensor,
        visible: torch.Tensor,
        blocks: torch.Tensor,
    ) -> None:
        """Top ``block_topk`` complete blocks per row, row-relative, -1 padded."""
        assert blocks.shape == (logits.shape[0], self.block_topk), (
            f"qsa block top-k output must be [rows, {self.block_topk}], got {tuple(blocks.shape)}"
        )
        if self._block_topk_kernel is not None:
            scratch_width = self._topk_scratch_width(logits.shape[1])
            scratch = (
                self._scratch("topk_scratch", logits.shape[0], scratch_width, dtype=torch.int32)
                if scratch_width
                else None
            )
            self._block_topk_kernel(logits, visible, blocks, scratch)
            return
        # The score kernel only writes columns below visible_blocks; mask the rest so a
        # stale row cannot win a slot. Real block scores are relu sums, never -inf.
        columns = logits.shape[1]
        column = torch.arange(columns, dtype=torch.int32, device=logits.device)
        logits.masked_fill_(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
        width = min(self.block_topk, columns)
        values, chosen = torch.topk(logits, width, dim=-1)
        blocks[:, :width] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
        if width < self.block_topk:
            blocks[:, width:] = -1

    def _topk_scratch_width(self, columns: int) -> int:
        """int32 columns per row the block top-k wants as scratch, 0 when it wants none."""
        if self._block_topk_kernel is None:
            return 0
        from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

        return qsa_block_topk_scratch_width(columns, self.block_topk)

    # ----- scratch ------------------------------------------------------------------------
    def _scratch(self, name: str, rows: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """A per-forward transient: the static decode buffer when it is wide enough (so a
        captured graph keeps one address), otherwise a fresh allocation."""
        buffer = self._graph.get(name)
        if buffer is not None and rows <= buffer.shape[0] and buffer.shape[1:] == shape:
            return buffer[:rows]
        return torch.empty((rows, *shape), dtype=dtype, device=self.device)

    # ----- CUDA graph (decode) --------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        table_width = get_global_ctx().page_table.shape[1]
        pages = -(-table_width // self.page_size)
        columns = pages * self.cmp_page_size
        # ⭐ #801 round 6 bullet 6: rows per request a CAPTURED step may forward. `engine/graph.py`
        #   sets it before this call (duck-typed, so no other backend needs to know what a verify
        #   step is); 1 -- the default -- builds exactly the buffers this file always built.
        verify = int(getattr(self, "verify_width", 1) or 1)
        # ⛔ THE SPLIT THAT MATTERS. `block_table` / `kvlen` / `table_idx` are per REQUEST and must
        #   NOT grow with the width: every QSA kernel reaches them THROUGH `token_to_req`, so a
        #   widened one would double-count each request. Everything else here is per TOKEN, and a
        #   step wider than its static buffer makes `_scratch` fall back to a fresh allocation --
        #   correct, and uncapturable, which is the whole failure this bullet exists to remove.
        rows = max_bs * verify
        chunk = max(1, min(rows, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))
        topk_scratch = self._topk_scratch_width(columns)

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        self._graph = {
            "block_table": torch.zeros((max_bs, pages), dtype=torch.int32, device=self.device),
            "kvlen": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "table_idx": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "token_to_req": torch.arange(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens": torch.arange(max_bs + 1, dtype=torch.int32, device=self.device),
            "logits": empty(chunk, columns, dtype=torch.float32),
            "visible": empty(rows, dtype=torch.int32),
            "blocks": empty(rows, self.block_topk, dtype=torch.int32),
            "indices": empty(rows, self.select_width, dtype=torch.int32),
            "pooled": empty(rows, self.index_head_dim, dtype=self.dtype),
            "first_pos": empty(rows, dtype=torch.int32),
            "q_index": empty(rows, self.index_heads, self.index_head_dim, dtype=self.dtype),
        }
        if verify > 1:
            # Constant per width, filled once: a uniform T-token step's token->request map is
            # `arange(rows) // T` and its query indptr is `arange(bs + 1) * T`. Neither depends
            # on the step, so `_stage_decode` copies neither.
            self._graph["token_to_req_verify"] = (
                torch.arange(rows, dtype=torch.int32, device=self.device) // verify
            )
            self._graph["cu_seqlens_verify"] = (
                torch.arange(max_bs + 1, dtype=torch.int32, device=self.device) * verify
            )
        if topk_scratch:
            self._graph["topk_scratch"] = empty(chunk, topk_scratch, dtype=torch.int32)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(md, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        # ⛔ #801 r6 b9d: this tensor is per ROW (`scheduler._make_input_tuple`), and every buffer
        #   `_stage_decode` writes is per REQUEST. `_request_rows` takes the one per request --
        #   AFTER `_stage_width`, so a ragged step still gets its own "must run eager" verdict
        #   rather than a reshape error naming neither.
        self._stage_decode(md, batch.padded_size, batch.active_table_idx.to(torch.int64))

    def reset_capture(self) -> None:
        super().reset_capture()
        self._graph = {}


__all__ = ["QSASparseAttnBackend", "QSASparseMetadata"]
