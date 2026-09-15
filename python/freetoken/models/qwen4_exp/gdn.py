# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `models/qwen4_exp/gdn.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 b8f25f400f1e4949544a3e17bf3ebc04,
# 273 lines) BIND-MOUNTED over the installed package, plus this block, `_verify_width`,
# `_varlen_conv_capturable`, `_conv_verify`, `_verify_fla` and a two-line branch inside
# `forward`'s decode arm. ⛔ In no
# image, in no Dockerfile ladder: `arm_mtp_801.sh`'s OVERLAY and ORIGS lists or nothing (#866).
#
# ⛔⛆ WHY IT EXISTS. A GDN layer's recurrent state and conv window are RECURRENCES, and they are
#   the half of a verify step that cannot be re-derived. A rejected KV page is a slot the next
#   step overwrites; a rejected token folded into the SSM state is there forever, and the row
#   serves subtly wrong text from then on with nothing erroring and no page to blame. So the
#   verify path runs with `disable_state_update=True` and caches every step's state, and
#   `spec.commit_linear_state` is the ONLY thing that advances the pool.
#
# ⭐ THE KERNEL ALREADY SUPPORTED THIS. `kernel/fla/fused_sigmoid_gating_recurrent.py` has taken
#   `intermediate_states_buffer`, `intermediate_state_indices`, `disable_state_update` and
#   `retrieve_parent_token` all along -- and names a `target_verify` mode in its own docstring.
#   Nothing in this engine passed them. Its `for _ in range(0, T)` loop reads T per request from
#   `cu_seqlens`, so the recurrence needs no kernel change at all: only the ragged indptr
#   (`attention/linear.py`, same round) and the buffer below.
#
# ⚠ THE CONV NEEDS NO KERNEL SUPPORT EITHER. A verify step is a 2-token CONTINUATION, which is
#   what the varlen (prefill) conv already does with `has_initial_state` set. What it does not do
#   is leave a rollback point -- so the window is cloned first.
#
# ⛔ VRAM. The buffer is `[bs, T, HV, Dk, Dv]` fp32 per layer, held until the commit. At TP=2 the
#   deployed row's local geometry is 24 v-heads x 128 x 128 -> 1.5 MiB per (request, step), so
#   bs=1 T=2 across 36 GDN layers is ~108 MiB per rank. It is indexed by BATCH ROW, not pool
#   slot: a slot-indexed buffer would be T copies of the entire GDN pool.
#
# ⛔⛆ PLE IS NOT FIXED HERE AND IS NOT OPTIONAL. `models/qwen4_exp/ple.py` builds its OWN decode
#   metadata (`cu_seqlens=arange(bs+1)`, `seq_lens=(1,)*bs`) and carries its own per-request conv
#   state with no rollback. It is the same class of bug as this file's and it is on the deployed
#   row's decode path ⇒ a verify load is wrong until it lands. Gated by name in
#   `test_gdn_verify_801.py::TestWhatIsStillUnfixed`.

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged

from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged
from freetoken.distributed import get_tp_info
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from freetoken.models.quant_linear import make_row_parallel_quant
from freetoken.utils import div_even


_GATE_ACTIVATIONS = ("silu", "swish", "sigmoid")


def _verify_width(batch) -> int:
    """#801: the widest per-request query length this forward carries — the intermediate
    buffer's step stride. Host arithmetic over the batch's own requests, computed once and
    reused by the other 35 GDN layers; the kernel still derives each request's own ``T`` from
    ``cu_seqlens``, so a ragged batch (one row speculating, one not) needs no padding."""
    width = getattr(batch, "linear_verify_width", None)
    if width is None:
        width = max(r.extend_len for r in batch.padded_reqs)
        batch.linear_verify_width = width
    return width


def _varlen_conv_capturable(
    x, weight, conv_states, cu_seqlens, cache_indices, has_initial_state, *, max_seq_len
):
    """`causal_conv1d_varlen`, launched so that it can run INSIDE a CUDA-graph capture.

    ⛔⛆ **THE BUG THIS EXISTS FOR, AND IT COST THE ROUND ITS FIRST LOAD.** The image's dispatcher
    (`kernel/causal_conv1d.py`) forwards six positional arguments and no more. On gfx1201
    `is_sgl_kernel_installed()` is FALSE, so it falls through to the triton implementation, whose
    wrapper ends its prologue with

        if max_seq_len is None:
            max_seq_len = int(seq_lens.max().item())

    -- a device-to-host read, which is `hipErrorStreamCaptureUnsupported` the moment a stream is
    capturing. The verify graph therefore died during capture at engine bring-up, 75 s in, with
    `CUDA error: operation not permitted when stream is capturing`. ⭐ The triton wrapper's own
    comment says what to do about it: *"max_seq_len / batch are host-known metadata in production
    (scheduler); pass them in to keep this launch graph-capturable (no .item() device->host
    sync)."* It takes the argument. Nothing passed it.

    ⭐ `batch` is left at its default deliberately: the wrapper derives it as
    ``cu_seqlens.numel() - 1``, which is a SHAPE and reads no device memory. Passing a second
    value would be a second thing to get wrong for no gain.

    ⛔ The sgl branch takes the dispatcher unchanged -- `causal_conv1d_fwd` is a fused C++ op with
    no Python prologue and no host read, so it is capturable as it stands. Spelled out rather
    than assumed: this image does not have it, and the branch that is never taken here is exactly
    the branch a later image would take.

    ⚠ Not folded into `_conv_prefill`. That function is the REAL prefill path that every served
    row runs, a prefill is never captured, and its `max_seq_len` is genuinely unknown on the host.
    Leaving it byte-for-byte is what keeps this round's change off the plain path.
    """
    from freetoken.kernel.backend import is_sgl_kernel_installed

    if is_sgl_kernel_installed():
        return causal_conv1d_varlen(
            x, weight, conv_states, cu_seqlens, cache_indices, has_initial_state
        )
    from freetoken.kernel.triton.causal_conv1d_triton import (
        causal_conv1d_varlen as triton_causal_conv1d_varlen,
    )

    return triton_causal_conv1d_varlen(
        x, weight, conv_states, cu_seqlens, cache_indices, has_initial_state,
        max_seq_len=max_seq_len,
    )


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by an ``activation(z)`` gate (HF Qwen4ExpTextRMSNormGated).

    Uses the fused fla ``rms_norm_gated`` triton kernel (norm(x) * act(z) in one
    kernel) instead of the unfused pow/mean/rsqrt/mul/act chain, matching sglang's
    ``RMSNormGated`` -- collapses ~8 elementwise kernels per GDN layer into one.
    Qwen3.8-Flash-Next gates with sigmoid where Qwen3.5 gates with silu."""

    def __init__(self, dim: int, eps: float, activation: str):
        # rms_norm_gated drops the gate entirely (no error) for a name it does not know.
        assert activation in _GATE_ACTIVATIONS, f"unsupported GDN output gate {activation!r}"
        self.weight = torch.empty(dim)
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation=self.activation,
        )


class Qwen4ExpGatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.

    ``output_gate`` is the gate activation name from ``LinearGatedDeltaGroupConfig``
    ("sigmoid" for Qwen3.8-Flash-Next).
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, output_gate: str = "sigmoid",
        expert_quant: str = "none", attn_quant: str = "none",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6/3.8 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        # Head counts, and every width derived from them, are RANK-LOCAL. The division is the
        # one LinearStatePool._linear_local_dims already uses, so the conv/recurrent state slots
        # and this module's tensors agree; ⛔⛆ a module that kept the full widths would slice the
        # next rank's heads out of its own (correctly halved) GEMM result.
        tp_size = get_tp_info().size
        self.num_k_heads = div_even(num_k_heads, tp_size, allow_replicate=True)
        self.num_v_heads = div_even(num_v_heads, tp_size, allow_replicate=True)
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = self.num_k_heads * head_k_dim
        self.value_dim = self.num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        full_key_dim = num_k_heads * head_k_dim
        full_value_dim = num_v_heads * head_v_dim
        full_conv_dim = 2 * full_key_dim + full_value_dim
        # LinearColParallelMerged divides each DECLARED size by tp_size, so the local conv_dim
        # built from local head counts must be that quotient -- it is not when only one of the
        # two head counts replicates. Fail here rather than build a layer of the wrong width.
        assert self.conv_dim * tp_size == full_conv_dim, (
            f"local conv_dim {self.conv_dim} does not tile {full_conv_dim} at tp_size={tp_size}: "
            f"k heads {num_k_heads} -> {self.num_k_heads}, v heads {num_v_heads} -> "
            f"{self.num_v_heads} divide differently"
        )
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches sglang/vLLM).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        self._fp8 = self._block_fp8 or self._pertensor_fp8

        # b|a are ONE COLUMN PER V HEAD, so they shard with the v heads like everything else.
        self._in_proj_split = [
            self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads
        ]
        if self._fp8:
            # ⛔⛆ The fp8 input projections are not TP-aware (a per-tensor scale is fitted to the
            # full row; a block scale grid is indexed in 128-wide tiles), so refuse rather than
            # emit a layer of the right shape that computes the wrong number.
            if tp_size > 1:
                raise NotImplementedError(
                    f"GDN fp8 in_proj is not implemented at tp_size={tp_size}"
                )
            ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
            self.in_proj_qkvz = ColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a. Declared with
            # the FULL widths -- the layer shards them itself.
            self.in_proj = LinearColParallelMerged(
                hidden_size,
                [full_conv_dim, full_value_dim, num_v_heads, num_v_heads],
                has_bias=False,
            )
            assert sum(self._in_proj_split) == self.in_proj.local_output_size, (
                f"declared split {self._in_proj_split} does not tile the local GEMM width "
                f"{self.in_proj.local_output_size} at tp_size={tp_size}"
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        # One entry per v head -> rank-local, like b|a which they gate.
        self.dt_bias = torch.empty(self.num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(self.num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation=output_gate)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_* stay bf16 in every mode (above), so a compressed-tensors
        # NVFP4 checkpoint (attn_quant=="nvfp4") only makes out_proj native FP4.
        # Column-sharded v heads mean each rank's core output is a PARTIAL sum: row-parallel,
        # all-reduces at TP>1, and declared with the FULL value_dim (the layer splits it).
        self.out_proj = make_row_parallel_quant(
            expert_quant, attn_quant, full_value_dim, hidden_size, has_bias=False
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _conv_verify(self, conv_in: torch.Tensor, pool, fla, batch):
        """#801: multi-token causal conv for a verify step, plus the window it overwrote.

        Returns ``(silu(conv) [total, conv_dim], window_before [bs, conv_dim, kernel-1])``.
        ⚠ `causal_conv1d_varlen` writes into a TRANSPOSED COPY of `conv_in` (see
        `_conv_prefill`), so `conv_in` survives as this step's raw input — which is what the
        conv state stores and therefore what `spec.commit_linear_state` rebuilds the window from.
        ⛔ The single-token `causal_conv1d_decode` does NOT have that property (it writes through
        a view of its argument), which is a second reason the verify path takes the varlen call.

        ⛔⛆ **IT DOES NOT CALL `_conv_prefill`, AND THAT IS THE WHOLE OF THE BULLET-9 FIX.** A
        verify step is CAPTURED; a prefill never is. See `_varlen_conv_capturable` for the device
        read that killed the round's first load, and `_verify_width` for why `max_seq_len` is
        host-known here and not in a prefill: it is the same number the intermediate-states
        buffer is already strided by, computed once per forward and shared by all 36 layers.
        """
        li = pool.local_index(self.layer_id)
        slots = fla.cache_indices.to(torch.int64)
        window = pool.conv_states[li].index_select(0, slots).clone()
        has_initial_state = torch.ones(
            slots.numel(), dtype=torch.bool, device=conv_in.device
        )
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total], as `_conv_prefill` builds it
        out = _varlen_conv_capturable(
            x, self._conv_weight(), pool.conv_states[li],
            fla.cu_seqlens, fla.cache_indices, has_initial_state,
            max_seq_len=_verify_width(batch),
        )
        # ⛔⛆ #801 r6 b9ah — THE ROOT CAUSE OF THE ROUND, AND IT IS THIS `.contiguous()`.
        #   `fused_sigmoid_gating_delta_rule_update` is passed exactly ONE stride per tensor —
        #   `q.stride()[1]`, the TOKEN axis — and then indexes `p_q = q + bos*stride_q + i_h*K +
        #   o_k`: the HEAD axis hard-coded `i_h*K` and the FEATURE axis `o_k`, i.e. contiguous.
        #   There is no stride to pass for either and no contiguity assert, so a transposed view is
        #   read at the WRONG ADDRESSES and nothing raises.
        #   Without this call `out.transpose(0, 1)` is `[total, conv_dim]` with stride
        #   `[1, total]`, and the `torch.split` + `reshape` below keep it a VIEW all the way into
        #   the launch: q arrives `[1, T, Hk, Dk]` stride `[T, 1, Dk*T, T]`.
        #   ⭐⭐⭐ AT T=1 THAT IS `[.., 1, Dk, 1]` — EXACTLY what the kernel assumes — which is why
        #   the served decode path (one token per request by construction) has always been correct
        #   and why this was invisible until speculation made T>1. At T=2 the head stride is 2*Dk
        #   and the feature stride 2, wrong on both axes.
        #   Contiguous here rather than on q/k/v after the split: same bytes copied (q, k and v ARE
        #   the three pieces of `mixed`) in ONE call instead of three, and it gives `mixed` exactly
        #   the layout `_conv_decode` already returns, so both paths meet the kernel's contract the
        #   same way. `repro_gdn_t2_801.kernel_layout_ok` states that contract.
        return out.transpose(0, 1).contiguous(), window

    # ----- the verify step's intermediate states ------------------------------------------
    def reserve_verify_states(self, max_bs: int, width: int, rec: torch.Tensor) -> None:
        """#801 round 6 bullet 6: allocate this layer's intermediate-states buffer ONCE, at final
        size, BEFORE any capture.

        ⛔⛆ An allocation made DURING ``torch.cuda.graph(...)`` comes out of THAT graph's private
        pool -- right on the eager warm-up, wrong on every replay. Bullet 5 allocated this with
        `torch.empty` per forward per layer, which is a fresh address every step and therefore
        uncapturable. `engine/graph.py::GraphRunner._capture_graphs` calls the model's
        `mtp_reserve_verify_buffers` hook, which calls this, before the first verify capture --
        the same "reserve, do not rely on capture order" rule `engine.py`'s own
        `mtp_reserve_graph_buffers` comment states.

        ⛔ VRAM, stated rather than discovered: ``[max_bs, width, HV, Dk, Dv]`` fp32 is ~1.5 MiB
        per (request, step) per rank on the deployed row, so bs=4 T=2 across 36 GDN layers is
        ~432 MiB per rank. That is why the CAP is the model's (`mtp_verify_graph_max_bs`) rather
        than the T=1 graph set's own `max_graph_bs`.
        """
        self._verify_states = torch.empty(
            (max_bs, width, *rec.shape[1:]), dtype=rec.dtype, device=rec.device
        )

    def _verify_buffer(self, bs: int, width: int, rec: torch.Tensor) -> torch.Tensor:
        """The reserved buffer when it fits this step, otherwise a fresh one.

        ⭐ The `attention/qsa_sparse.py::_scratch` idiom, and the fallback means the same thing
        there and here: **correct, and uncapturable**. A ragged verify step (narrower than the
        reserved width) takes it, and `GraphRunner` already refuses to capture one.

        ⛔ ``[:bs]`` of a ``[max_bs, W, ...]`` buffer is contiguous; ``[:bs, :w]`` for ``w < W``
        is not, and the fused kernel writes THROUGH this buffer -- which is the second reason a
        narrower step allocates instead of slicing.
        """
        buffer = getattr(self, "_verify_states", None)
        if (
            buffer is not None
            and bs <= buffer.shape[0]
            and width == buffer.shape[1]
            and buffer.dtype == rec.dtype
            and buffer.shape[2:] == rec.shape[1:]
        ):
            return buffer[:bs]
        assert not (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ), (
            f"#801: the GDN verify buffer for bs={bs} width={width} is being allocated while a "
            "CUDA graph is capturing -- it would come out of that graph's private pool and every "
            "replay would write somewhere else. Reserve it before the capture."
        )
        return torch.empty((bs, width, *rec.shape[1:]), dtype=rec.dtype, device=rec.device)

    def _verify_fla(self, q, k, v, a, b, *, pool, li, fla, batch, conv_in, conv_window):
        """#801: the fused fla recurrence over T tokens per request, with a rollback point.

        `disable_state_update=True` leaves the pool holding the step-ENTRY state, so an abandoned
        step changes nothing and `spec.commit_linear_state` is the only advance. The per-step
        states land in a per-forward buffer; the snapshot that names them rides on the batch, so
        this bullet needs no `core.py` overlay (bullet 2 made the same call for `Req`).
        """
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        from .spec import LinearSnapshot, gdncheck_wants_layer

        rec = pool.recurrent_states[li]
        bs = fla.cache_indices.numel()
        buffer = self._verify_buffer(bs, _verify_width(batch), rec)
        # ⛔⛆ #801 r6 b9af: THE ARGUMENTS ARE A DICT SO THE DUMP CANNOT TRANSCRIBE THEM WRONG.
        #   `spec.gdn_dump_cell` records the scalars of this launch — scale, the two softplus
        #   constants, `use_qk_l2norm_in_kernel`, `disable_state_update`, whether a buffer was
        #   passed — into a manifest a replay then reproduces. Assembling that list a second time
        #   inside `_gdncheck` would be a transcription that can drift from the call it claims to
        #   describe, silently, and the replay would measure a shape nothing ran. That is #866
        #   exactly: print the body the probe sends. ⭐ The expansion is identical to the explicit
        #   call it replaces, and the dict is built only on steps whose python runs at all — a
        #   graph REPLAY never reaches this line.
        fla_kwargs = dict(
            A_log=self.A_log, a=a, dt_bias=self.dt_bias,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b,
            initial_state_source=rec,
            initial_state_indices=fla.cache_indices,
            scale=self.head_k_dim ** -0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=fla.cu_seqlens,
            disable_state_update=True,
            intermediate_states_buffer=buffer,
            intermediate_state_indices=torch.arange(bs, dtype=torch.int32, device=rec.device),
        )
        o = fused_sigmoid_gating_delta_rule_update(**fla_kwargs)
        snapshots = getattr(batch, "linear_snapshots", None)
        if snapshots is None:
            snapshots = {}
            batch.linear_snapshots = snapshots
        snapshots[self.layer_id] = LinearSnapshot(
            intermediate_states=buffer,
            conv_window=conv_window,
            conv_inputs=conv_in,
            cu_seqlens=fla.cu_seqlens,
            cache_indices=fla.cache_indices,
        )
        # ⛔⛆ #801 r6 b9x. ASKED BEFORE the call, not inside it: the capture pass runs this same
        #   branch inside `torch.cuda.graph(...)`, and everything `_gdncheck` builds is device work
        #   that would be BAKED INTO THE GRAPH and paid on every replay. `spec.stage_verify` is the
        #   only writer of the owner and it never sees the capture's batch.
        if gdncheck_wants_layer(batch, li):
            self._gdncheck(batch, q=q, k=k, v=v, a=a, b=b, rec=rec, fla=fla, fused_out=o,
                           local_layer=li, fla_kwargs=fla_kwargs)
        return o[0]

    def _gdncheck(self, batch, *, q, k, v, a, b, rec, fla, fused_out, local_layer=0,
                  fla_kwargs=None) -> None:
        """#801 r6 b9x: what THIS kernel call RETURNED, against bullet 9w's pure-torch oracle.

        ⛔⛆ **9t's FINDING 1 IS THE WHOLE REASON.** `recurrent_gated_delta_rule` returns
        ``(output, state)`` and every call site this round wrote is ``_, state = …``; the round
        has gated what a verify forward STORES and never what it RETURNS, which is the one thing
        loads 6/10 and 9t's check row all say is wrong at row 0.

        ⭐⭐ **SOUND WHERE `CHECKROW` IS NOT, and that is why it can live here.**
        ``disable_state_update=True`` means ``rec`` still holds the step-ENTRY state on this line,
        so the reference runs on the same inputs the kernel just consumed with **no rewind at
        all** -- no second backbone forward, no KV or QSA-index write, and none of 9p's
        still-unaffirmed judgement call.

        ⭐ **A LAYER SET, defaulting to the first local layer alone** (#801 r6 b9aa; it was
        `li == 0` outright through loads 11 and 12). 9w's reasoning still holds and is why the
        DEFAULT is unchanged: a layer-0 disagreement is the kernel or its arguments, a clean
        layer 0 under a garbage logit row is something that grows on the way up. **Load 12
        answered that question** -- row 0 is wrong by the same margin as row 1 at layer 0,
        ratio 1.008 -- so the live question is *where on the way up*, and only a set can ask it.

        ⛔ **THE GQA EXPAND BELONGS HERE**, because this is the only place that knows both counts.
        The fla kernel handles GQA in-kernel and takes ``q``/``k`` at ``num_k_heads``;
        `gdn_reference` does not -- its own forward calls ``repeat_interleave`` before the rule.

        ⛔⛆ **``o[0]``, not ``o[0][0]`` -- AND 9x HAD IT THE OTHER WAY ROUND (#801 r6 b9z).**
        The kernel ALLOCATES ``o = q.new_empty(NK, *v.shape)`` at ``[NK, B, T, Hv, Dv]``, and 9x
        wrote this docstring from that line. Its LAST TWO LINES are ``o = o.squeeze(0); return o``
        -- so what a caller receives is ``v.shape`` == ``[B, T, Hv, Dv]``, ``B == 1``, NK already
        gone. ``o[0][0]`` is therefore ``[Hv, Dv]``, which right-aligns against the oracle's
        ``[T, Hv, Dv]`` and raised inside warmup on load 11.
        ⭐ ``_verify_fla``'s own ``return o[0]`` is the corroboration: it is the SERVED path and
        it is correct, so ``o`` cannot have had a spare leading dim.
        ⛔ The risk 9x named is real and is now caught LOUDLY rather than by arithmetic:
        `spec.gdncheck_layer` states the rank contract against ``q``'s geometry and raises.
        """
        from .spec import GDNCHECK_OWNER, gdn_dump_armed, gdn_dump_cell, gdncheck_batch

        g, beta = self._gate_params(a, b)
        query, key = q[0], k[0]
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=1)
            key = key.repeat_interleave(rep, dim=1)
        total = query.shape[0]
        # ⛔ #801 r6 b9af: HOISTED, so the DUMP and the ORACLE receive the SAME tensors and not two
        #   separately-built ones. A second `repeat_interleave` or a second `index_select` here
        #   would make the replay a measurement of this method's determinism rather than of the
        #   cell — and the two would look identical in every bank.
        g = g.reshape(total, self.num_v_heads)
        beta = beta.reshape(total, self.num_v_heads)
        entry_state = rec.index_select(0, fla.cache_indices.to(torch.int64))
        t1_out = self._gdncheck_t1(q=q, k=k, v=v, a=a, b=b, rec=rec, fla=fla)
        report = gdncheck_batch(
            batch,
            t1_out=t1_out,
            layer_id=self.layer_id,
            # ⛔ #801 r6 b9aa: BOTH. `layer_id` is the global id a log line names; `local_layer`
            #   is `pool.local_index(layer_id)`, which the state pool is keyed by and which the
            #   layer SET selects on. They coincide at 0 and nowhere else is guaranteed.
            local_layer=local_layer,
            q=query,
            k=key,
            v=v[0],
            # ⛔ reshaped rather than passed through: a `g` that arrived [1, T, Hv] would BROADCAST
            #   against the oracle's [T, Hv] instead of raising, and the report would be of a
            #   recurrence nothing ran.
            g=g,
            beta=beta,
            entry_state=entry_state,
            cu_seqlens=fla.cu_seqlens,
            fused_out=fused_out[0],
        )

        # ⭐⭐⭐ #801 r6 b9af: ONE CELL'S TENSORS, ON DISK, SO THE BISECT NEEDS NO SECOND LOAD.
        #   9ae acquitted the kernel off the model -- every arm exact to the bf16 quantum -- so the
        #   defect is in the live DATA, the live ENVIRONMENT or this INSTRUMENT, and no seventh
        #   GDNCHECK sweep separates those. `repro_gdn_t2_801.py --from` replays what is written
        #   here and splits all three; its `replay_verdict` fixes the reading in advance.
        #
        # ⛔⛆ **PAIRED WITH THE REPORT, NOT WITH THE DECISION TO MEASURE.** `report is None` means
        #   the budget declined this cell, so it has no line in the bank -- and a replay with
        #   nothing to be read beside is a second opinion, not a bisect. The dump therefore spends
        #   only where a report was actually emitted.
        #
        # ⛔ **BOTH HALVES, because 9ae put the instrument back on the suspect list.** The KERNEL's
        #   own arguments come straight out of `fla_kwargs` -- the dict the call above was made
        #   with -- and the ORACLE's come from the marshalling ten lines up. If the two disagree on
        #   the same cell, the difference IS this method, which reads correct at the desk and is
        #   therefore last, but is no longer excluded.
        if report is not None and gdn_dump_armed(getattr(batch, GDNCHECK_OWNER, None)):
            bounds = [int(x) for x in fla.cu_seqlens.tolist()]
            kw = fla_kwargs or {}
            gdn_dump_cell(
                getattr(batch, GDNCHECK_OWNER, None),
                tensors={
                    # the KERNEL's own arguments, as passed -- strides and all
                    "q": kw["q"], "k": kw["k"], "v": kw["v"], "a": kw["a"], "b": kw["b"],
                    "A_log": kw["A_log"], "dt_bias": kw["dt_bias"],
                    "entry_state": entry_state,
                    "cu_seqlens": fla.cu_seqlens, "cache_indices": fla.cache_indices,
                    # the ORACLE's, exactly as `gdncheck_batch` above received them
                    "oracle_q": query, "oracle_k": key, "oracle_v": v[0],
                    "oracle_g": g, "oracle_beta": beta,
                    # what the cell RETURNED, both calls
                    "fused_out": fused_out[0], "t1_out": t1_out,
                },
                # ⛔ The POOL is described and NEVER saved: `[slots, Hv, Dv, Dk]` fp32 is ~100 GB
                #   at the deployed slot count. The replay compacts it to `bs` rows, so a defect in
                #   the slot stride or in the gather does not reproduce -- which is why the shape
                #   and stride are recorded rather than the hypothesis being dropped.
                pool={"shape": list(rec.shape), "stride": list(rec.stride()),
                      "dtype": str(rec.dtype)},
                call={
                    "t": bounds[1] - bounds[0] if len(bounds) > 1 else 0,
                    "bs": int(fla.cache_indices.numel()),
                    "buffer": kw.get("intermediate_states_buffer") is not None,
                    "disable_state_update": bool(kw.get("disable_state_update")),
                    # ⛔ `None` on purpose: `num_stages` is a local inside the image's wrapper, so
                    #   this call does not know its own. The replay OBSERVES it at launch. Writing
                    #   a 3 because 9ad read one in the source would assume the dial under test.
                    "num_stages": None,
                    "scale": float(kw["scale"]),
                    "use_qk_l2norm_in_kernel": bool(kw["use_qk_l2norm_in_kernel"]),
                    "softplus_beta": float(kw["softplus_beta"]),
                    "softplus_threshold": float(kw["softplus_threshold"]),
                },
                layer_id=self.layer_id, local_layer=local_layer,
                step=int(report["step"]), report=report,
            )

    def _gdncheck_t1(self, *, q, k, v, a, b, rec, fla) -> torch.Tensor:
        """The SAME kernel, on the SAME entry state, restricted to each request's FIRST token.
        #801 r6 b9ab.

        ⛔⛆ **LOAD 13 IS WHY. The oracle has never once read CLEAN**, and an instrument that has
        never been shown able to agree cannot convict. Load 13 measured ``row0_rel`` ≈ 1.0-2.7 at
        **all 36 local GDN layers**, layer 0 included, with no rise and no step -- an error the
        size of the signal, everywhere. Two hypotheses fit that equally: the T=2 verify-mode
        return is wrong at every layer, or **the oracle is**, in which case loads 11/12/13
        measured the instrument. Nothing in the report separates them.

        ⭐⭐ **THIS CALL IS THE CONTROL, AND IT IS THE SAME KERNEL.**
        `gdn_kernels.gdn_decode_fla` -- the SERVED decode path, which produces correct text every
        step -- is `fused_sigmoid_gating_delta_rule_update` with the identical
        ``use_qk_l2norm_in_kernel``, ``softplus_beta``/``softplus_threshold``, ``scale``,
        ``initial_state_source``/``initial_state_indices`` and ``o[0]`` strip. It differs from
        the verify call in exactly two things: ``disable_state_update`` and **T**. So running the
        verify step's own tensors through it at **T=1** re-creates the known-good call:

        * the reference agrees at T=1 ⇒ **the oracle is sound** and the T=2 return is the defect;
        * it disagrees at T=1 too ⇒ **the marshalling is wrong** and three loads measured it.

        ⭐⭐⭐ **AND IT BUYS A COMPARISON THE REFERENCE IS NOT PART OF.** A gated delta rule is
        CAUSAL, so token 0's output cannot depend on token 1: the T=2 call's first row and this
        T=1 call's row must agree. That check has **no oracle in it at all** -- it is the kernel
        against ITSELF -- so a disagreement there is a statement about the hardware that survives
        whatever is true of `gdn_reference`.

        ⛔ ``disable_state_update=True`` and **no** ``intermediate_states_buffer``: the wrapper
        sets ``CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None`` and
        ``DISABLE_STATE_UPDATE`` independently, so this second call writes **nothing** -- not the
        pool, not a buffer. Read from the image's own wrapper, through the ``return``, not from
        its signature (9z).

        ⛔ ``q``/``k`` arrive at ``num_k_heads`` and stay there: the kernel expands GQA itself.
        The oracle's GQA expansion is for `gdn_reference`, which does not -- passing the expanded
        tensors here would be the same convention error the other way round.

        ⛔ ``o[0]``, never ``o[0][0]`` -- ``o`` is ``[B, T, HV, V]`` after the wrapper's own
        ``squeeze(0)``, and here ``T`` is the request count, so ``o[0]`` is ``[bs, HV, V]``. That
        is `gdn_decode_fla`'s own strip and its own comment says why (*"o[0,0] would drop B>1"*).
        """
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        # ⛔ Each request's FIRST token, not the first `bs` rows: on a verify step the rows are
        #   packed T-per-request, so rows 0..bs-1 would be request 0's whole window whenever T>1.
        first = fla.cu_seqlens[:-1].to(torch.int64)
        bs = first.numel()
        # ⛔ arange, matching the decode path's own query indptr: bs sequences of ONE token, which
        #   is what makes this the decode call rather than a shorter verify one.
        cu1 = torch.arange(bs + 1, dtype=fla.cu_seqlens.dtype, device=fla.cu_seqlens.device)
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log, a=a[first], dt_bias=self.dt_bias,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q[:, first], k=k[:, first], v=v[:, first], b=b[first],
            initial_state_source=rec,
            initial_state_indices=fla.cache_indices,
            scale=self.head_k_dim ** -0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu1,
            disable_state_update=True,
        )
        return o[0]

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._fp8:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            # #801: `total != bs` is a verify step — T query rows for some request.
            verify = total != fla.cache_indices.numel()
            if verify:
                mixed, conv_window = self._conv_verify(conv_in, pool, fla, batch)
            else:
                mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            if verify:  # #801
                core_out = self._verify_fla(
                    q, k, v, a, b, pool=pool, li=li, fla=fla, batch=batch,
                    conv_in=conv_in, conv_window=conv_window,
                )
            else:
                core_out = gdn_decode_fla(
                    q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                    state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                    cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                )
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen4ExpGatedDeltaNet"]
