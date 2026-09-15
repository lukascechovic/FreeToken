"""#801 round 4 bullet 7 — the LOOSE half of the round's gate: real kernels vs torch, on the box.

⛔⛆ **What this is, and what it is not.** Bullets 3-5 settled the head's WIRING exactly -- observed
max |Δ| = 0.0 against a plain-torch oracle with no engine imports, off-GPU, with the MoE injected
identically on both sides. What that gate could not touch is the thing a load is for: the real
triton / flashinfer / NVFP4 kernels at the served geometry and the served shard. This module is
that half. It re-proves nothing about the wiring and must never be read as doing so.

⭐ **Why it needs no oracle of its own.** Every kernel the head runs already HAS a torch twin in
the engine, written as the thing the kernel is diffed against:

    layers/norm.py          `gemma_plus_one_rms_norm`              vs flashinfer/triton gemma_rmsnorm
    layers/rotary.py        `apply_rope_with_cos_sin_cache_torch`  vs `_rope_tiled`
    qwen4_exp/hc.py         `GatedResidual._mix_torch` / `._combine_torch`  vs the vLLM triton kernels
    qwen4_exp/attention.py  `TorchDenseQSAReference`               vs `QSASparseAttnBackend`

so every comparison here is the ENGINE'S OWN TWO IMPLEMENTATIONS of one function, on the SAME live
GPU tensors. ⛔ A re-spelled oracle would be a third implementation agreeing with itself -- and at
TP=2 it could not be written at all: `o_proj` is row-parallel, so a rank-local torch head holds a
PARTIAL sum and no arithmetic makes it whole. ⇒ the torch side BORROWS exactly two engine ops,
`self_attn.qkv_proj.forward` and `self_attn.o_proj.forward`: a column-sharded GEMM and a GEMM plus
a collective. Neither is arithmetic this round ported.

⭐ **Every stage is measured on IDENTICAL INPUTS.** The two chains are not run end to end against
each other and then differenced -- each stage's kernel and its twin are handed the SAME tensor, so
a per-stage number is that kernel's own error and not an accumulation of the ones before it. The
one end-to-end number (`head_end_to_end`) is stated separately and is the accumulation, against the
head's REAL forward from this same step.

⛔⛆ **The preconditions, and why none is optional.**
1. QSA is exactly dense only while a request sees at most ``index_budget + index_ratio - 1``
   tokens. Past that the backend SELECTS blocks and disagreeing with a dense attend is CORRECT
   behaviour, not a defect.
2. Every request must be a FRESH, UNCHUNKED prefill: a dense attend over this forward's own q/k/v
   is the same function as the paged one only when there is no history it cannot see.
3. `moe_prefill_overlap` must be OFF. The overlap path's double buffer asserts
   ``Prefill overlap buffer is being reused before release``, and this module calls the head's real
   MoE a second time. The deployed row already disables it (#912, PR #957) -- this is the gate that
   says so rather than the arm remembering to.
4. Decode is refused outright: it is CUDA-graph captured, and this module allocates.

:func:`_eligible` reports WHICH of these declined, so "nothing was checked" can never read as a
pass.

⚠ **What the numbers do NOT mean.** The head is fed a STAND-IN token (`draft_input_ids`) and its
output is discarded: nothing here is about draft quality or acceptance. α is the issue body's
bullets 4-5.

⚠ **Routing is discrete.** ``routing_agreement`` is reported beside the MoE stage because the
mixer's kernel/torch difference can flip a token's top-k experts, turning a 1e-3 upstream delta
into a large downstream one. An end-to-end tail at 100 % routing agreement and one at 97 % are
different findings wearing the same number.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from freetoken.layers.norm import gemma_plus_one_rms_norm
from freetoken.layers.rotary import apply_rope_with_cos_sin_cache_torch


def _rank() -> int:
    """This process's TP rank, from the engine's own registry.

    ⛔ Not an environment variable: a TP=2 arm runs both ranks under ONE `docker run` with one
    environment, so an env-derived rank would label both files `rank0` and one would silently
    overwrite or interleave with the other.
    """
    from freetoken.distributed import get_tp_info

    return int(get_tp_info().rank)


def _d(kernel: torch.Tensor, ref: torch.Tensor) -> dict:
    """``max |Δ|`` and relative rms of a kernel against its torch twin, as plain floats."""
    a, b = kernel.detach().float(), ref.detach().float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "rel_rms": ((a - b).pow(2).mean().sqrt() / (b.abs().mean() + 1e-12)).item(),
        "ref_absmean": b.abs().mean().item(),
        "shape": list(kernel.shape),
    }


def _eligible(causal_lm, batch, args, ctx) -> str | None:
    """``None`` when this forward can be checked; otherwise the reason it cannot."""
    if causal_lm.mtp is None or causal_lm.model.draft_hidden is None:
        return "no head, or FREETOKEN_MTP801_RUN_HEAD=0 (nothing real to compare against)"
    if not batch.is_prefill:
        return "decode batch (captured, and this module allocates)"
    cache = getattr(ctx, "moe_offload_cache", None)
    if cache is not None and getattr(cache, "prefill_overlap", False):
        return "moe_prefill_overlap is ON (its double buffer forbids a second MoE call)"
    ratio = getattr(ctx.kv_cache, "index_ratio", 1)
    window = args.index_budget + ratio - 1
    for r in batch.reqs:
        if r.cached_len:
            return f"request carries cached_len={r.cached_len} (history a dense attend cannot see)"
        if r.extend_len > window:
            return f"request is {r.extend_len} tokens, past the dense-equivalence window {window}"
    return None


# ======================================================================================
# the torch twins, stage by stage, each on the kernel's own input
# ======================================================================================


def _dense_attend(q, k, v, batch, sm_scale: float) -> torch.Tensor:
    """Rank-local causal dense attention over THIS forward's own q/k/v.

    ⛔ Rank-local on purpose: `TorchDenseQSAReference` takes its head counts from the GLOBAL
    `ModelConfig`, which is the unsharded count -- right for a CPU test, wrong for a TP=2 rank. The
    math is that class's `_attend`; the storage is this forward instead of a paged pool, which
    precondition 2 is what makes equivalent.
    """
    num_q, num_kv = q.shape[1], k.shape[1]
    rep = num_q // num_kv
    out = torch.empty_like(q)
    offset = 0
    for r in batch.reqs:
        n = r.extend_len
        rows = slice(offset, offset + n)
        keys = k[rows].repeat_interleave(rep, dim=1).float()
        values = v[rows].repeat_interleave(rep, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), keys) * sm_scale
        pos = torch.arange(n, device=q.device)
        scores = scores.masked_fill(~(pos.unsqueeze(0) <= pos.unsqueeze(-1)), float("-inf"))
        out[rows] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(q.dtype)
        offset += n
    return out


def _fuse_input_torch(head, input_ids, R) -> torch.Tensor:
    """`Qwen4ExpMTPHead.fuse_input` with both `GemmaPlusOneRMSNorm`s on their torch branch."""
    e = head._embed_tokens.forward(input_ids)
    norm = head.pre_fc_norm_embedding
    e = head.fc_embedding.forward(gemma_plus_one_rms_norm(e, norm.weight, norm.eps))
    norm = head.pre_fc_norm_hidden
    h = gemma_plus_one_rms_norm(R, norm.weight, norm.eps)
    h = head.fc_hidden.forward(h.unflatten(-1, (head.hc_count, head.hidden_size)))
    return (e.unsqueeze(-2) + h).flatten(-2)


def _attention_torch(attn, x, batch, report: dict) -> torch.Tensor:
    """`Qwen4ExpAttention.forward` with the q/k norms, the rope and the attend on their twins.

    Records each sub-kernel's own delta; returns the torch chain's projected output.
    """
    qg, k, v = attn.qkv_proj.forward(x).split(attn._qkv_split, dim=-1)
    qg = qg.view(-1, attn.num_q, attn.head_dim * 2)
    q = qg[..., : attn.head_dim].contiguous()
    gate = qg[..., attn.head_dim :].reshape(-1, attn.qo_attn_dim)
    k = k.contiguous().view(-1, attn.num_kv, attn.head_dim)
    v = v.contiguous().view(-1, attn.num_kv, attn.head_dim)

    q_kernel, k_kernel = q.clone(), k.clone()
    attn.q_norm.forward_inplace(q_kernel)
    attn.k_norm.forward_inplace(k_kernel)
    q_torch = gemma_plus_one_rms_norm(q, attn.q_norm.weight, attn.q_norm.eps)
    k_torch = gemma_plus_one_rms_norm(k, attn.k_norm.weight, attn.k_norm.eps)
    report["q_norm"] = _d(q_kernel, q_torch)
    report["k_norm"] = _d(k_kernel, k_torch)

    # in place on both sides, on clones, from the SAME post-norm torch input
    qr_kernel = q_torch.reshape(-1, attn.qo_attn_dim).clone()
    kr_kernel = k_torch.reshape(-1, attn.kv_attn_dim).clone()
    attn.rotary.forward(batch.positions, qr_kernel, kr_kernel)
    qr_torch = q_torch.reshape(-1, attn.qo_attn_dim).clone()
    kr_torch = k_torch.reshape(-1, attn.kv_attn_dim).clone()
    apply_rope_with_cos_sin_cache_torch(
        positions=batch.positions,
        query=qr_torch,
        key=kr_torch,
        head_size=attn.rotary.head_size,
        cos_sin_cache=attn.rotary._cos_sin_cache,
        is_neox=attn.rotary.is_neox,
    )
    report["rope_q"] = _d(qr_kernel, qr_torch)
    report["rope_k"] = _d(kr_kernel, kr_torch)

    o = _dense_attend(
        qr_torch.view(-1, attn.num_q, attn.head_dim),
        kr_torch.view(-1, attn.num_kv, attn.head_dim),
        v,
        batch,
        attn.head_dim**-0.5,
    )
    out = attn.o_proj.forward(o.reshape(-1, attn.qo_attn_dim) * torch.sigmoid(gate))

    # ⛔⛆ THE ONE STAGE THAT HAD NO GATE, AND THE REASON IT NEEDS ONE. Every other comparison here
    #   is a kernel against a twin over the same few hundred values; the ATTEND is the only stage
    #   whose difference can GROW WITH SEQUENCE LENGTH, and the first banked run showed exactly
    #   that (23 tokens -> 2.8e-2, 53 -> 6.1e-2) at 100 % routing agreement and an IDENTICAL fused
    #   input -- which is what refuted "the end-to-end tail is routing flips".
    # ⚠ Calling the real attention a second time is safe and was checked before it was written:
    #   `store_kv` writes the same K/V to the same `out_loc`, and the index plan writes the same
    #   compressed keys to `out_loc // ratio` and the same ring rows at `position % capacity`.
    #   Both are idempotent on identical input. ⛔ It is NOT safe on a batch that has moved on,
    #   which is why `_eligible` refuses anything but a fresh unchunked prefill.
    report["attention_out"] = _d(attn.forward(x, batch), out)
    return out


def _routing_agreement(mlp, x_kernel, x_torch, top_k: int):
    """How often the mixer's kernel/torch difference alone flips a token's top-k experts.

    ⭐ Returns the per-token agreement MASK as well as the counts, because the end-to-end number
    is only interpretable split by it: a flipped token ran through DIFFERENT EXPERTS and its
    downstream difference is not a rounding error at all.

    ⛔⛆ And the margin is reported with it. `top_k` over a 512-way router is a discrete function of
    a continuous score, so a token whose k-th and (k+1)-th logits are a rounding apart flips for
    free -- that is a TIE, the same thing round 3 spent two bullets separating from a real
    disagreement, not evidence the mixer is wrong. The median margin on flipped tokens against the
    median on agreed ones is what tells the two apart.
    """
    la = mlp.gate.forward(x_kernel).float()
    lb = mlp.gate.forward(x_torch).float()
    a = la.topk(top_k, dim=-1).indices.sort(-1).values
    b = lb.topk(top_k, dim=-1).indices.sort(-1).values
    same = (a == b).all(-1)
    # gap between the last SELECTED and the first REJECTED logit, on the torch side
    edge = lb.topk(top_k + 1, dim=-1).values
    margin = (edge[:, top_k - 1] - edge[:, top_k]).abs()
    flipped = ~same

    def med(t):
        return float(t.median().item()) if t.numel() else None

    return {
        "tokens": int(same.numel()),
        "agreed": int(same.sum().item()),
        "top_k": top_k,
        "median_margin_agreed": med(margin[same]),
        "median_margin_flipped": med(margin[flipped]),
        "logit_absmean": float(lb.abs().mean().item()),
    }, same


def _head_forward_torch(causal_lm, head, input_ids, R, batch, report: dict):
    """One head forward, every ported kernel replaced by its twin, each stage measured.

    Returns ``(sample_hidden, routing_agreement_mask)``."""
    layer = head.layers.op_list[0]
    attn_hc, mlp_hc = layer.attn_hyper_connection, layer.mlp_hyper_connection

    hidden = _fuse_input_torch(head, input_ids, R)
    report["fuse_input"] = _d(head.fuse_input(input_ids, R), hidden)

    x_k, s_k = attn_hc._mix_kernel(hidden)
    x_t, s_t = attn_hc._mix_torch(hidden)
    report["attn_hc_mix"] = _d(x_k, x_t)
    report["attn_hc_inject_logits"] = _d(s_k, s_t)

    y = _attention_torch(layer.self_attn, x_t, batch, report)

    c_k = attn_hc.combine(hidden, y, s_t)
    c_t = attn_hc._combine_torch(hidden, y, s_t)
    report["attn_hc_combine"] = _d(c_k, c_t)
    hidden = c_t

    x2_k, s2_k = mlp_hc._mix_kernel(hidden)
    x2_t, s2_t = mlp_hc._mix_torch(hidden)
    report["mlp_hc_mix"] = _d(x2_k, x2_t)
    report["mlp_hc_inject_logits"] = _d(s2_k, s2_t)
    report["routing_agreement"], agreed = _routing_agreement(
        layer.mlp, x2_k, x2_t, causal_lm._config.num_experts_per_tok
    )

    # ⭐ the REAL MoE -- 512 NVFP4 experts through the offload cache -- on the torch side's own
    #   input: the injected-callable shape bullets 3-5 used, so the MoE is not what is being diffed.
    y2 = layer.mlp.forward(x2_t)

    d_k = mlp_hc.combine(hidden, y2, s2_t)
    d_t = mlp_hc._combine_torch(hidden, y2, s2_t)
    report["mlp_hc_combine"] = _d(d_k, d_t)
    hidden = d_t

    m_k = head.hyper_connection_mixer._mix_kernel(hidden)[0]
    m_t = head.hyper_connection_mixer._mix_torch(hidden)[0]
    report["final_mixer"] = _d(m_k, m_t)
    return m_t, agreed


# ======================================================================================
# the entry point `Qwen4ExpForCausalLM.forward` calls
# ======================================================================================


def run_headcheck(causal_lm, batch) -> int:
    """Check one forward. Returns 1 if it ran, 0 if a precondition declined it.

    ⛔ Called AFTER the head's real forward on the same step, so both sides see one set of inputs.
    """
    from freetoken.core import get_global_ctx

    ctx = get_global_ctx()
    args = causal_lm._config.qwen4_args
    why = _eligible(causal_lm, batch, args, ctx)
    if why is not None:
        _emit({"ran": False, "declined": why})
        return 0

    report: dict = {
        "ran": True,
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tp_rank": _rank(),
        "batch": {
            "reqs": len(batch.reqs),
            "tokens": int(batch.input_ids.shape[0]),
            "extend_lens": [int(r.extend_len) for r in batch.reqs],
        },
    }
    # ⛔ The head ran on the FULL pre-mixer multi-stream, not the tap's rows -- see
    #   `Qwen4ExpModel.forward`'s `draft_head` note. Reconstructing it here would be a second
    #   spelling, so the model hands it over: the tap's buffer holds only the consumer's rows.
    hidden = causal_lm.model.draft_stream
    report["R"] = {"shape": list(hidden.shape), "dtype": str(hidden.dtype)}
    report["tap_rows"] = list(causal_lm.multi_stream.shape)
    with torch.inference_mode():
        got, agreed = _head_forward_torch(
            causal_lm, causal_lm.mtp, batch.input_ids, hidden, batch, report
        )
        want = causal_lm.model.draft_hidden
        report["head_end_to_end"] = _d(want, got)
        # ⛔⛆ THE TWO ARE NOT THE SAME CLAIM. `head_end_to_end` mixes a continuous kernel error
        #   with a DISCRETE one: a token whose top-k flipped ran through different experts, and no
        #   tolerance on it means anything. The gated number is this one -- the same difference
        #   over the tokens that routed identically -- and the flip count above is reported beside
        #   it rather than folded into it.
        report["head_end_to_end_agreed"] = (
            _d(want[agreed], got[agreed]) if bool(agreed.any()) else None
        )
    _emit(report)
    return 1


def _emit(report: dict) -> None:
    """One JSON line per checked forward, to the mounted check dir (stderr when unset)."""
    line = json.dumps(report, default=str)
    where = os.getenv("FREETOKEN_MTP801_CHECK_DIR", "").strip()
    if where:
        with open(os.path.join(where, f"headcheck-rank{_rank()}.jsonl"), "a") as fh:
            fh.write(line + "\n")
    print(f"[#801 headcheck] {line}", file=sys.stderr, flush=True)


__all__ = ["run_headcheck"]
