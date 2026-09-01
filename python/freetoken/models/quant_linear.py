"""Quant-aware dense-linear factories, shared by the models that serve quantized
dense projections (qwen3_5_moe, muse_glimmer).

Maps the model's quant config (``expert_quant`` for the dense MLP / shared-expert path,
``attn_quant`` for attention + GatedDeltaNet projections) to the right ``BaseOP`` linear:
block-FP8, per-tensor-FP8 and NVFP4 implementations live under ``freetoken.kernel.triton``;
the bf16 fallback is the framework's TP-aware ``freetoken.layers``. Only the *dispatch*
(config -> layer class) lives here.
"""

from __future__ import annotations


def make_col_merged_quant(expert_quant: str, attn_quant: str, in_f: int,
                          output_sizes: list[int], has_bias: bool = False):
    """Column-merged linear for a dense projection: block-fp8 / per-tensor-fp8 / nvfp4 / bf16."""
    if expert_quant == "fp8_block":
        from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged

        return Fp8BlockColMerged(in_f, output_sizes, has_bias)
    if attn_quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

        return Fp8PerTensorColMerged(in_f, output_sizes, has_bias)
    if attn_quant == "nvfp4":  # compressed-tensors W4A16 attention (q/k/v fused)
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged

        return Nvfp4DenseColMerged(in_f, output_sizes, has_bias)
    from freetoken.layers import LinearColParallelMerged

    return LinearColParallelMerged(in_f, output_sizes, has_bias=has_bias)


def _quantized_mode(expert_quant: str, attn_quant: str) -> str | None:
    """The quant mode a dense projection dispatches on, or ``None`` for bf16.

    ⭐ The dispatch ORDER lives here and only here. Both factories below select on this result
    rather than re-testing ``expert_quant``/``attn_quant``, so the replicated path and the
    row-parallel path cannot drift into disagreeing about which mode a config is in.
    """
    if expert_quant == "fp8_block":
        return "fp8_block"
    if attn_quant in ("fp8_pertensor", "nvfp4"):
        return attn_quant
    return None


def make_replicated_quant(expert_quant: str, attn_quant: str, in_f: int, out_f: int,
                          has_bias: bool = False):
    """Replicated linear for a dense projection: block-fp8 / per-tensor-fp8 / nvfp4 / bf16."""
    mode = _quantized_mode(expert_quant, attn_quant)
    if mode == "fp8_block":
        from freetoken.kernel.triton.fp8_block_linear import Fp8BlockLinear

        return Fp8BlockLinear(in_f, out_f, has_bias)
    if mode == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_f, out_f, has_bias)
    if mode == "nvfp4":  # compressed-tensors W4A16 attention o_proj / GDN out_proj
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

        return Nvfp4DenseLinear(in_f, out_f, has_bias)
    from freetoken.layers import LinearReplicated

    return LinearReplicated(in_f, out_f, has_bias=has_bias)


def make_row_parallel_quant(expert_quant: str, attn_quant: str, in_f: int, out_f: int,
                            has_bias: bool = False):
    """Row-parallel linear for a dense projection whose INPUT is column-sharded upstream.

    The bf16 path is the framework's ``LinearRowParallel``: it splits ``in_f`` across the TP
    group and all-reduces, because each rank then holds only a partial sum.

    ⛔⛆ The quantized paths are **refused** at TP>1 rather than sharded. None of the three
    quantized linears is TP-aware on its input axis and none all-reduces: block-FP8 carries a
    ``[out/128, in/128]`` scale grid, per-tensor FP8 a single scale fitted to the FULL row, and
    NVFP4 packs two values per byte along the input axis with its own group scales. Splitting
    ``in_f`` on any of them yields a layer of the right shape that computes the wrong number, so
    this raises instead. At TP=1 they are returned unchanged -- no collective, no behaviour change.
    """
    from freetoken.distributed import get_tp_info

    quant = _quantized_mode(expert_quant, attn_quant)
    tp_size = get_tp_info().size
    if quant is not None:
        if tp_size > 1:
            raise NotImplementedError(
                f"row-parallel {quant} is not implemented: the quantized linears are not "
                f"TP-aware on the input axis and do not all-reduce (tp_size={tp_size})"
            )
        return make_replicated_quant(expert_quant, attn_quant, in_f, out_f, has_bias)
    from freetoken.layers import LinearRowParallel

    return LinearRowParallel(in_f, out_f, has_bias=has_bias)


def make_replicated(config, in_f: int, out_f: int, has_bias: bool = False):
    """Config-driven replicated linear: ``Fp8BlockLinear`` under block-fp8, ``Fp8PerTensorLinear``
    under per-tensor-fp8 attention, ``Nvfp4DenseLinear`` under nvfp4, else ``LinearReplicated``."""
    return make_replicated_quant(
        getattr(config, "expert_quant", "none"), getattr(config, "attn_quant", "none"),
        in_f, out_f, has_bias,
    )


def make_col_merged(config, in_f: int, output_sizes: list[int], has_bias: bool = False):
    """Config-driven column-merged linear: ``Fp8BlockColMerged`` under block-fp8,
    ``Fp8PerTensorColMerged`` under per-tensor-fp8 attention, ``Nvfp4DenseColMerged`` under
    nvfp4, else ``LinearColParallelMerged``."""
    return make_col_merged_quant(
        getattr(config, "expert_quant", "none"), getattr(config, "attn_quant", "none"),
        in_f, output_sizes, has_bias,
    )


__all__ = [
    "make_col_merged_quant",
    "make_replicated_quant",
    "make_row_parallel_quant",
    "make_replicated",
    "make_col_merged",
]
