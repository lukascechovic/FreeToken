"""Qwen3.8-Flash-Next vision tower (``model.visual.*``): pixels -> soft tokens.

A vanilla ViT, ported against ``transformers`` 5.16.1's ``Qwen4ExpVisionModel`` (the
reference this file is verified element-wise against, not merely inspired by):

* ``patch_embed``  a Conv3d whose stride equals its kernel, i.e. exactly a ``Linear`` over the
  flattened ``in_channels * temporal_patch_size * patch_size**2`` = 1536 per-patch vector. The
  checkpoint's ``[H, 3, 2, 16, 16]`` weight is reshaped to ``[H, 1536]`` by the loader, so the
  reshape is paid once at load rather than per forward.
* ``pos_embed``    a learned ``[num_position_embeddings, H]`` table over a square
  ``48 x 48`` grid, **bilinearly** resampled (``align_corners=True``, border clamp) to each
  image's own ``(h, w)`` patch grid. ⛔ Not bicubic -- HF's ``interpolation_mode`` is
  ``"bilinear"``; a bicubic port would be wrong by a small, plausible-looking amount.
* 27 x pre-LN blocks  ``LayerNorm(bias)`` -> fused ``qkv`` (bias, no qk-norm, standard
  ``head_dim**-0.5`` scale) -> ``LayerNorm(bias)`` -> non-gated ``fc1 -> gelu_tanh -> fc2``.
* ``merger``       ``LayerNorm`` over the un-merged ``H`` (``use_postshuffle_norm=False``),
  then ``2x2`` patches folded into one ``4*H`` row -> ``fc1`` -> **exact (erf) GELU** ->
  ``fc2`` -> ``out_hidden_size``. ⚠ The block MLP's gelu is the *tanh* approximation and the
  merger's is not; HF uses ``ACT2FN[hidden_act]`` for one and a bare ``nn.GELU()`` for the other.

⭐ There is no separate projector: the merger *is* the projection into the text hidden size
(2560), so ``encode_images`` returns its output directly.

⭐⭐ No M-RoPE. ``text_config`` carries no ``rope_scaling``/``mrope_section``, so the
scheduler's existing 1-D positions are correct and only the *vision* 2-D RoPE lives here.

Input contract (unchanged from gemma4, see ``models/gemma4/model.py``): ``pixel_values``
``[N, P, 1536]`` and ``position_ids`` ``[N, P, 2]`` of ``(h, w)`` patch coordinates in
spatial-merge-block order, padded to a common ``P`` with ``(-1, -1)``. Attention is
bidirectional *within* an image and never crosses one, which the padded batch expresses as a
key mask rather than HF's ``cu_seqlens`` packing -- the two are equivalent, and the mask form
is what the existing contract can carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import os

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LayerNorm, LinearReplicated, OPList

if TYPE_CHECKING:
    from freetoken.models.qwen4_exp.config import VisionConfig

# HF hardcodes the vision LayerNorm epsilon (Qwen4ExpVisionBlock / Qwen4ExpVisionPatchMerger);
# it is not a config field, so neither is it one here.
_VISION_LAYERNORM_EPS = 1e-6


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _axis_taps(
    index: torch.Tensor, size: torch.Tensor, side: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear resampling taps for one axis: ``[..., 2]`` table indices and their weights.

    Closed form of ``torch.linspace(0, side - 1, size)[index]`` -- HF's ``align_corners=True``
    branch, with out-of-range taps clamped to the border exactly as ``F.interpolate`` does.
    ``size`` broadcasts against ``index`` so a ragged batch needs no per-image loop.
    """
    src = index.to(torch.float32) * (side - 1) / torch.clamp(size - 1, min=1).to(torch.float32)
    floor = torch.floor(src)
    offsets = torch.arange(2, device=index.device, dtype=torch.float32)
    taps = (floor.unsqueeze(-1) + offsets).long().clamp(0, side - 1)
    distance = (src.unsqueeze(-1) - floor.unsqueeze(-1) - offsets).abs()
    return taps, (1.0 - distance).clamp(min=0.0)


class _Embedding(BaseOP):
    """A bare lookup table. Exists only to place the checkpoint's ``pos_embed.weight`` key."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.weight = torch.empty(num_embeddings, embedding_dim)


class Qwen4ExpVisionPatchEmbed(BaseOP):
    """Conv3d-as-Linear over the flattened per-patch vector (see the module docstring)."""

    def __init__(self, vc: VisionConfig) -> None:
        self.proj = LinearReplicated(vc.patch_input_dim, vc.hidden_size, has_bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(pixel_values.to(self.proj.weight.dtype))


class Qwen4ExpVisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig) -> None:
        self.linear_fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x), approximate="tanh"))


class Qwen4ExpVisionAttention(BaseOP):
    """Bidirectional MHA over one image's patches: fused qkv with bias, no qk-norm, 2-D RoPE."""

    def __init__(self, vc: VisionConfig) -> None:
        self.num_heads = vc.num_heads
        self.head_dim = vc.head_dim
        self._scale = vc.head_dim**-0.5
        self.qkv = LinearReplicated(vc.hidden_size, 3 * vc.hidden_size, has_bias=True)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        N, P, _ = x.shape
        qkv = self.qkv.forward(x).view(N, P, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # each [N, P, heads, head_dim]

        # HF applies the vision rope in fp32 and casts back (apply_rotary_pos_emb_vision).
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        qf, kf = q.float(), k.float()
        q = ((qf * c) + (_rotate_half(qf) * s)).to(q.dtype)
        k = ((kf * c) + (_rotate_half(kf) * s)).to(k.dtype)

        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, scale=self._scale,
        )
        return self.proj.forward(o.transpose(1, 2).reshape(N, P, self.num_heads * self.head_dim))


class Qwen4ExpVisionBlock(BaseOP):
    def __init__(self, vc: VisionConfig) -> None:
        self.norm1 = LayerNorm(vc.hidden_size, eps=_VISION_LAYERNORM_EPS)
        self.norm2 = LayerNorm(vc.hidden_size, eps=_VISION_LAYERNORM_EPS)
        self.attn = Qwen4ExpVisionAttention(vc)
        self.mlp = Qwen4ExpVisionMLP(vc)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, attn_mask)
        return x + self.mlp.forward(self.norm2.forward(x))


class Qwen4ExpVisionPatchMerger(BaseOP):
    """``2x2`` patch fold + the projection into the text hidden size.

    ⚠ ``norm`` is applied *before* the fold (``use_postshuffle_norm=False``), so it norms over
    ``hidden_size``, not over ``4 * hidden_size`` -- which is why the checkpoint's
    ``merger.norm.weight`` is ``[1152]`` and not ``[4608]``. Getting this backwards still
    produces a merger that loads and answers colour questions.
    """

    def __init__(self, vc: VisionConfig) -> None:
        merged = vc.merged_hidden_size
        self.norm = LayerNorm(vc.hidden_size, eps=_VISION_LAYERNORM_EPS)
        self.linear_fc1 = LinearReplicated(merged, merged, has_bias=True)
        self.linear_fc2 = LinearReplicated(merged, vc.out_hidden_size, has_bias=True)
        self._merged_hidden_size = merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x).view(-1, self._merged_hidden_size)
        # HF's merger uses a bare nn.GELU() -- the exact erf form, NOT the tanh approximation
        # the block MLP uses.
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen4ExpVisionModel(BaseOP):
    """Pixels -> ``[num_soft_tokens, out_hidden_size]``, padding stripped, images in input order."""

    def __init__(self, vc: VisionConfig) -> None:
        self.patch_embed = Qwen4ExpVisionPatchEmbed(vc)
        self.pos_embed = _Embedding(vc.num_position_embeddings, vc.hidden_size)
        self.blocks = OPList([Qwen4ExpVisionBlock(vc) for _ in range(vc.depth)])
        self.merger = Qwen4ExpVisionPatchMerger(vc)
        self._vc = vc

    def _interpolated_pos_embed(
        self, pos: torch.Tensor, grid_h: torch.Tensor, grid_w: torch.Tensor
    ) -> torch.Tensor:
        """Resample the square learned table to each image's ``(h, w)`` grid. ``[N, P, hidden]``."""
        side = self._vc.num_grid_per_side
        h_taps, h_weights = _axis_taps(pos[..., 0], grid_h.unsqueeze(-1), side)
        w_taps, w_weights = _axis_taps(pos[..., 1], grid_w.unsqueeze(-1), side)
        # 2-D separable: the outer product of the per-axis taps gives 4 taps per patch.
        indices = (h_taps.unsqueeze(-1) * side + w_taps.unsqueeze(-2)).flatten(-2)
        weights = (h_weights.unsqueeze(-1) * w_weights.unsqueeze(-2)).flatten(-2)
        table = F.embedding(indices, self.pos_embed.weight)  # [N, P, 4, hidden]
        return (table.float() * weights.unsqueeze(-1)).sum(dim=-2)

    def _rope(self, pos: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_dim = self._vc.head_dim // 2
        inv = 1.0 / (
            self._vc.rope_theta
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32, device=pos.device) / spatial_dim)
        )
        freqs = (pos.float().unsqueeze(-1) * inv).flatten(-2)  # [N, P, 2 * spatial_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [N, P, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    # ⭐⭐ #841 bullet 3 -- BOUND THE VISION BATCH.
    #
    # The tower used to run on the whole request's images at once: `pixel_values [N, P, 1536]`
    # right-padded to a common P.  Every intermediate is then `[N, P, ...]` and the SDPA score
    # path is `[N, heads, P, P]`, so the tower's PEAK MEMORY IS LINEAR IN N.  Torch's caching
    # allocator must find a new, larger contiguous block every time a request's image count
    # grows, and the previous smaller ones are stranded as fragments it cannot reuse -- measured
    # at ~0.19 GiB x N on the `-visi` row, and the whole reason #841 exists.
    #
    # ⭐⭐ SPLITTING THE BATCH IS THE SAME COMPUTATION, because NOTHING IN THIS TOWER CROSSES
    # IMAGES.  Checked op by op: attention masks to each image's own patches
    # (`attn_mask = valid[:, None, None, :]`), the position embedding is interpolated from each
    # image's own `grid_h`/`grid_w`, and the merger ALREADY loops per image and returns a
    # `torch.cat` over them.  The `N` dimension is pure parallelism.
    #
    # ⛔⛔ SAME, BUT *NOT BIT-IDENTICAL* -- an earlier draft of #841 claimed it was, wrongly.
    # A different batch shape means different matmul tiling and SDPA reduction order, so results
    # move at float rounding: measured **max|diff| 1.5e-08 in float32** (~3e-07 relative) on
    # `test_vit_group_841.py`'s fixture.  ⭐ That it is ROUNDING and not a logic error is measured,
    # not assumed: the same fixture in float64 gives 2.8e-17, a ratio of 5.4e08 that tracks machine
    # epsilon.  The served tower runs in bf16, whose own rounding is orders of magnitude coarser.
    #
    # ⭐ Each group is also TRIMMED to its own longest image, so a group no longer pays the whole
    # request's max `P`.  At group size 1 that removes the padding entirely -- strictly less memory
    # AND less compute than the padded batch, since the attention no longer runs over pad columns.
    #
    # ⚠ The trade is tower parallelism: smaller groups serialise the encode.  Measure it.
    # ⛔ `0` / unset keeps the historical all-at-once behaviour, so this is inert until switched on.
    _VIT_GROUP_ENV = "FREETOKEN_VIT_GROUP"

    def _group_size(self) -> int:
        try:
            return max(0, int(os.environ.get(self._VIT_GROUP_ENV, "0")))
        except ValueError:  # noqa: BLE001 -- a bad env value must not take the tower down
            return 0

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        n = pixel_values.shape[0]
        g = self._group_size()
        if g <= 0 or n <= g:
            return self._forward_group(pixel_values, position_ids)
        outputs = []
        for s in range(0, n, g):
            outputs.append(
                self._forward_group(pixel_values[s:s + g], position_ids[s:s + g])
            )
        return torch.cat(outputs, dim=0)

    def _forward_group(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        valid = (position_ids >= 0).all(dim=-1)  # [N, P] -- (-1, -1) marks padding
        # Trim to THIS group's longest image. Valid patches are a contiguous prefix (asserted
        # below), so the tail being dropped is padding only and the result is unchanged.
        keep = int(valid.sum(dim=1).amax().item()) if valid.numel() else 0
        if keep and keep < valid.shape[1]:
            pixel_values = pixel_values[:, :keep]
            position_ids = position_ids[:, :keep]
            valid = valid[:, :keep]
        pos = position_ids.clamp(min=0)
        # Each image's own patch grid, read off its position ids (padding excluded).
        masked = torch.where(valid.unsqueeze(-1), position_ids, torch.full_like(position_ids, -1))
        grid_h = masked[..., 0].amax(dim=1) + 1
        grid_w = masked[..., 1].amax(dim=1) + 1

        h = self.patch_embed.forward(pixel_values)
        h = h + self._interpolated_pos_embed(pos, grid_h, grid_w).to(h.dtype)
        h = h.masked_fill(~valid.unsqueeze(-1), 0.0)

        cos, sin = self._rope(pos, h.dtype)
        attn_mask = valid[:, None, None, :]  # [N, 1, 1, P] True = attend
        for block in self.blocks.op_list:
            h = block.forward(h, cos, sin, attn_mask)

        # Merge per image over its own valid patches. They are a contiguous prefix because the
        # batch is right-padded, and a multiple of merge**2 because the processor emits whole
        # spatial-merge blocks -- both asserted rather than assumed.
        merge_unit = self._vc.spatial_merge_size**2
        counts = valid.sum(dim=1).tolist()
        outputs = []
        for i, count in enumerate(counts):
            assert bool(valid[i, :count].all()) and not bool(valid[i, count:].any()), (
                f"image {i}: valid patches are not a contiguous prefix -- the padded batch must "
                "be right-padded with (-1, -1)"
            )
            assert count % merge_unit == 0, (
                f"image {i}: {count} patches is not a multiple of {merge_unit}; the processor "
                "emits whole spatial-merge blocks"
            )
            outputs.append(self.merger.forward(h[i, :count]))
        return torch.cat(outputs, dim=0)


__all__ = ["Qwen4ExpVisionModel"]
