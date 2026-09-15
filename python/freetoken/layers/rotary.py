from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


# ── #801 overlay marker ──────────────────────────────────────────────────────────────
# This file is `layers/rotary.py` from image `llm-server/freetoken-gfx1201:2026-09-09-agree-0022`
# (md5 be50e942799f264c10414d28c2c3340d, 234 lines) BIND-MOUNTED over the installed package, plus this block
# and the CPU branch inside `RotaryEmbedding.forward`. ⛔ It is NOT a patch in the Dockerfile
# ladder and NOT in any image.
#
# ⛔⛆ WHY IT EXISTS — the SECOND blocker of the same shape as `overlay/norm.py`'s, found by
#   running round 4 bullet 4, not by reading. `RotaryEmbedding.forward` dispatches
#   unconditionally to flashinfer's or triton's `apply_rope_with_cos_sin_cache_inplace`; the
#   triton one opens with `assert query.is_cuda and key.is_cuda and positions.is_cuda`. So
#   `Qwen4ExpAttention` — which the MTP head reuses whole — could not run off a GPU at all, and
#   the round's agreed EXACT off-GPU gate cannot cover the head's layer without this branch.
#
# ⭐ WHAT THE BRANCH IS DIFFED AGAINST, so it is not merely my second opinion of my first.
#   The cos/sin CACHE is the engine's, untouched: `__init__` builds `_cos_sin_cache` above and
#   this branch only consumes it. `head_ref_801.apply_rope` builds its frequencies from scratch
#   out of the HF definition and knows nothing about the cache's layout. The gate in
#   `test_mtp_801.py` runs the engine's attention against that reference, so an agreement is a
#   real cross-check of the cache layout, not a restatement.
#   ⚠ The rotation itself is transcribed from `kernel/triton/rope.py::_rope_tiled`: cos is the
#   FIRST half of each cache row and sin the second, both over `rotary_dim`; NeoX pairs `d` with
#   `d + rotary_dim/2`, interleave pairs `2d` with `2d+1`; fp32 math, cast back at the store;
#   dims past `rotary_dim` pass through untouched.
#
# ⚠ No marker print, for the reasons `overlay/norm.py` gives: a failed mount is loud on CPU and
#   a no-op on GPU (this branch is dead code there), and the suite asserts the mount by md5.


def apply_rope_with_cos_sin_cache_torch(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Pure-torch `apply_rope_with_cos_sin_cache_inplace`, for tensors that are not on a GPU.

    ``query``/``key`` are ``[nnz, num_heads*head_size]`` and are rotated IN PLACE over their
    first ``rotary_dim`` dims; ``cos_sin_cache`` is ``[max_position, rotary_dim]`` fp32.
    """
    if cos_sin_cache.dtype is not torch.float32:
        raise ValueError("cos_sin_cache should be float32")
    nnz = query.shape[0]
    if nnz == 0:
        return
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2

    row = cos_sin_cache[positions.to(torch.int64)]  # [nnz, rotary_dim]
    cos = row[:, None, :half]  # [nnz, 1, half], broadcast over heads
    sin = row[:, None, half:]

    if is_neox:
        d0 = torch.arange(half, device=query.device)
        d1 = d0 + half
    else:  # GPT-J interleave: adjacent pairs
        d0 = torch.arange(0, rotary_dim, 2, device=query.device)
        d1 = d0 + 1

    for tensor in (query, key):
        heads = tensor.view(nnz, -1, head_size)
        x0 = heads[..., d0].float()
        x1 = heads[..., d1].float()
        # both halves are read before either is written: the rotation mixes them
        out0 = (x0 * cos - x1 * sin).to(tensor.dtype)
        out1 = (x1 * cos + x0 * sin).to(tensor.dtype)
        heads[..., d0] = out0
        heads[..., d1] = out1


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        proportional: bool = False,
        attention_factor: float = 1.0,
        is_neox: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # NeoX (half-rotation, HF default) vs GPT-J interleaved (adjacent pairs,
        # ``rope_interleave`` models: GLM MLA lineage). Both underlying kernels
        # accept the flag; the cos/sin cache layout is identical.
        self.is_neox = is_neox
        if proportional:
            assert 0 < rotary_dim <= head_size
            assert rotary_dim % 2 == 0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size)
            )
            if rotary_dim < head_size:
                inv_freq[rotary_dim // 2 :] = 0.0
        else:
            # Standard (NeoX) rope. Supports partial rotary (rotary_dim < head_size):
            # rope is applied to the first ``rotary_dim`` dims of each head, the rest pass
            # through. Frequencies are spaced over ``rotary_dim`` (matches HF default
            # partial rope, e.g. Qwen3.5 partial_rotary_factor, MiniMax-M2's
            # ``apply_rotary_pos_emb``). Full rope is rotary_dim == head_size and is
            # unaffected. ``head_size`` is passed to flashinfer separately so it rotates
            # only the first ``rotary_dim`` dims.
            assert 0 < rotary_dim <= head_size
            assert rotary_dim % 2 == 0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
            )
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import apply_rope_with_cos_sin_cache_inplace
        else:
            from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # #801: the kernels are GPU-only; off a GPU take the torch chain (see the overlay marker)
        if not query.is_cuda:
            apply_rope_with_cos_sin_cache_torch(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self._cos_sin_cache,
                is_neox=self.is_neox,
            )
            return query, key
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
            is_neox=self.is_neox,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox: bool = True,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)

        case "proportional":
            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                proportional=True,
                is_neox=is_neox,
            )

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(
                head_dim, rotary_dim, max_position, base, post_process, is_neox=is_neox
            )

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def get_mscale(scale: float, mscale: float = 1.0) -> float:
                if scale <= 1:
                    return 1.0
                return 0.1 * mscale * math.log(scale) + 1.0

            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                mscale = rope_scaling.get("mscale")
                mscale_all_dim = rope_scaling.get("mscale_all_dim")
                # Truthiness, not presence: HF falls back to get_mscale(factor) when
                # mscale_all_dim is 0 (a real DeepSeek-lineage default).
                if mscale and mscale_all_dim:
                    attention_factor = get_mscale(factor, mscale) / get_mscale(
                        factor, mscale_all_dim
                    )
                else:
                    attention_factor = get_mscale(factor)

            def _find_correction_dim(num_rotations: float) -> float:
                return (
                    rotary_dim
                    * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = _find_correction_dim(beta_fast)
            high = _find_correction_dim(beta_slow)
            if rope_scaling.get("truncate", True):
                low = math.floor(low)
                high = math.ceil(high)
            low = max(low, 0)
            # rotary_dim - 1, per HF's find_correction_range and this repo's own faithful copy in
            # models/deepseek_v4/ops.py. Clamping to rotary_dim//2 - 1 instead forces the ramp to
            # reach 1.0 at the last entry, fully interpolating the longest-wavelength dims that
            # the reference deliberately leaves partly extrapolated.
            high = min(high, rotary_dim - 1)
            if low == high:  # HF nudges instead of flooring the gap at 1 ("truncate": false)
                high += 0.001

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                post_process,
                attention_factor=float(attention_factor),
                is_neox=is_neox,
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox: bool = True,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
