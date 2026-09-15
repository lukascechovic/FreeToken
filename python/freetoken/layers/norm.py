from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import rmsnorm
        else:
            from freetoken.kernel.triton.norm import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class GemmaRMSNorm(BaseOP):
    """Gemma4-style RMSNorm backed directly by sgl_kernel.

    Gemma4 scales by the raw checkpoint weight. ``with_scale=False`` uses a
    runtime ones vector that is intentionally not part of ``state_dict``.
    """

    def __init__(self, size: int, eps: float, with_scale: bool = True) -> None:
        from freetoken.kernel.backend import is_sgl_kernel_installed

        if is_sgl_kernel_installed():
            from sgl_kernel import fused_add_rmsnorm, rmsnorm
        else:
            from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.size = size
        self.with_scale = with_scale
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
        if with_scale:
            self.weight = torch.empty(size)
        else:
            self._ones_weight: torch.Tensor | None = None

    def _kernel_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_scale:
            return self.weight
        if self._ones_weight is None:
            self._ones_weight = torch.ones(self.size, device=x.device, dtype=x.dtype)
        return self._ones_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.rmsnorm(x, self._kernel_weight(x), self.eps)
        original_shape = x.shape
        out = self.rmsnorm(
            x.contiguous().reshape(-1, original_shape[-1]),
            self._kernel_weight(x),
            self.eps,
        )
        return out.reshape(original_shape)

    def forward_add_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fused_add_rmsnorm(x, residual, self._kernel_weight(x), self.eps)
        return x, residual


# ── #801 overlay marker ──────────────────────────────────────────────────────────────
# This file is `layers/norm.py` from image `llm-server/freetoken-gfx1201:2026-09-09-agree-0022`
# (md5 8dec317a893facd831cd45a7ff8e19d7, 195 lines) BIND-MOUNTED over the installed package, plus
# this block and the CPU branch inside `GemmaPlusOneRMSNorm`. ⛔ It is NOT a patch in the
# Dockerfile ladder and NOT in any image.
#
# ⛔⛆ WHY IT EXISTS. `GemmaPlusOneRMSNorm` dispatched UNCONDITIONALLY to flashinfer's or triton's
#   `gemma_rmsnorm`, so it could not run off a GPU at all: in a CPU-only container it raised
#   `RuntimeError: 0 active drivers ([])`. It is also the class the MTP head's two fusion norms
#   must be (#801 round 4, finding 2 -- ONE statistic over all `hc_count*hidden`, not one per
#   stream), so the round's agreed EXACT off-GPU gate had no interpreter until this branch existed.
#   `GroupedPlusOneRMSNorm` (models/qwen4_exp/hc.py:56-60) already had exactly this branch; the
#   shape below is copied from it, and `hc.grouped_plus_one_rms_norm(..., num_groups=1)` is the
#   independent implementation the new one is gated against.
#
# ⚠ NO MARKER PRINT, unlike `overlay/weight.py`. That file prints because a mount that silently
#   did not take is INVISIBLE there -- the row loads the image's weights and every log line looks
#   like the arm you launched (#866). Here a failed mount is loud on CPU (the driver error above)
#   and a no-op on GPU (the branch is dead code there), and this module is imported by every norm
#   in the engine, so a print would be noise on the load. The suite asserts the mount took by md5
#   instead: `test_mtp_801.py::TestTheOverlayTookEffect`.
#
# ⚠ `GemmaPlusOneRMSNormFused` is deliberately NOT given a branch: nothing this round constructs
#   one off-GPU, and an unexercised CPU path is a claim, not a capability.


def gemma_plus_one_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the WHOLE last dim on an fp32 statistic, then scale by (1+w).

    The pure-torch body of :class:`GemmaPlusOneRMSNorm`, written out so the CPU path and the
    (1+w) semantics can be tested without a GPU. Mirrors ``hc.grouped_plus_one_rms_norm``: fp32
    intermediates, cast back at the store, so the two agree to fp32 rounding.
    """
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


class GemmaPlusOneRMSNorm(BaseOP):
    """(1 + w)-scaled RMSNorm (Gemma semantics: the checkpoint stores ``scale - 1``
    and the effective multiplier is ``1 + weight``, added in fp32 at runtime --
    never folded into the bf16 weight, which would round away the precision the
    format exists to keep). Used by MiniMax-M3 (``use_gemma_norm``) for the decoder
    layernorms, the per-head q/k norms, and the indexer q/k norms.

    Per-head 3D inputs are collapsed to 2D before the kernel call: flashinfer's
    ``gemma_rmsnorm`` CUDA binding is CHECK_DIM(2) (it rejects 3D outright on
    wheels without the CuTe path), and the per-head weight makes the 2D view
    exactly equivalent. The M3 call sites pass contiguous buffers, so the views
    are free; a non-contiguous 3D input is rejected rather than silently copied
    (an in-place norm on a copy would be dropped).
    """

    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer.norm import gemma_rmsnorm
        else:
            from freetoken.kernel.triton.norm import gemma_rmsnorm

        self.eps = eps
        self.size = size
        self.weight = torch.empty(size)
        self.gemma_rmsnorm = gemma_rmsnorm

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        assert x.is_contiguous(), "per-head gemma norm needs a contiguous buffer"
        return x.view(-1, self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # #801: the kernels are GPU-only; off a GPU take the torch chain (see the overlay marker)
        if not x.is_cuda:
            return gemma_plus_one_rms_norm(x, self.weight, self.eps)
        return self.gemma_rmsnorm(self._flat(x), self.weight, self.eps).view(x.shape)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if not x.is_cuda:
            # computed in full first: the fp32 chain reads every element of the row it writes
            x.copy_(gemma_plus_one_rms_norm(x, self.weight, self.eps))
            return
        flat = self._flat(x)
        self.gemma_rmsnorm(flat, self.weight, self.eps, out=flat)


class GemmaPlusOneRMSNormFused(BaseOP):
    """(1 + w)-scaled RMSNorm with the fused-add-residual API of ``RMSNormFused``
    (Gemma semantics, see :class:`GemmaPlusOneRMSNorm`). Drop-in for the decoder
    layernorm seam: ``forward(x, residual)`` returns ``(normed, residual)``."""

    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
        else:
            from freetoken.kernel.triton.norm import (
                gemma_fused_add_rmsnorm,
                gemma_rmsnorm,
            )

        self.eps = eps
        self.weight = torch.empty(size)
        self.gemma_rmsnorm = gemma_rmsnorm
        self.gemma_fused_add_rmsnorm = gemma_fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.gemma_rmsnorm(x, self.weight, self.eps), x
        self.gemma_fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import fused_add_rmsnorm, rmsnorm
        else:
            from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class LayerNorm(BaseOP):
    """Standard (mean-subtracting) LayerNorm with an optional bias.

    ``layers/norm.py`` otherwise ships only RMSNorm variants, because every text decoder
    served here is RMSNorm-based. Qwen4-Exp's vision tower is a vanilla ViT and uses
    LayerNorm *with* bias in all three places it norms (``blocks.N.norm{1,2}`` and
    ``merger.norm``), so the primitive has to exist before the tower can be ported.

    Backed by ``F.layer_norm``, which accumulates in fp32 for a bf16 input -- matching HF's
    ``nn.LayerNorm`` exactly, so a ported tower is comparable against the reference
    element-wise rather than merely in aggregate.
    """

    def __init__(self, size: int, eps: float, has_bias: bool = True) -> None:
        self.size = size
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.layer_norm(x, (self.size,), self.weight, self.bias, self.eps)
