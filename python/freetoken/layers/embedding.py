# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `layers/embedding.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 9c138b30f0641b4aa562d7f7201558ac,
# 125 lines) BIND-MOUNTED over the installed package, plus this block and ONE condition in
# `ParallelLMHead.forward`. ⛔ In no image and in no Dockerfile ladder -- it must be in
# `arm_mtp_801.sh`'s OVERLAY and ORIGS lists or the row runs the image's copy silently (#866).
#
# ⛔⛆ WHY IT EXISTS (round 6 bullet 6, and it was NOT in the round's plan). `forward`'s
#   tensor-parallel gather has a fast path guarded on `bs == 1`, where `bs` is `batch.size` --
#   REQUESTS, not rows. It is only the general path's answer when the forward carried ONE ROW.
#   A #801 verify step forwards TWO rows for one request, and at TP=2 that reshape concatenates
#   RANK 0's two tokens where the general path interleaves ONE TOKEN's two ranks: one row out
#   instead of two, and the values are not the row's logits.
#
# ⛔ SILENT ONLY AT TP=1, AND THIS ROUND IS TP=2 THROUGHOUT. bs=1 + TP=2 is the round's own
#   measured shape, so the verify graph cannot even be CAPTURED through the image's version --
#   `GraphCaptureBuffer.logits[:rows] = model.forward()` gets one row where it expects two.
#   Gated by differential against `.orig` in `test_graph_capture_801.py
#   ::TestTheLmHeadReturnsOneRowPerTokenNotPerRequest`, which also pins that every shape served
#   today (one row, any TP) comes back byte-identical.

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        # #801: ONE ROW, not one request. `all_gather` returns [tp * rows, vocab_tp] rank-major,
        # so `view(1, -1)` is the interleave the general path below does only when rows == 1.
        # A verify step (bs == 1, two rows) would otherwise get rank 0's two TOKENS glued
        # together, at half the row count, with nothing erroring.
        if input_shape[0] == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]