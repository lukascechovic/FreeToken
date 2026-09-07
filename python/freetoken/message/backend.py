from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). Only used by
    # the in-process offline path; the online path leaves it None and sends the pixels below.
    mm_embeds: torch.Tensor | None = None
    # The online multimodal path. A GPU tensor cannot cross the wire, and the vision tower lives
    # with the model, so the tokenizer worker sends the *preprocessed pixels* and the scheduler
    # runs `encode_images` on admission. The prompt ceiling in
    # `Scheduler._multimodal_ceiling_error` binds whichever field carries an image.
    # ⭐⭐ #890 (patch 0018): CPU, PACKED -- [sum(P), D] float32 and [sum(P), 2] int64, every
    # image's patches back to back with no padding rows, plus the counts that say where each
    # image ends. This used to be the tower's right-padded [N, P_max, D] tray, whose cost is
    # `n_images x P_max` and is therefore unbounded by the soft-token cap (#883 rung B6 killed
    # the row at 51% of it). `scheduler/mm_encode.py` splits the run per image at the device.
    pixel_values: torch.Tensor | None = None
    image_position_ids: torch.Tensor | None = None
    image_patch_counts: List[int] | None = None
    # Set by the scheduler on admission (never on the wire): ``input_ids`` with each image's
    # first placeholders replaced by pixel-hash markers -- the prefix cache's key stream
    # (scheduler/mm_key.py). None keeps the cache bypass for image requests.
    cache_key_ids: torch.Tensor | None = None


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
