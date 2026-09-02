from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    # Prefix-cache key stream for an image request (scheduler/mm_key.py); None = real ids.
    cache_key_ids: torch.Tensor | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def cache_ids(self) -> torch.Tensor:
        """What the prefix cache keys on: the marker stream for an image request, else ``input_ids``."""
        return self.input_ids if self.cache_key_ids is None else self.cache_key_ids

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
