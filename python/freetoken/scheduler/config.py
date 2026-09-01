from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    # A multimodal prompt must fit ONE prefill chunk (the tower's soft tokens are scattered
    # in a single pass), so it is capped separately from a text prompt. None = whatever the
    # prefill budget allows; a number lowers it. The number itself is a policy decision, not
    # an engine fact -- it is set on the command line, never assumed here.
    max_multimodal_prompt_tokens: int | None = None
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    def multimodal_prompt_limit(self, prefill_budget: int | None = None) -> int:
        """The longest prompt this server will admit WITH an image.

        The hard half is the prefill budget (a chunked multimodal prompt is unimplemented
        upstream); the soft half is the operator's cap. The lower wins.

        Called with the live budget by the scheduler, and without it by the frontend, which
        cannot see the cache manager's chunk cap and so falls back to ``max_extend_tokens``.
        On a model whose cache caps the chunk below that (the sliding-window pools), the two
        answers differ and a prompt between them is refused by the scheduler after the stream
        has already started. Setting ``max_multimodal_prompt_tokens`` at or below the chunk
        cap collapses the gap: both callers then return the same number.
        """
        limit = self.max_extend_tokens if prefill_budget is None else prefill_budget
        if self.max_multimodal_prompt_tokens is not None:
            limit = min(limit, self.max_multimodal_prompt_tokens)
        return limit

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
