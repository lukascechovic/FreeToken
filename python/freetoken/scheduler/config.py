from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    # An optional operator cap on the longest prompt admitted WITH an image. None (the
    # default) = an image prompt is bounded by the context length alone, like a text prompt:
    # the scheduler chunks a multimodal prompt across prefill passes exactly as it chunks text
    # (PrefillAdder hands each chunk its own soft-token rows). The number is a policy decision,
    # never an engine fact -- it is set on the command line, never derived here.
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

    def multimodal_prompt_limit(self) -> int | None:
        """The operator's cap on a prompt carrying an image, or None when there is none.

        Pure policy: the engine no longer needs an image prompt to fit one prefill chunk, so
        without a cap the only ceiling is the ordinary context check every prompt gets. The
        frontend pre-check and the scheduler's admission check both read this one number, so
        they cannot disagree (the live prefill budget used to make them).
        """
        return self.max_multimodal_prompt_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
