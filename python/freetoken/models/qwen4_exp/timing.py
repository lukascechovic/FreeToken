"""#801 round 5 bullet 5 — `t_draft`, eager: how long the real draft step (bullet 2) takes.

⭐ **A NEW engine file**, `models/qwen4_exp/timing.py`, same discipline as `draft.py`/`shadow.py`:
no `.orig` beside it, BIND-MOUNTED, in no image.

Two pieces, deliberately separate (see the round's own checklist for why this is not folded into
`shadow.py::ShadowTracker`): that tracker is uid-keyed and per-request; a decode step's draft
latency is one scalar per step, batch-wide. Composed only where something needs to print or bank
both together (bullet 7), not here.

- `RollingStats` needs neither `torch` nor `freetoken` — plain Python floats in, floats out — so
  it runs on the HOST python same as `shadow.py` (`test_timing_801.py`, not in `conftest.py`'s
  `_TORCH_ONLY`).
- `time_draft_call` wraps a call in a `torch.cuda.Event(enable_timing=True)` pair. It cannot run
  on the HOST (no torch) or in the CPU-only wiring container (no CUDA device at all — constructing
  a `torch.cuda.Event` there raises immediately) — real values wait for a box load (bullet 7). The
  CPU-only wiring suite monkeypatches this function away entirely, same as it already does for
  `draft_next_token_ids`.

⚠ **The sync is deliberate, not an oversight.** `elapsed_time()` needs both events complete, and
the cheapest way to guarantee that is a `.synchronize()` right after `ended.record()` — a real
device stall, paid every shadow-mode decode step. A lagged, sync-free read (piggyback on whatever
host sync the scheduler already does before it can return `next_tokens_cpu`) would be faster, but
`t_draft` EAGER is exactly the baseline bullet 6's captured graph has to beat — understating eager's
own cost here would make that comparison lie. This path is decode-only and gated behind
`FREETOKEN_MTP801_SHADOW=1` (never the deployed default), so the honest stall is the right call.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, TypeVar

T = TypeVar("T")


class RollingStats:
    """min/mean/p50/p90 over the last `window` samples — old samples age out, never averaged in
    forever. Every stat reads `None` before the first `update()`, rather than faking a `0.0` a
    caller could mistake for a real fast measurement."""

    def __init__(self, window: int = 256) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def min(self) -> float | None:
        return min(self._values) if self._values else None

    @property
    def mean(self) -> float | None:
        return sum(self._values) / len(self._values) if self._values else None

    @property
    def p50(self) -> float | None:
        return self._percentile(50)

    @property
    def p90(self) -> float | None:
        return self._percentile(90)

    def _percentile(self, p: int) -> float | None:
        if not self._values:
            return None
        ordered = sorted(self._values)
        index = round(p / 100 * (len(ordered) - 1))
        return ordered[index]


def time_draft_call(fn: "Callable[[], T]") -> "tuple[T, float]":
    """Run `fn`, return its result alongside the elapsed device time in milliseconds.

    ⛔⛆ Constructing a `torch.cuda.Event` with no CUDA device visible raises immediately — the
    CPU-only wiring container has none at all. Real values only ever come from a box load.
    """
    import torch

    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    result = fn()
    ended.record()
    ended.synchronize()
    return result, started.elapsed_time(ended)


__all__ = ["RollingStats", "time_draft_call"]
