"""#801 round 5 bullet 3 — the cross-step shadow tracker: did the draft's guess land.

``t_mtp``'s α term (fraction of draft predictions the real sampler agrees with) needs, every
decode step, "did what the head guessed LAST step match what the sampler chose THIS step" — per
live request, surviving slot reuse and requests finishing mid-batch. `ShadowTracker` answers that
with no dependency on `torch` or `freetoken` at all: everything it sees is already plain Python
(uids as `int`, token ids as whatever the caller compares with `==`), so this module's own tests
(``test_shadow_801.py``) run on host python.

⭐ **A NEW engine file**, `models/qwen4_exp/shadow.py`, same as `draft.py`: no `.orig` beside it,
BIND-MOUNTED, in no image. ⛔ Living under `overlay/` is what makes it reachable from
`Qwen4ExpForCausalLM.mtp_shadow_step` (bullet 4, `model.py`) — the SAME file is also imported
directly on the host for its own tests (`sys.path` trick, not a package import), since it needs
neither `torch` nor `freetoken` to run: one source, two runners.

See ``test_shadow_801.py``'s own module docstring for why the one-step lag is not optional.
"""

from __future__ import annotations

from typing import Hashable, Sequence


class ShadowTracker:
    """Rebuilt fresh from the live batch every step — eviction and slot reuse are free.

    No state survives beyond what the last `step()` call put in it: a uid absent from this
    step's `uids` is simply not carried forward, whether it finished or its slot got reused by
    an unrelated new request.
    """

    def __init__(self) -> None:
        self._pending: dict[Hashable, object] = {}

    def step(
        self,
        uids: Sequence[Hashable],
        sampled_ids: Sequence[object],
        predicted_ids: Sequence[object],
    ) -> dict[Hashable, bool]:
        """Check this step's real tokens against LAST step's held predictions, then hold this
        step's own predictions for the next call.

        ``uids``, ``sampled_ids`` and ``predicted_ids`` are the same length and row-aligned: index
        ``i`` is one live request. A uid with no held prediction (its first sighting, or the
        request that finished last step) has nothing to check yet and is simply absent from the
        returned map — never reported as a checked miss.
        """
        accepted = {
            uid: self._pending[uid] == sampled
            for uid, sampled in zip(uids, sampled_ids)
            if uid in self._pending
        }
        self._pending = dict(zip(uids, predicted_ids))
        return accepted


__all__ = ["ShadowTracker"]
