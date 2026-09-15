"""#801 round 5 bullet 7b — α: turning `ShadowTracker`'s per-step verdicts into a RATE.

Bullets 3 and 4 answer "did the head's guess for THIS request land on THIS step". Nothing
aggregated that over a run, and α — the number the whole issue turns on — is an aggregate. This
module is that aggregate, and it is deliberately the smallest thing that can be one.

⭐ **A NEW engine file**, `models/qwen4_exp/accept.py`, same discipline as `shadow.py`: no `.orig`
beside it, BIND-MOUNTED, in no image. Like `shadow.py` it imports neither `torch` nor `freetoken` —
everything it sees is already plain Python — so `test_accept_801.py` runs on HOST python and is
NOT in `conftest.py`'s `_TORCH_ONLY` list.

⭐⭐ **Why it is keyed per REQUEST and not one pooled number.** α is content-dependent: the priors
`README.md` banks span **40.7–81.5 %** across content on llama.cpp's 27B head. One pooled rate over
a whole load would report the middle of that spread as if it were a measurement, and the round's
own gate (≥ +15 % at TP=2) sits inside the spread. ⇒ one row per request, pooled only in the
analyser, where the pooling is visible and the spread can be printed beside it.

⛔⛆ **THE LAST REQUEST OF A RUN NEVER RETIRES ITSELF.** A uid is reported when it stops appearing
in a step's live `uids` — which is the next request's first decode step. There is no shutdown hook
to flush from, and adding one to `engine.py` for a research probe is not worth an every-served-row
edit. ⇒ **the probe sends one extra throwaway request at the end of a run**, whose only job is to
be the step that retires the real last one; `analyse_alpha_801.py` drops it. :meth:`drain` exists
for a caller that can flush properly, and today nothing can.

⚠ **`mass` is not the same quantity as `accepted`, and mixing them would be the round's worst
error.** `accepted` is TOKEN EQUALITY — did the draft's id equal the target's. That is exactly α
under greedy, where both sides are deterministic. Under the served sampler (`top_k 20, top_p
0.95`) it is NOT the acceptance the verify path (issue bullet 5) will get: that path is rejection
sampling, which accepts with probability ``Σ_x min(p(x), q(x))`` over the two FILTERED
distributions, and counts a draw that merely differed as a rejection only part of the time. Token
equality therefore UNDERSTATES the sampled arm's α, and α is what the gate is read off. So both
are carried, separately, per request, and `draft.py::acceptance_mass` computes the second one.
⭐ Under greedy the two coincide (both distributions are one-hot), which is a free cross-check the
analyser prints rather than an assumption anything here makes.
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence


class _Row:
    """One live request's running counters. Plain attributes: this is a tally, not a model."""

    __slots__ = ("uid", "checked", "accepted", "mass_sum", "mass_n")

    def __init__(self, uid: Hashable) -> None:
        self.uid = uid
        self.checked = 0
        self.accepted = 0
        self.mass_sum = 0.0
        self.mass_n = 0

    def as_dict(self) -> dict:
        """⚠ `rate` and `mass_mean` read ``None``, never ``0.0``, before anything was counted —
        the same rule `timing.py::RollingStats` follows, and for the same reason: a zero here is
        a real and very bad measurement, so it must not be what "no measurement" looks like."""
        return {
            "uid": self.uid,
            "checked": self.checked,
            "accepted": self.accepted,
            "rate": (self.accepted / self.checked) if self.checked else None,
            "mass_n": self.mass_n,
            "mass_mean": (self.mass_sum / self.mass_n) if self.mass_n else None,
        }


class AcceptanceCounter:
    """Per-request α, reported when the request leaves the batch.

    Slot reuse and mid-batch finishes need no special handling for the same reason
    `ShadowTracker` needs none: liveness is read off the step's own `uids`, never remembered.
    """

    def __init__(self) -> None:
        self._rows: dict[Hashable, _Row] = {}

    def step(
        self,
        uids: Sequence[Hashable],
        accepted: Mapping[Hashable, bool],
        mass: Mapping[Hashable, float] | None = None,
    ) -> list[dict]:
        """Fold one decode step in; return a row per request that just RETIRED.

        ``uids`` is this step's live batch. ``accepted`` is `ShadowTracker.step`'s return — a
        SUBSET of ``uids`` (a request on its first sighting has no held prediction to check).
        ``mass`` is the same subset's ``Σ min(p, q)``, or ``None`` on a greedy arm where token
        equality already is the acceptance.

        ⛔ Retirement is computed BEFORE this step's counters are folded in, so a request that
        finished last step is reported with its own totals and not with a successor's.
        """
        live = set(uids)
        retired = [row.as_dict() for uid, row in self._rows.items() if uid not in live]
        self._rows = {uid: row for uid, row in self._rows.items() if uid in live}
        for uid in uids:
            if uid not in self._rows:
                self._rows[uid] = _Row(uid)
        for uid, ok in accepted.items():
            row = self._rows[uid]
            row.checked += 1
            row.accepted += 1 if ok else 0
        for uid, value in (mass or {}).items():
            row = self._rows[uid]
            row.mass_sum += float(value)
            row.mass_n += 1
        return retired

    def drain(self) -> list[dict]:
        """Report and forget every still-live request.

        ⛔ Nothing calls this today — see the module docstring. It exists so that the day
        `engine.py` grows a shutdown hook, the fix is one call and not a rethink, and so that the
        probe's throwaway-final-request trick is visibly a WORKAROUND for a missing flush rather
        than the design.
        """
        rows = [row.as_dict() for row in self._rows.values()]
        self._rows = {}
        return rows


__all__ = ["AcceptanceCounter"]
