from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch

# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `scheduler/status.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 643f6a26ddd300d8424594931f068f88,
# 144 lines) plus TWO edits:
#   1. round 7: `report_batch` takes an optional `generated_tokens` and threads it to
#      `_report_decode`, which adds it to the window instead of `len(batch.reqs)`.
#   2. #967: the decode line prints its own NUMERATOR and `gap` beside the rate.
# ⛔ In no image, in no
# Dockerfile ladder: absent from `arm_mtp_801.sh`'s OVERLAY and ORIGS lists, the scheduler runs the
# image's copy and every log line looks like the arm we think we launched (#866). The differential
# is gated by `test_status_801.py`, on the HOST.
#
# ⛔⛆ WHY THE STATUS REPORTER IS TOUCHED AT ALL. `_report_decode` counts
# `self._decode_generated_tokens += len(batch.reqs)` -- ONE token per request per forward. That is
# the image's own invariant and nothing in the image states it. It is true for a plain decode row
# and FALSE for a speculating one, which commits 1 OR 2 tokens per request per forward. ⇒ on the
# `-mtp-` row `gen throughput (token/s)` is STEPS/s, not tokens/s. Measured 2026-09-14, same
# 599-token request, each row against its OWN client-side reading:
#       -mtp-…-np4-tp2-   client 35.20 tok/s   engine 20.57 (median, 9 windows)   ratio 1.711
#       -visi-…-np4-tp2-  client 33.28 tok/s   engine 33.91 (median, 13 windows)  ratio 0.981
# ⚠ llama-swap computes NOTHING here -- all six of its `metrics` rows for this row read
# `tokens_per_second: -1`. The dashboard number IS this log line, so this is the only place to fix.
# ⛔ Do NOT read 35.20 > 33.28 as MTP being faster: the control is the STOCK engine and the `-mtp-`
# row runs 24 overlay files (⚠ 23 when that pair was taken -- `status.py` itself is the 24th and
# changes no compute), so that pair measures MTP + overlay. Round 6's matched pair is -1.0 %.
#
# ⭐ THE `None` DEFAULT IS WHAT MAKES THIS SAFE ON AN EVERY-SERVED-ROW FILE. Every caller in the
# image passes nothing and gets `len(batch.reqs)` -- byte-identical behaviour, gated shape by shape
# against this file's own `.orig` rather than argued.
#
# ⛔⛆ #967: THE LOG LINE PRINTED ONLY ITS QUOTIENT, SO THE ENGINE'S OWN TOKEN COUNT COULD NOT BE
# RECOVERED FROM ANY BANKED LOG, EVER. `gen_throughput = self._decode_generated_tokens / gap` and
# NEITHER term reached the log. That is what left round 7's `/code-review` finding unsettlable: the
# ratio's denominator was a MEDIAN OF PER-WINDOW RATES against a TOKEN-WEIGHTED numerator, and the
# exact comparison is total tokens over total time -- which no banked log carries and no re-reading
# will produce. ⇒ the decode line now carries `#gen-token`, `#gen-forward` and `gap (s)`.
#
# ⭐ THE SUFFIX IS CONDITIONAL, ON THE SAME PRINCIPLE AS THE `None` DEFAULT. It is printed only for
# a window in which at least one forward NAMED a count; a window where no caller named one is the
# image's line, byte for byte, and `test_status_801.py`'s differential still gates that unchanged.
# ⇒ the suffix's PRESENCE is itself the claim that the numerator is the drain's measured commit.
# ⛔ `#gen-forward: c/n` is `c` counted forwards out of the window's `n`, and it is NOT decoration:
# a window that mixed measured counts with the image's `len(batch.reqs)` fallback reads `37/40` and
# says so, rather than publishing a silently half-fallback numerator as a measurement.
import os as _ft801_os
import sys as _ft801_sys

print(
    "[#801] overlay ACTIVE: scheduler/status.py bind-mounted from the repo "
    f"(pid {_ft801_os.getpid()}, base md5 643f6a26ddd300d8424594931f068f88)",
    file=_ft801_sys.stderr,
    flush=True,
)


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)
    # ── #967: how many of THIS window's forwards named a count ───────────────────────────────
    # ⛔ A counter, not a flag -- see the `#gen-forward` note in the marker above.
    _decode_counted_forwards: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        # ── #801 round 7: what this forward actually COMMITTED, across the whole batch ───────
        # ⛔ Optional and `None`-defaulted on purpose: this file runs on every served row, and a
        #   caller that names no count must behave exactly as the image does. The scheduler's
        #   drain passes `sum(p.numel() for p in _ft801_published)` -- the same per-request token
        #   lists it ships to the client -- and nothing else in the image passes anything.
        generated_tokens: int | None = None,
    ) -> None:
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )
        elif batch.is_decode:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                generated_tokens=generated_tokens,
            )

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        generated_tokens: int | None = None,
    ) -> None:
        self._decode_forward_count += 1
        # ── #801 round 7: TOKENS, not steps ──────────────────────────────────────────────────
        # ⛔⛆ `is None`, NEVER `or`. A forward whose every request was aborted or had already
        #   finished commits nothing, and `generated_tokens or len(batch.reqs)` would silently
        #   fall back to the request count on exactly that forward -- reporting tokens that were
        #   never generated, which is the bug this edit exists to remove.
        # ⚠ `_decode_forward_count` is left alone deliberately: it gates only the log CADENCE
        #   (every `decode_log_interval`-th decode batch). The throughput denominator below is
        #   wall-clock `gap`, so the denominator was never wrong -- only this numerator was.
        self._decode_generated_tokens += (
            len(batch.reqs) if generated_tokens is None else generated_tokens
        )
        if generated_tokens is not None:
            self._decode_counted_forwards += 1
        if self._decode_forward_count % self.decode_log_interval != 0:
            return

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        # ── #967: the TERMS, read off before the window resets ────────────────────────────────
        # ⛔ Both are captured here and not re-read below: the two assignments that follow zero
        #   the window, and a suffix built from the fields afterwards would print zeros.
        window_tokens = self._decode_generated_tokens
        counted_forwards = self._decode_counted_forwards
        self._decode_generated_tokens = 0
        self._decode_counted_forwards = 0
        # ⚠ `gap` at 4 dp is 0.1 ms on a ~20 s window -- enough that Σtokens / Σgap over a
        #   request's windows is exact rather than a harmonic-mean approximation.
        terms = (
            ""
            if counted_forwards == 0
            else (
                f", #gen-token: {window_tokens}"
                f", #gen-forward: {counted_forwards}/{self.decode_log_interval}"
                f", gap (s): {gap:.4f}"
            )
        )
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"#queue-req: {queue_reqs}"
            # ⛔ APPENDED, never inserted. Every consumer of this line anchors on a PREFIX --
            #   `analyze_816.py`/`analyze_871.py`/`track2.sh` regex `#token: N,` then
            #   `gen throughput (token/s): R`, `measure_ratio_801.py` splits on that label and
            #   takes the first comma, and the shell arms `grep -oE 'gen throughput[^,]*'`.
            #   Appending at the end leaves all of them exact; inserting would not.
            f"{terms}"
        )


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "
