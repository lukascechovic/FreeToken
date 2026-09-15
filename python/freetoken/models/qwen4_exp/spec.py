"""#801 round 6 — the speculative-decoding step, off `models/qwen4_exp/model.py`.

⭐⭐ **WHY THIS FILE EXISTS.** Round 5 grew `Qwen4ExpForCausalLM.mtp_shadow_step` to ~140 lines
owning capture eligibility, the interleave phase, two countdowns, three timing windows, the
acceptance-mass lag, the tracker and the reporting. Round 5's `/code-review` called it fairly as
Feature Envy and parked the extraction FOR round 6, on the reasoning that round 6 rewrites the
step into a verify/commit path anyway and a pure refactor with no load behind it is the exact
shape that produced FOUR "green CPU suite, broken real path" findings in round 5.

⛔ `models/qwen4_exp/model.py` is an EVERY-SERVED-ROW file: it carries a `.orig`, the launcher
asserts that `.orig` against the image before the container starts, and
`test_tap_801.py::test_it_deletes_nothing_and_adds_nothing_at_the_top_level` refuses any new
top-level name in it. Every line of speculative logic living there is a line the flag-off path has
to be argued byte-identical around. ⭐ THIS file is a #801-only overlay -- no `.orig`, in no image,
in no Dockerfile ladder -- so it is the right home, and it is where bullets 2-8 add the verify
path itself.

⚠ **WHAT THE EXTRACTION DELIBERATELY DID NOT DO: re-home the mutable state.** The countdowns,
the three `RollingStats` windows, the tracker, the acceptance counter and the one-shot flags stay
ATTRIBUTES OF THE MODEL, and :func:`shadow_step` reaches for them. That is localised Feature Envy,
not resolved Feature Envy, and it is worth saying plainly rather than implying the call was closed:
60 CPU gates in `test_shadow_wiring_801.py` address that state on the model, and moving it would
be a second, untested change riding inside a refactor. ⇒ the state gets its home here at bullet 8,
where the step is rewritten and a box load can gate the move.

⭐ The in-function `from .draft import ...` / `from .timing import ...` imports are KEPT as
in-function imports. They are what lets the wiring suite monkeypatch `draft_module.<name>` and
have the patch take -- a module-level `from .draft import draft_next_token_ids` would bind the
function object at import time and every spy in that suite would silently miss.

⛔ Bind-mounted, in NO image. See `arm_mtp_801.sh`'s `OVERLAY` manifest -- an overlay this file is
missing from is an overlay whose `-v` silently does not take (#866's shape).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import torch

    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs

    from .model import Qwen4ExpForCausalLM

# ⛔⛆ #801 round 6 bullet 2: `import torch` USED TO BE AT THE TOP OF THIS FILE, and moving it into
#   :func:`shadow_step` is load-bearing, not tidying. The verify arithmetic below is integers, and
#   the host python this repo gates on (3.14, numpy + pytest, NO torch) can only run it if
#   importing this module costs no torch. `test_verify_801.py` is that suite and it is NOT in
#   `conftest.py`'s `_TORCH_ONLY` list; a module-level `import torch` returning here would take the
#   whole file out of collection with a COLLECTION ERROR on the host -- loud, and gated anyway by
#   `TestTheModuleStaysHostImportable`. ⭐ In-function imports are already this module's convention
#   (see `shadow_step`'s `from .draft import ...`), for a different but equally deliberate reason.


def _stats(window) -> dict:
    """A `timing.py::RollingStats` window as a plain dict, for the α line (#801 bullet 7b).

    ⚠ Every field reads ``None`` before the first sample -- `RollingStats`' own rule,
    preserved through serialisation, so "no captured step ran" cannot be read as "the
    captured step took 0 ms". `count` is what tells the two apart at a glance in the bank.

    ⚠ It was a STATIC METHOD on `Qwen4ExpForCausalLM` until round 6 bullet 1, and the reason was
    not taste: `test_tap_801.py::test_it_deletes_nothing_and_adds_nothing_at_the_top_level`
    refuses any new top-level name in `model.py`, because every served row imports that file. That
    pressure is exactly what this module removes -- a top-level name HERE costs the served rows
    nothing, because nothing in the image imports this file at all.
    """
    if window is None:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None}
    return {
        "count": window.count,
        "min": window.min,
        "mean": window.mean,
        "p50": window.p50,
        "p90": window.p90,
    }


def shadow_step(
    model: "Qwen4ExpForCausalLM",
    sampled_ids: torch.Tensor,
    batch: "Batch",
    sampling_args: "BatchSamplingArgs",
    logits: torch.Tensor | None,
) -> dict:
    """The real draft step (llm-server #801 round 5 bullets 2-4): guess ``sampled_ids``'
    successor from THIS forward's tapped `multi_stream`, then check LAST step's guess against
    ``sampled_ids`` -- the one-step lag `shadow.py::ShadowTracker` holds. Called from
    `engine.py`'s `forward_batch`, right after the real sampler produces ``sampled_ids``.

    ⛔ Called through `Qwen4ExpForCausalLM.mtp_shadow_step`, which is a bare delegate to this
    function -- including the early return. The gating decision lives HERE, in one place, rather
    than being split between a guard in the every-row file and a body in this one.

    ⛔ Read-only: the prediction is never fed back into ``sampled_ids`` or anything the
    sampler used, so decode text is byte-identical with this dial on or off. No-op (empty
    map) without a head, with the dial off, or on a PREFILL batch.

    ⛔⛆ **DECODE ONLY, ON PURPOSE.** `multi_stream` is the TAP's rows -- one per REQUEST. A
    decode batch's `out_loc`/`positions` are ALSO one per request (one new token each), so the
    shapes agree. A ragged PREFILL's are one per TOKEN: feeding the tap's rows to a head that
    stores KV at `batch.out_loc` is the exact mismatch that killed round 4 bullet 7's first
    load (`store.cu:88`, "expected 1 but got 53") -- `Qwen4ExpModel.forward`'s own docstring
    names it. MTP has nothing to speculate about during prefill anyway: every prefill token is
    already known, so skipping it costs nothing real.

    ⭐ `multi_stream` can hold MORE rows than ``sampled_ids`` on a padded decode batch (the tap
    takes "every row" of a captured decode forward, `Qwen4ExpModel.forward`'s docstring again;
    `sampled_ids` is already trimmed to `batch.size` by `forward_batch`) -- trimmed to
    ``sampled_ids``'s own row count here, the same convention `batch_logits = logits[:
    batch.size]` already uses, on the same assumption padding rows sit after the real ones
    (`test_tap_801.py::make_batch` builds it that way and round 4 gated it).

    ⚠ The returned ``{uid: accepted}`` map is not yet consumed by anything -- bullet 5/7's
    job, not this one's.

    ⭐ **Round 5 bullet 6, `t_draft` captured.** When `FREETOKEN_MTP801_DRAFT_CAPTURE=1` and a
    graph was captured at bring-up (`mtp_capture_draft_graph`), a bs=1 decode step replays it
    (`capture.py::DraftGraphRunner.replay`) instead of calling `draft.py` eagerly -- same
    `time_draft_call`/`_draft_timing` instrumentation either way, so bullet 7 banks both arms
    from the same measurement path. ⛔ `DraftGraphRunner` is bs=1 only (its own docstring): a
    bs > 1 shadow step, or the dial on before the graph exists, falls back to the eager call
    rather than silently skipping the step -- printed once, not per-step, so the fallback is
    never a silent no-op (#866's shape) but also never floods the log.
    """
    if model._shadow_tracker is None or not batch.is_decode:
        return {}
    import torch

    from .draft import (
        acceptance_mass,
        draft_next_token_ids,
        draft_next_token_ids_and_logits,
        index_sampling_args,
        is_greedy,
    )
    from .timing import time_draft_call

    # ── which ROWS of this forward belong to which request (round 6 bullet 8c) ──────────
    # ⛔⛆ ON A VERIFY STEP THESE TENSORS ARE ONE ROW PER TOKEN AND `batch.reqs` IS ONE PER
    #   REQUEST, AND `zip` DOES NOT SAY SO. The pre-8c line zipped the two and kept row 0 for
    #   every request. Measured in the image before this was written: an ACCEPTED step then
    #   published a draft for the position it had just committed, the next step's verify
    #   rejected it, and α alternated at roughly half its true value -- with byte-identical
    #   text, because the verify rule rejects a stale draft rather than serving it. ⛔ It
    #   costs the round its NUMBER, not its correctness, which is why no earlier gate saw it.
    # ⭐ TWO DIFFERENT ROWS are wanted from the same forward, and :func:`spec_rows` names each
    #   once: a request drafts from its LAST COMMITTED row and is CHECKED at its VERIFIED row.
    staged = verify_step_of(batch)
    if staged is None:
        # ⛔ Today's path: on a plain decode step the rows ARE the requests, and the tap can
        #   hold padding rows past them. ⛔⛆ Read through `mtp_tap_rows` since bullet 9h -- the
        #   pre-9h `model.multi_stream[:rows]` silently returned ONE row whatever `rows` said on
        #   any step whose forward was a REPLAY: latent at bs=1, armed by #949's mr=4.
        rows = sampled_ids.shape[0]
        verify_rows = draft_rows = list(range(rows))
        uids = [r.uid for r in batch.reqs][:rows]
        draft_ids = check_ids = sampled_ids
        R = model.mtp_tap_rows(rows)
        picked = None
    else:
        verify_rows, draft_rows = spec_rows(staged)
        uids = list(staged.plan.uids)
        rows = len(draft_rows)
        picked = torch.tensor(draft_rows, dtype=torch.int64, device=sampled_ids.device)
        draft_ids = sampled_ids[picked]
        check_ids = sampled_ids[
            torch.tensor(verify_rows, dtype=torch.int64, device=sampled_ids.device)
        ]
        # ⛔⛆ #801 round 6 bullet 9h: the tap is `num_tokens` rows WIDE on a replayed verify
        #   step, and `Qwen4ExpForCausalLM.multi_stream` -- the VIEW the prefill sized -- is not.
        #   See :meth:`Qwen4ExpForCausalLM.mtp_tap_rows`; `picked` indexes the step's FLAT rows.
        R = model.mtp_tap_rows(staged.plan.num_tokens)[picked]
    # ⛔⛆ #801 round 6 bullet 9e: the head's OWN batch. Bullet 8c picked the rows and left the
    #   batch alone; `out_loc`, `positions` and `attn_metadata` are all still the backbone's T=2
    #   shapes, and the head forwards one row per request. :func:`draft_view` is a no-op -- the
    #   same object back -- on every step that is not staged.
    draft_batch = draft_view(batch, staged, draft_rows, picked)
    # ⛔⛆ #801 round 5 bullet 7b: GREEDY is part of capture eligibility, and was missing.
    #   `capture.py::DraftGraphRunner` bakes an `argmax` into the graph (its own docstring
    #   says so and bullet 6 narrowed the scope on purpose). Replaying it for a request the
    #   server is SAMPLING would hand the tracker the head's greedy token while the target
    #   drew from a filtered distribution -- α measured against the wrong drafter, with
    #   nothing erroring and the served-sampler arm reading low. The served-sampler arm is
    #   the deployed configuration, so that is the arm the gate would have been decided on.
    greedy = is_greedy(sampling_args)
    # ⭐ Arm the greedy agreement check (round 5 `/code-review`) BEFORE capture eligibility,
    #   because it overrides it. Keeping the head's logits is all this costs on the draft
    #   side; the mass itself is computed NEXT step, on the same one-step lag everything else
    #   here runs on.
    # ⛔⛆ A masschecked step must NOT replay the captured graph: `DraftGraphRunner` returns an
    #   id, the mass needs the head's LOGITS, and a step that took the graph would burn a
    #   countdown and print nothing -- the budget silently emptying with no check performed,
    #   which reads exactly like a check that passed. It runs eager-with-logits instead, and
    #   pays for that by being excluded from every timing window below.
    masscheck_armed = greedy and model._masscheck_left > 0
    if masscheck_armed:
        model._masscheck_left -= 1
    # ⛔⛆ `staged is None`: `capture.py`'s module docstring makes the captured draft graph
    #   depend on the BACKBONE's decode replay having refreshed the shared attention-SELECTION
    #   scratch earlier in the SAME step. On a verify step the backbone replayed the T=2 graph,
    #   so that scratch is the verify step's and the draft graph -- which cannot tell -- would
    #   draft from row 0's window whichever row it is handed. A verify step takes the eager
    #   call. ⚠ Not a property of bullet 8c's row selection: it is bullet 6's capture question,
    #   reopened by bullet 10's concurrency arm.
    eligible_for_capture = (
        model._draft_capture
        and model._draft_graph is not None
        and model._draft_graph.captured
        and staged is None
        and rows == 1
        and greedy
        and not masscheck_armed
    )
    # ⭐ The interleave (bullet 7b): flip on every ELIGIBLE step, so the eager and captured
    #   windows are drawn from the same load, alternating at step granularity. A step that is
    #   not eligible is not a flip -- otherwise a run of sampled requests would leave the
    #   alternation phase-locked to whatever came before them. ⛔ A masschecked step is
    #   ineligible by the clause above, so it does not consume a flip either.
    if eligible_for_capture and model._draft_alternate:
        model._draft_alternate_captured_next = not model._draft_alternate_captured_next
        eligible_for_capture = model._draft_alternate_captured_next
    draft_logits = None
    draftcheck_step = eligible_for_capture and model._draftcheck_left > 0
    if draftcheck_step:
        model._draftcheck_left -= 1
        diff, draft_ms = time_draft_call(
            lambda: model._draft_graph.diff_against_eager(
                model.mtp, model.lm_head, draft_ids, R, draft_batch
            )
        )
        print(f"[#801] draftcheck: {diff}", file=sys.stderr, flush=True)
        predicted_ids = torch.tensor(
            [diff["captured_id"]], dtype=torch.int64, device=draft_ids.device
        )
    elif eligible_for_capture:
        predicted_ids, draft_ms = time_draft_call(
            lambda: model._draft_graph.replay(draft_ids, draft_batch)
        )
    else:
        if model._draft_capture and not model._draft_capture_fallback_fired:
            model._draft_capture_fallback_fired = True
            print(
                "[#801] draft graph unavailable/ineligible this step (bs != 1, not greedy, "
                "or not yet captured) -- falling back to the eager draft call",
                file=sys.stderr,
                flush=True,
            )
        if greedy and not masscheck_armed:
            predicted_ids, draft_ms = time_draft_call(
                lambda: draft_next_token_ids(
                    model.mtp, model.lm_head, draft_ids, R, draft_batch, sampling_args
                )
            )
        else:
            # ⭐ The sampled arm keeps the head's own logits: they are ``q`` for bullet 7b's
            #   acceptance mass. ONE forward either way -- the sibling just does not throw
            #   the logits away (`draft.py::draft_next_token_ids_and_logits`).
            (predicted_ids, draft_logits), draft_ms = time_draft_call(
                lambda: draft_next_token_ids_and_logits(
                    model.mtp, model.lm_head, draft_ids, R, draft_batch, sampling_args
                )
            )
    # ⭐ THREE windows, one measurement path. Which one a step lands in is which code actually
    #   ran, not which dial is set -- a fallback step is an EAGER sample.
    # ⛔⛆ round 5 `/code-review`, two defects in the two-window version:
    #   (1) the served arm's draft call landed in the SAME window as the greedy one, so
    #       "capture saves 27.2 %" priced capture and greedy-versus-sampled together -- and
    #       the probe runs the served tier last, so the rolling window was almost all served.
    #   (2) a DRAFTCHECK step's `draft_ms` is a replay AND an eager call AND the comparison
    #       between them, and it went into the CAPTURED window: with DRAFTCHECK=16 that is up
    #       to 16 of the 256 samples behind the round's `t_draft` = 1.948 ms, every one of
    #       them inflated. An instrumented step is not a measurement of either path.
    instrumented = draftcheck_step or masscheck_armed
    if instrumented:
        window = None
    elif eligible_for_capture:
        window = model._draft_timing_captured
    elif greedy:
        window = model._draft_timing
    else:
        window = model._draft_timing_sampled
    if window is not None:
        window.update(draft_ms)
    if window is not None and not model._draft_timing_fired:
        model._draft_timing_fired = True
        if eligible_for_capture:
            which = "captured"
        elif greedy:
            which = "eager"
        else:
            which = "eager/sampled"
        print(
            f"[#801] t_draft first sample: {draft_ms:.3f} ms ({which})",
            file=sys.stderr,
            flush=True,
        )
    # ── the acceptance mass, on the SAME one-step lag the tracker uses ──────────────────
    # ⛔⛆ The draft held from LAST step is what this step's target distribution scores. Using
    #   this step's own draft logits would compare the head's guess at token t+1 against the
    #   target's distribution at token t -- a number with no meaning that would nonetheless
    #   land in the same field and read as a measurement.
    # ⚠ Computed OUTSIDE `time_draft_call`: it is the instrument, not the draft's own cost.
    # ⭐ ONE computation, TWO destinations, decided by the arm:
    #   served -- the mass IS α, and goes to `accept.py`'s per-request counter;
    #   greedy -- the mass must EQUAL the id comparison, so it goes to the cross-check and
    #   NOT to the counter. Keeping it off the counter is deliberate: the greedy arm's
    #   `mass_n` stays 0, so the analyser's leak check still means what it meant, and the
    #   banked greedy rows keep their shape.
    mass = None
    greedy_mass = None
    if logits is not None and model._pending_draft_logits:
        scored = [i for i, uid in enumerate(uids) if uid in model._pending_draft_logits]
        if scored:
            held = torch.cat([model._pending_draft_logits[uids[i]] for i in scored], dim=0)
            # ⛔ `scored` indexes REQUESTS (so does `index_sampling_args`); the target's `p`
            #   lives at each one's VERIFIED row, which on a plain step is the same number.
            values = acceptance_mass(
                logits[[verify_rows[i] for i in scored]],
                held,
                index_sampling_args(sampling_args, scored),
            )
            values = {uids[i]: float(v) for i, v in zip(scored, values.tolist())}
            if greedy:
                greedy_mass = values
            else:
                mass = values
    model._pending_draft_logits = (
        {uid: draft_logits[i : i + 1] for i, uid in enumerate(uids)}
        if draft_logits is not None
        else {}
    )
    predicted_host = predicted_ids.tolist()
    # ⭐ #801 round 6 bullet 8: the draft the NEXT step verifies. ⛔ Free -- the tracker below
    #   already pays this `.tolist()`, and that is the sync round 5's α was measured under
    #   (see this module's bullet-8 header: the shadow arm was never overlap-clean either).
    #   `publish_drafts` REPLACES the map, so a request absent from this batch has no draft next
    #   step and `stage_verify` decodes it plainly.
    publish_drafts(model, dict(zip(uids, predicted_host)), None)
    accepted = model._shadow_tracker.step(
        uids=uids,
        sampled_ids=check_ids.tolist(),
        predicted_ids=predicted_host,
    )
    # ── the greedy agreement check: Σ min(p,q) against the id comparison ────────────────
    # ⛔⛆ `accepted` is the shadow tracker's own verdict for the SAME lagged step the mass
    #   above scored, so the two are directly comparable and nothing here recomputes either.
    #   Under greedy both filtered distributions are one-hot, so the mass is 1.0 on a match
    #   and 0.0 otherwise -- exactly `accepted`. A disagreement means one of the two is wired
    #   wrong, and the served arm's α, which no check can reach directly, shares that code.
    if greedy_mass:
        for uid, value in greedy_mass.items():
            if uid not in accepted:
                continue
            match = 1.0 if accepted[uid] else 0.0
            print(
                "[#801] masscheck: "
                + json.dumps(
                    {
                        "uid": uid,
                        "mass": value,
                        "token_match": match,
                        "abs_diff": abs(value - match),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    # ⭐ One line per RETIRED request, JSON after a fixed marker so `analyse_alpha_801.py`
    #   parses it rather than scraping prose. See :func:`print_alpha_rows`.
    print_alpha_rows(model, model._accept_counter.step(uids=uids, accepted=accepted, mass=mass))
    if not model._shadow_fired:
        model._shadow_fired = True
        print(
            f"[#801] shadow step fired first time, {len(predicted_ids)} row(s), "
            f"accepted={accepted}",
            file=sys.stderr,
            flush=True,
        )
    return accepted


# ══ #801 round 6 bullet 2: the verify step's arithmetic ═══════════════════════════════════════
#
# ⭐⭐ **THE SHAPE OF A STEP.** At step entry a decoding request has ``cached_len = C`` and
#   ``device_len = C + 1``: positions ``[0, C)`` are in KV and the single token at position ``C``
#   is what the forward runs. `core.py::Req.complete_one` (``cached_len = device_len;
#   device_len += 1``) is what leaves every request in that state, and it is the ONLY shape the
#   engine has ever decoded in.
#
#   A VERIFY step forwards ``[committed, draft]`` instead, by setting ``device_len = C + 2``:
#     - the row at position ``C`` produces the target's own token for position ``C+1``. Against
#       the draft sitting at ``C+1``, that comparison IS the verify;
#     - the row at position ``C+1`` produces the BONUS token for ``C+2``, meaningful only if the
#       draft was accepted.
#   ⇒ a step commits **1 or 2** tokens, and :func:`commit_verify` is the one place that says which.
#
# ⭐ **`scheduler.py` needs no arithmetic of its own for the batch itself.** `_make_positions`,
#   `_make_input_tuple` and `out_loc` are already written in terms of ``Req.extend_len``
#   (= ``device_len - cached_len``), so they generalise to two tokens per request for free. What
#   does NOT generalise, and is why this round overlays four more files: `_make_write_tuple` is one
#   row per REQUEST, `qsa_sparse.py` builds ``token_to_req = arange(bs)``, `gdn.py`'s decode branch
#   calls a single-token conv, and `graph.py` would replay a T=1 graph for a T=2 batch. Each fails
#   SILENTLY. This module hands all four the per-token layout they need, computed once.
#
# ⛔⛆ **The host suite is the point.** Everything below is integers and tuples -- no torch, no
#   `freetoken` import, not even a real `Req` (it reads five integer fields off one). That is what
#   lets `test_verify_801.py` run on the box's host python in milliseconds, where round 5 learned
#   four times over that a green CPU suite can sit on top of a broken real path: the arithmetic is
#   the half that CAN be gated without a GPU, so it is gated there exhaustively and the tensor half
#   is gated in the container.

#: ⛔⛆ Where a request's page HIGH-WATER is remembered. See :func:`pages_for_step` for what it is
#: for. Stashed as an attribute on the `Req` rather than added as a field, because `core.py` is an
#: every-served-row file with a `.orig` the launcher asserts byte-for-byte against the image --
#: adding a dataclass field there would put speculative state into every non-speculative row.
#: ⚠ `Req` is a plain (unslotted) dataclass that already grows `_ids_buf` in `__post_init__`, so an
#: extra attribute is the same kind of thing the class already does to itself.
ALLOC_PAGE_END = "_ft801_alloc_page_end"

#: The draft `stage_verify` wrote into ``token_pool`` this step, as ``(position, token_id)``;
#: ``None`` on a request that is not verifying. Consumed by :func:`commit_verify`.
STAGED_DRAFT = "_ft801_staged_draft"

#: What this step's commit owes ``req.input_ids``: ``{position: token_id}`` for ids it committed
#: that the drain has not appended yet. See :func:`host_token_id`.
UNDRAINED_IDS = "_ft801_undrained_ids"


def host_token_id(req: "Req", position: int) -> "int | None":
    """The token id at absolute ``position``, or ``None`` when the host does not know it yet.

    ⛔⛆ **`req.input_ids` LAGS, and on a plain decode step it lags by exactly the amount that
    leaves `ple_disk._context` with ZERO SLACK.** `scheduler.overlap_loop` runs `_forward` BEFORE
    `_process_last_data`, so nothing a forward reads on the host has seen the tokens of the batch
    still in flight. What holds at host-fill time is

        len(input_ids) == cached_len - (the PREVIOUS step's accepted_len - 1)

    ⇒ at one token per step ``position == len(input_ids)`` exactly, and ``ids[position - 1]`` is
    the LAST element -- on every row this box serves. An ACCEPTED verify step commits two, the
    drain is an iteration behind, and the next fill asks for ``ids[len]``: #801 bullet 9i, load 5's
    `IndexError: index 123 is out of bounds for dimension 0 with size 123`, on both ranks.

    ⭐ Exactly ONE id is ever missing and it is host-known -- the accepted DRAFT, which
    `verify_and_commit` commits on the accepted branch in both the greedy collapse and the sampled
    rule. :func:`commit_verify` records it here rather than syncing the committed tokens back: a
    ``.tolist()`` of them would put a SECOND device sync on every speculative step (#912), on top
    of the one operator decision (a) already bought. The bonus token is device-only, and it lands
    at ``cached_len``, which the PREVIOUS drain has already shipped.

    ⚠ ``input_ids`` WINS. An entry the drain has since caught up on is shadowed rather than
    preferred, so a stale record can never serve an id the request no longer holds.
    """
    ids = req.input_ids
    if 0 <= position < ids.numel():
        return int(ids[position])
    return getattr(req, UNDRAINED_IDS, {}).get(position)


def _div_ceil(a: int, b: int) -> int:
    """`freetoken.utils.misc.div_ceil`, re-spelled rather than imported.

    ⛔ Importing it would drag `freetoken` in and cost this module its host-importability, which
    is the whole reason the arithmetic lives here. It is three tokens of code and it is pinned
    against the engine's own definition, on real values, by
    `test_shadow_wiring_801.py::TestThePlanIsTheEnginesOwnArithmetic` -- in the container, where
    the engine exists. ⚠ A copy nothing compares against is a fork waiting to happen; this one is
    compared against.
    """
    return (a + b - 1) // b


@dataclass(frozen=True)
class VerifyPlan:
    """What ONE forward runs, as plain data: every consumer reads the same tuples.

    ⭐ Frozen on purpose. This crosses from the scheduler (which stages it) to the attention
    metadata, the GDN metadata, the graph runner and the commit, and a plan that any one of them
    could edit is a plan that means something different in the next one.

    ⚠ **Row** below always means a row of the FLAT token dimension of the forward
    (``sum(tokens_per_req)`` of them), never a request. The T=1 special case -- one row per
    request -- is what the engine does today, and a no-draft plan reproduces it exactly.
    """

    #: one per request, in batch order
    uids: tuple = ()
    #: 1 (plain decode) or 2 (verify), per request
    tokens_per_req: tuple[int, ...] = ()
    #: what each request's ``device_len`` must be set to before `_prepare_batch` runs
    device_lens: tuple[int, ...] = ()
    #: flat: the sequence position each row's input token sits at
    positions: tuple[int, ...] = ()
    #: flat: row -> index into :attr:`uids`. Today's engine hard-codes ``arange(bs)``.
    token_to_req: tuple[int, ...] = ()
    #: ``len(uids) + 1`` prefix sums of :attr:`tokens_per_req`. Today's is ``arange(bs + 1)``.
    cu_seqlens: tuple[int, ...] = (0,)
    #: flat: where row ``r``'s SAMPLED token is written in `scheduler.py`'s `token_pool`
    write_positions: tuple[int, ...] = ()
    #: ``(request index, position, token id)`` per verifying request: the draft token has to be
    #: in `token_pool` at that position BEFORE the forward, because `_make_input_tuple` reads the
    #: second row's input token from exactly there.
    draft_writes: tuple[tuple[int, int, int], ...] = ()

    @property
    def num_tokens(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def is_speculative(self) -> bool:
        """True if ANY request is verifying. A mixed batch is normal: a request whose draft was
        not produced this step decodes plainly alongside one that is verifying."""
        return any(t > 1 for t in self.tokens_per_req)

    @property
    def verify_rows(self) -> tuple[int, ...]:
        """Per request: the row whose logits carry the target's own token for position ``C+1``.
        It is the request's FIRST row either way -- for a plain request that is simply its next
        token, which is why this is not ``None`` there."""
        return self.cu_seqlens[:-1]

    @property
    def bonus_rows(self) -> tuple[int | None, ...]:
        """Per request: the row whose logits carry the bonus token, or ``None`` when the request
        forwarded one token. ⛔ ``None``, never "the same row as the verify" -- a T=1 request's
        single row IS its ordinary next token, and reading it as a bonus as well would append
        that token twice.
        """
        return tuple(
            (self.cu_seqlens[i + 1] - 1 if t > 1 else None)
            for i, t in enumerate(self.tokens_per_req)
        )


def plan_verify(reqs: Sequence["Req"], draft_ids: Mapping[Any, int | None]) -> VerifyPlan:
    """Lay out one decode forward over ``reqs``, verifying ``draft_ids[uid]`` where there is one.

    ``draft_ids`` is keyed by ``Req.uid``, the same convention `shadow.py::ShadowTracker` and
    `accept.py` already use -- a missing uid, or one mapped to ``None``, simply decodes plainly.
    ⭐ Keyed rather than positional because the draft map is produced a step earlier, against a
    batch whose membership may have changed; positional alignment would silently hand request A's
    draft to request B.

    ⛔ **READ-ONLY on every request.** The plan says what ``device_len`` must become; assigning it
    is the caller's, at the one place the batch is actually built. A plan discarded for any reason
    (an abort landing between plan and prepare) must not leave a request advanced with no forward
    behind it.

    ⛔⛆ **A draft is DROPPED when accepting it would overrun the output budget.** `Req` may never
    reach ``device_len > max_device_len``: that ceiling is the client's ``max_tokens``, it is what
    `can_decode` reports as "length", and `__post_init__` asserts it. A verify step that accepts
    emits TWO tokens, so it needs two of budget; with one left the request decodes plainly. ⚠ The
    overrun would not raise anywhere near the forward -- the request would just return one token
    past what the client asked for, which is the kind of thing that is noticed in a bug report.
    """
    uids: list = []
    tokens_per_req: list[int] = []
    device_lens: list[int] = []
    positions: list[int] = []
    token_to_req: list[int] = []
    cu_seqlens: list[int] = [0]
    write_positions: list[int] = []
    draft_writes: list[tuple[int, int, int]] = []
    for i, req in enumerate(reqs):
        # ⛔ `device_len == cached_len + 1` IS step entry. Planning a request twice without a
        #   commit between would stage a three-token forward nothing downstream is built for, and
        #   the extra row would read as a perfectly valid bonus token.
        assert req.device_len == req.cached_len + 1, (
            f"req {req.uid} is not at step entry: cached_len={req.cached_len}, "
            f"device_len={req.device_len} (expected {req.cached_len + 1})"
        )
        draft = draft_ids.get(req.uid)
        # `Req.remain_len` at entry, spelled out rather than read off the property so this
        # function depends on integer fields only (see the module note on host-importability).
        room = req.max_device_len - req.device_len
        tokens = 2 if (draft is not None and room >= 2) else 1
        uids.append(req.uid)
        tokens_per_req.append(tokens)
        device_lens.append(req.cached_len + tokens)
        for t in range(tokens):
            position = req.cached_len + t
            positions.append(position)
            token_to_req.append(i)
            # ⭐ ONE PAST ITS OWN INPUT POSITION. `scheduler.py::_make_write_tuple` writes today's
            #   single sampled token at ``req.device_len``, which for a one-token step is exactly
            #   this. ⛔ It also means the verify row's write lands ON TOP of the draft: accepted,
            #   it rewrites the same id; rejected, it replaces it -- so the next step reads the
            #   corrected token straight out of `token_pool` and there is nothing to undo.
            write_positions.append(position + 1)
        if tokens > 1:
            draft_writes.append((i, req.cached_len + 1, int(draft)))
        cu_seqlens.append(cu_seqlens[-1] + tokens)
    return VerifyPlan(
        uids=tuple(uids),
        tokens_per_req=tuple(tokens_per_req),
        device_lens=tuple(device_lens),
        positions=tuple(positions),
        token_to_req=tuple(token_to_req),
        cu_seqlens=tuple(cu_seqlens),
        write_positions=tuple(write_positions),
        draft_writes=tuple(draft_writes),
    )


def print_alpha_rows(model, rows: "list[dict]") -> int:
    """Print one `[#801] alpha` line per row and return how many. Returns 0 on an empty list.

    ⭐ `t_draft` is composed in HERE, at print time — `timing.py`'s own docstring reserved that
    for this bullet, and it is why `RollingStats` was never folded into the tracker.
    """
    for row in rows:
        row["t_draft_eager"] = _stats(model._draft_timing)
        row["t_draft_captured"] = _stats(model._draft_timing_captured)
        row["t_draft_eager_sampled"] = _stats(model._draft_timing_sampled)
        print(f"[#801] alpha {json.dumps(row)}", file=sys.stderr, flush=True)
    return len(rows)


def drain_alpha(model) -> int:
    """⭐⭐⭐ #801 r6 b9am: flush the LAST request's α row. Returns how many rows printed.

    ⛔⛆ **WHY NO α ROW HAS EVER PRINTED IN THIS ROUND, across eighteen loads.**
    `AcceptanceCounter.step` computes retirement as *"rows whose uid is not in THIS step's live
    batch"*, so a request's row is emitted **one shadow step AFTER it leaves the batch**. At bs = 1
    the flush therefore has to come from the NEXT request — and on a probe whose last sample is the
    last request there is no next step, ever. `accept.py::drain` was written for exactly this and
    its docstring says so: *"Nothing calls this today … the probe's throwaway-final-request trick
    is visibly a WORKAROUND for a missing flush rather than the design."* This is the call.

    ⭐ **Called from `scheduler.run_when_idle`, BEFORE `check_integrity`** — deliberately, and it is
    not cosmetic: `check_integrity` is what killed loads 17 and 18, and a drain placed after it
    would lose the α row of precisely the run that most needs explaining.

    ⛔ A NO-OP with the flag off: `model.py` leaves `_accept_counter` at ``None`` unless the shadow
    tracker is attached, and every attribute here is read defensively, because this runs on an
    every-served-row idle path where an `AttributeError` would take down a healthy server.
    """
    counter = getattr(model, "_accept_counter", None)
    if counter is None:
        return 0
    return print_alpha_rows(model, counter.drain())


def commit_verify(req: "Req", accepted_len: int, *, page_size: int) -> None:
    """Advance ``req`` by the ``accepted_len`` tokens this step actually committed.

    ⭐ **This REPLACES `core.py::Req.complete_one` for a speculative row**, it does not wrap it:
    ``complete_one`` advances by exactly one and would have to be undone. With no draft
    (``accepted_len == 1`` on a one-token step) the two are identical, which is the flag-off
    claim and is gated as such.

    ⭐ **The rollback IS the short commit.** Position ``C+1`` holds the rejected draft's KV;
    leaving ``cached_len`` at ``C+1`` means the next forward re-runs that position and overwrites
    it. Nothing is erased, and `token_pool` was already corrected by this step's own write.

    ⛔ ``accepted_len`` is at least 1: rejecting the DRAFT never rejects the token the TARGET
    itself just produced. Zero would stall the request forever with nothing erroring.

    ⚠ The caller still owes `append_host` the same ``accepted_len`` token ids -- `Req.append_host`
    already takes a multi-token tensor, so that needs no `core.py` change.
    """
    forwarded = req.device_len - req.cached_len
    assert 1 <= accepted_len <= forwarded, (
        f"req {req.uid} forwarded {forwarded} token(s) this step; cannot commit {accepted_len}"
    )
    # ⛔⛆ Record the page high-water BEFORE the lengths move -- `req.device_len` is still this
    #   step's, and `allocate_paged` charged pages up to exactly its ceiling. See
    #   :func:`pages_for_step` for the leak this prevents.
    end = _div_ceil(req.device_len, page_size)
    setattr(req, ALLOC_PAGE_END, max(getattr(req, ALLOC_PAGE_END, 0), end))
    req.cached_len += accepted_len
    req.device_len = req.cached_len + 1
    assert req.device_len <= req.max_device_len, (
        f"req {req.uid} committed {accepted_len} token(s) past its output budget: "
        f"device_len={req.device_len} > max_device_len={req.max_device_len}"
    )
    # ⛔⛆ #801 bullet 9j: what this commit owes `req.input_ids`, which the drain only appends an
    #   ITERATION LATER (`overlap_loop` forwards before it drains). ONE forward's worth, REPLACED
    #   every commit rather than accumulated: the only fill that needs an entry is the very next
    #   one, and by the one after it the drain has caught up. ⚠ Set on every commit, including a
    #   rejecting one -- recording the staged draft unconditionally would leave a token the model
    #   never committed sitting at a position the REJECTED step overwrote, and nothing would raise.
    staged = getattr(req, STAGED_DRAFT, None)
    setattr(req, UNDRAINED_IDS,
            {staged[0]: staged[1]} if (accepted_len > 1 and staged is not None) else {})


@dataclass(frozen=True)
class PublishPlan:
    """Which tokens of ONE commit the drain may ship before the output budget ends the reply.

    ``length_flags[k]`` is the *length* verdict for the commit's ``k``-th published token — the
    `hit_length` the image spells as ``not req.can_decode``. EOS and stop strings are NOT here:
    they depend on the token id and on the detokenizer's text, they are already per-token in the
    drain, and they were never wrong. ⭐ This carries the one verdict that was.
    """

    length_flags: tuple[bool, ...]


def publish_plan(*, own_cached_len: int, committed: int, max_device_len: int) -> PublishPlan:
    """The per-token *length* verdict for a commit of ``committed`` tokens.

    ⭐⭐⭐ **#801 r6 b9bh: THE DRAIN WAS JUDGING A PAIR'S FIRST TOKEN BY THE WHOLE PAIR.**
    `scheduler.py::_process_last_data` read `hit_length = not req.can_decode` once per published
    token, off the request's LIVE ``device_len`` — and :func:`commit_verify` advanced that by the
    WHOLE accepted pair before the drain ran. So on the one trajectory where a pair lands exactly
    on the cap, the FIRST token is judged as if the second had already shipped: it trips *length*,
    the publish loop `break`s, and 9az's tail loop appends the second token to ``req.input_ids``
    while shipping no `DetokenizeMsg` for it. The client's reply is one token short, and load 24
    measured exactly that on seven of nine cells (510 against the control's 511).

    ⭐⭐ **THE RULE: token ``k`` is judged against the ``device_len`` a ONE-TOKEN-PER-STEP row
    would have had at the same generated index**, which is ``own_cached_len + 2 - (committed - k)``
    (b9bn; 9bh wrote it ``cached_len + 1 - (committed - k)`` off the LIVE value, which is the same
    integer on a one-wide next forward and one too far on a two-wide one). Undoing the part of the
    advance that belongs to tokens not yet published is the whole change.

    ⛔ **A NON-SPECULATING ROW IS BYTE-IDENTICAL, and that is arithmetic here, not an argument.**
    With ``committed == 1`` the expression is ``own_cached_len + 2``, which IS the LIVE
    ``device_len`` the image reads — the next forward committed one, so ``live == own + 1`` and
    ``live + 1 == own + 2``. The flag-off path therefore reads the same verdict it read before
    this function existed. See the b9bn block below for the second case (no next forward at all).

    ⛔⛆ **``device_len`` IS DELIBERATELY NOT A PARAMETER.** By drain time it carries two advances
    that do not belong to the token being judged: the rest of this commit, and the NEXT step's
    staged width (`overlap_loop` schedules and launches batch N+1, whose `_prepare_batch` stages
    ``device_len`` to ``cached_len + T``, *before* draining batch N). The staged width cannot
    currently flip a verdict — `plan_verify` stages two wide only when ``room >= 2``, which puts
    the staged ``device_len`` at ``cached_len + 2 <= max_device_len - 1``, strictly below the cap —
    but it can only stay unable to if the verdict cannot read it at all. `test_publish_801.py`
    gates the absence of the parameter, not just the values.

    ⚠ **9bh's own "worth exactly one token" note is SUPERSEDED and kept here labelled rather than
    deleted** ([[feedback_stale_number_is_state_bound]]). It read: *"a true flag terminates the
    request, so the drain can lose at most one; load 24's cell 2 (509) and warmup (61) are short by
    TWO, [so] the second token is a different mechanism."* The bound was derived from a fixture
    whose drain saw its OWN forward's lengths — see ``own_cached_len`` below. Under the deployed
    order the image's rule loses up to **two**, and `test_publish_801.py` now reproduces
    511 / 510 / **509** and 63 / 62 / **61** from this one seam. ⇒ there was never a second
    mechanism, and the detokenizer was never the suspect.

    ⭐⭐⭐ **#801 r6 b9bn: ``own_cached_len`` IS THE DRAINED FORWARD'S OWN POST-COMMIT VALUE, AND
    THE RENAME IS THE FIX.** 9bh got the arithmetic right and was handed the wrong number: the
    drain read `req.cached_len` LIVE, and by then `overlap_loop` has already run the NEXT forward's
    commit — the next forward's post-commit value on **36 of 36** banked hardware rows (load 25).
    The excess is ``W_next - 1``: **0** on a non-speculating row, which is always one wide, and
    **1** whenever the next forward accepted a pair. So the verify arm's verdict was read one
    position ahead and the row finished one generated token early.

    ⛔⛆ **THE VALUE CANNOT BE RECOVERED FROM LIVE STATE, WHICH IS WHY IT IS A PARAMETER AND WHY
    IT IS CARRIED.** After any commit ``device_len == cached_len + 1`` always, so the gap between
    the two carries nothing about ``W_next``. `engine/engine.py::forward_batch` records it on the
    BATCH (``batch.ft801_post_commit``) after BOTH advance paths — this function's `commit_verify`
    for a staged row, `Req.complete_one` for every prefill, flag-off row and no-draft step — and
    the batch is a fresh object per forward (`scheduler/decode.py` builds one every call), so the
    drain reads the forward it is actually draining. ⛔ A carrier held on the REQUEST could not:
    at drain time the request's own copy is already the next forward's, and a two-deep rotation
    reads two advances stale exactly when the request skipped the next forward.

    ⛔⛆ **WHY ``+ 2`` AND NOT ``+ 1``, AND THE CONTROL IS BYTE-IDENTICAL BY ARITHMETIC, NOT BY
    ASSERTION.** On a non-speculating row the next forward commits exactly one, so
    ``live == own + 1`` and 9bh's ``live + 1`` IS ``own + 2`` — the same integer, so the flag-off
    path reads the verdict it read before this parameter existed. Where no next forward ran at all
    the row was dropped by `can_decode`, which means ``own + 1 >= max_device_len`` already, so both
    expressions are `True` and agree again. ⭐ Those are the only two cases, and both are gated.

    ⛔ **The ``- 1`` in the image's own count is NOT encoded here and must never be** (9bi): the
    control publishes ``max_tokens - 1`` because of this same one-forward lateness, on both arms.
    The target is whatever the non-speculating row emits, and every gate is relative to it.
    """
    assert committed >= 1, f"a commit publishes at least one token, got {committed}"
    return PublishPlan(
        length_flags=tuple(
            (own_cached_len + 2 - (committed - k)) >= max_device_len
            for k in range(1, committed + 1)
        )
    )


def owned_page_end(req: "Req", *, page_size: int) -> int:
    """One past the last page ``req`` actually OWNS — the single number both sides of the page
    ledger must read.

    ⭐⭐⭐ **#801 r6 b9ak: the two sides had drifted, and load 17 died of it.**
    `scheduler/cache.py::_padded_tail` freed a finishing request through
    ``div_ceil(cached_len) * page_size``, and its docstring states the premise: *"allocate_paged
    allocates whole pages, so the padding [cached_len, page_ceil) belongs to the finishing
    request."* True only while the allocation ceiling **is** ``div_ceil(cached_len)``.

    ⛔⛆ **A REJECTED FINAL DRAFT BREAKS IT.** The step forwarded to ``device_len`` and pulled the
    page that position needs; the rejection commits only ``cached_len = device_len - 1``. The page
    is allocated, is in the page-table row, and sits ABOVE where the free stopped — so it is
    returned to nobody. The engine's own `CacheManager.check_integrity` catches it at the next idle
    moment and kills the row: ``free_pages(4094) + cache_pages(1) != num_pages(4096)``.

    ⭐⭐ **It is not bullet 8's high-water.** Upstream's naive ``div_ceil`` arithmetic strands the
    same page against the same old bound; the guard only changed WHICH pages a step charges, never
    where the free stops. Both counterfactuals are gated in
    `test_verify_801.TestEveryPageAllocatedIsAlsoFreed`.

    ⭐ **A row that never speculates cannot strand at all** — it commits the token it forwarded, so
    ``cached_len == device_len`` at finish and the two ceilings coincide. Which is why the
    MTP-off control never showed this, and why the attribute being absent here makes the ``max``
    degrade to today's exact bound.

    ⚠ ~**one page per ``page_size`` tokens of final length** (final lengths 65, 129, 193, … at the
    deployed 64), so it is rare enough to look like a lottery and certain enough to kill any long
    run. Load 17 lost exactly one page of 4096.
    """
    return max(_div_ceil(req.cached_len, page_size), getattr(req, ALLOC_PAGE_END, 0))


def pages_for_step(req: "Req", *, page_size: int) -> tuple[int, int]:
    """The page range ``[first, last)`` this step must actually allocate for ``req``.

    ⛔⛆ **THE LEAK THIS EXISTS TO PREVENT, and it is silent end to end.**
    `scheduler/cache.py::allocate_paged` charges ``[div_ceil(cached_len), div_ceil(device_len))``
    and `_write_page_table` writes the request's page-table row for them. With the deployed
    64-token pages (`attention/__init__.py` registers `qsa_sparse` with ``page_sizes=(64,)`` and
    `engine.py` OVERRIDES `config.page_size` to it, so this is not a tunable), a verify step at
    ``C = 63`` reaches ``device_len = 65`` and pulls page index 1. Reject, and ``cached_len`` rolls
    back to 64 -- but page 1 is already allocated and already in the row. The next step recomputes
    ``div_ceil(64) = 1``, allocates page 1 a SECOND time and writes the new slots over the same
    row. The first page is now referenced by nothing, and every free path walks the page table, so
    it is never returned. No exception, no log line; the pool just drains.

    ⇒ the fix is a monotone per-request high-water (:data:`ALLOC_PAGE_END`, set by
    :func:`commit_verify`), and the range starts at whichever of the two is further along.

    ⭐ **Identical to today's arithmetic for a request that never speculated**: the attribute is
    absent, the ``max`` degrades to ``div_ceil(cached_len)``, and the flag-off row allocates
    exactly the pages it allocates now. That is gated over a spread of lengths rather than argued.

    ⚠ `page_size == 1` (the engine default, and every non-QSA row) cannot produce the hazard at
    all -- every token is its own page -- and the guard is a no-op there by construction.
    """
    first = owned_page_end(req, page_size=page_size)
    last = _div_ceil(req.device_len, page_size)
    # ⛔ NOT clamped. The high-water can equal `last` (it does on the step right after a rollback)
    #   but can never exceed it: the step that set it committed at least ``forwarded - 1`` tokens,
    #   so the next ``device_len`` is at least that step's. An inversion means the bookkeeping
    #   drifted, and silently returning an empty range would starve the request of KV instead.
    assert first <= last, f"req {req.uid}: page high-water {first} is past this step's {last}"
    return first, last


# ══ #801 round 6 bullet 3: the verify/commit RULE ═════════════════════════════════════════════
#
# ⭐⭐ Bullet 2 said how far a step advances. This says WHICH TOKENS it advances with, and it is
#   the half that has to be LOSSLESS: the whole justification for speculative decoding is that the
#   text is distributed exactly as the target model alone would have produced it. A scheme that is
#   merely "close" buys throughput by quietly changing the model, which is the one cost this round
#   is not allowed to pay -- bullet 9's gate is byte-identical greedy output against the MTP-off
#   control, and that gate only means something because the rule below is exact.
#
# ⭐ **The rule**, per verifying request, with ``p`` the target's filtered distribution at the
#   verify row and ``q`` the draft head's filtered distribution for the same position:
#     - the draft ``d`` was DRAWN FROM ``q`` a step earlier (`draft.py::sample_ids`);
#     - accept it with probability ``min(1, p(d)/q(d))``;
#     - on rejection commit a draw from the residual ``norm(max(p - q, 0))``.
#   The committed token is then distributed exactly as ``p`` -- that is the standard speculative
#   sampling identity, and the acceptance RATE it implies is ``Σ_x min(p(x), q(x))``, which is
#   precisely what round 5's `draft.py::acceptance_mass` already measures in shadow mode. ⇒ the
#   two halves of the round are the same number, and a gate checks the live path's empirical
#   acceptance against that instrument rather than against a second derivation of it.
#
# ⛔⛆ **BOTH distributions are filtered by the SAME sampling args**, and the issue body names this
#   as the condition for losslessness. ``q`` filtered by the DRAFT's own top-k/top-p and ``p`` by
#   the target's would be a scheme nobody proposed; worse, an UNFILTERED ``q`` makes the residual
#   negative-clipped against a denominator that never sums to one, and the committed distribution
#   drifts off ``p`` by an amount that grows with how hard the filter bites. Nothing errors. The
#   `q`-is-filtered property is gated empirically, over draws, not asserted.
#
# ⛔ **Greedy takes the cheap path, and the cheap path is gated against the general one.** Under
#   greedy both distributions are one-hot, so the rule collapses to "did the head's id equal the
#   target's argmax", and running it through `filtered_probs` would allocate two vocab-wide rows
#   per step for a comparison of two integers. `verify_and_commit` therefore branches -- and a
#   gate runs the same inputs through both paths and requires the same answer.


@dataclass(frozen=True)
class VerifyOutcome:
    """What one verify forward decided. Tensors throughout, on the logits' own device.

    ⛔⛆ **Nothing here is a host value, on purpose.** Reading `accepted_len` on the host is a
    device sync, and where that sync goes is bullet 8's open design question (see
    :func:`commit_verify`'s note): `Engine.forward_batch` advances lengths BEFORE the sampler
    today because the advance needs no token values, and a verify step's does. Returning tensors
    keeps that decision out of this function instead of baking a sync into it.
    """

    #: ``[num_tokens]`` -- the token to write at each row's `VerifyPlan.write_positions`. ⭐ EVERY
    #: row gets one, including a rejected request's bonus row: that write lands at a position
    #: ``cached_len`` will not reach, and the next step overwrites it. Writing it unconditionally
    #: is what keeps the scatter shape fixed, which a captured graph needs.
    tokens: "torch.Tensor" = None
    #: ``[n_verify]`` bool, one per VERIFYING request, in batch order
    accepted: "torch.Tensor" = None
    #: ``[n_req]`` -- 1 or 2, exactly the ``accepted_len`` :func:`commit_verify` takes
    accepted_len: "torch.Tensor" = None


def verify_and_commit(
    plan: VerifyPlan,
    target_logits: "torch.Tensor",
    draft_logits: "torch.Tensor | None",
    draft_ids: "torch.Tensor",
    args: "BatchSamplingArgs",
    *,
    generator: "torch.Generator | None" = None,
) -> VerifyOutcome:
    """Decide every row's committed token, and how many of them each request keeps.

    ``target_logits [num_tokens, vocab]`` is this forward's own output, flat over ``plan``'s rows.
    ``draft_logits [n_verify, vocab]`` and ``draft_ids [n_verify]`` are the head's, held from the
    PREVIOUS step for the verifying requests only, in batch order -- ``draft_logits`` may be
    ``None`` under greedy, where the rule needs no ``q``. ``args`` is the batch's own
    `BatchSamplingArgs`, indexed per REQUEST, which is why every row reaches it through
    ``plan.token_to_req``.

    ⭐ **A non-verifying row is an ordinary sampler draw**, through `draft.py::sample_ids` -- the
    same function the draft step uses, so a plain decode row inside a speculative batch is not a
    second spelling of the served sampler. That covers both a request with no draft this step and
    every bonus row.

    ⚠ ``generator`` is threaded through for the distribution gates, which need many draws to be
    reproducible. Serving passes ``None`` and gets torch's global RNG, like the real sampler.
    """
    import torch

    from .draft import filtered_probs, index_sampling_args, is_greedy, sample_ids

    num_tokens = plan.num_tokens
    assert target_logits.shape[0] == num_tokens, (
        f"target_logits has {target_logits.shape[0]} rows, the plan lays out {num_tokens}"
    )
    device = target_logits.device
    verify_reqs = [i for i, t in enumerate(plan.tokens_per_req) if t > 1]
    verify_rows = [plan.cu_seqlens[i] for i in verify_reqs]
    # ⚠ The set is hoisted, not inlined into the comprehension's `if`. Inlined it is rebuilt once
    #   per row -- O(rows x requests), which at a serving batch size is invisible and at the
    #   200k-draw distribution gate is a hang. Found by that gate, which is the only caller big
    #   enough to see it.
    is_verify_row = set(verify_rows)
    plain_rows = [r for r in range(num_tokens) if r not in is_verify_row]

    tokens = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    if plain_rows:
        tokens[plain_rows] = sample_ids(
            target_logits[plain_rows],
            index_sampling_args(args, [plan.token_to_req[r] for r in plain_rows]),
        ).to(torch.int64)

    accepted = torch.zeros(len(verify_reqs), dtype=torch.bool, device=device)
    if verify_reqs:
        row_args = index_sampling_args(args, [plan.token_to_req[r] for r in verify_rows])
        verify_logits = target_logits[verify_rows]
        ids = draft_ids.to(torch.int64).reshape(-1)
        assert ids.shape[0] == len(verify_reqs), (
            f"{ids.shape[0]} draft id(s) for {len(verify_reqs)} verifying request(s)"
        )
        if is_greedy(row_args):
            # ⭐ The collapse: ``p`` is one-hot on the argmax and ``q`` on the head's own id, so
            #   acceptance is id equality and the committed token is the target's argmax either
            #   way -- ACCEPTED OR NOT. No distribution is materialised.
            committed = verify_logits.argmax(dim=-1)
            accepted = committed == ids
        else:
            p = filtered_probs(verify_logits, row_args)
            assert draft_logits is not None, "a sampled verify step needs the head's own logits"
            q = filtered_probs(draft_logits, row_args)
            index = ids.unsqueeze(-1)
            p_d = p.gather(-1, index).squeeze(-1)
            q_d = q.gather(-1, index).squeeze(-1)
            u = torch.rand(p.shape[0], device=device, generator=generator)
            # ⛔⛆ ``u * q_d < p_d``, not ``u < p_d / q_d``: STRICT, and multiplied rather than
            #   divided. Strict because `torch.rand` can return exactly 0.0, and ``0 <= p_d``
            #   would then accept a token the target gives ZERO mass to -- rare enough to never
            #   show in a gate and a real losslessness violation. Multiplied because ``q_d`` is a
            #   filtered probability and can be denormal; the division is the only place this
            #   rule can produce an inf.
            # ⚠ ``q_d > 0`` cannot fail for a draft actually drawn from ``q``. It fails if the
            #   draft was drawn under DIFFERENT sampling args than the verify -- in which case no
            #   rule restores losslessness, and rejecting at least keeps the committed token on a
            #   ``p``-derived distribution rather than committing an out-of-support id.
            accepted = (q_d > 0) & (u * q_d < p_d)
            residual = (p - q).clamp_min(0.0)
            mass = residual.sum(dim=-1, keepdim=True)
            # ⚠ ``mass == 0`` means ``p`` and ``q`` agree everywhere, where acceptance is
            #   certain and this draw is never selected. Falling back to ``p`` keeps the
            #   multinomial defined rather than handing it an all-zero row.
            residual = torch.where(mass > 0, residual / mass.clamp_min(1e-30), p)
            resampled = torch.multinomial(residual, 1, generator=generator).squeeze(-1)
            committed = torch.where(accepted, ids, resampled)
        tokens[verify_rows] = committed.to(torch.int64)

    accepted_len = torch.ones(len(plan.tokens_per_req), dtype=torch.int64, device=device)
    if verify_reqs:
        accepted_len[verify_reqs] += accepted.to(torch.int64)
    return VerifyOutcome(tokens=tokens, accepted=accepted, accepted_len=accepted_len)


# ⛔⛆ #801 round 6 bullet 5: THE LINEAR STATE IS THE HALF OF A VERIFY STEP THAT CANNOT BE
#   RE-DERIVED. A rejected KV page is just a slot the next step overwrites (bullet 2). A GDN
#   layer's recurrent state and conv window are RECURRENCES: once the second token has folded
#   into them there is no way back except a snapshot, and a row that keeps a rejected token's
#   state serves subtly wrong text forever after, with nothing erroring and no page to blame.


class MissingVerifySnapshots(RuntimeError):
    """A staged verify step reached a state-advancing call carrying no record of its own forward.

    ⛔⛆ **#801 round 6 bullet 9r.** Raised, never logged and never defaulted away, because the
    three loads this round spent chasing it all read as *success*: the arm served, every cell
    completed, no traceback, and the text was quietly wrong.
    """


def require_linear_snapshots(batch: "Batch", plan: VerifyPlan, *, where: str) -> dict:
    """The GDN snapshots this step's forward recorded -- or a raise. ⛔ Never a default empty map.

    ⛔⛆ **THE DEFAULT IS THE BUG, AND IT HID A CORRECT FIX FOR A WHOLE LOAD.** Both call sites
    used to read the carrier with a trailing ``or {}``, which turns *"this step's forward was
    never recorded"* into *"this step had nothing to commit"*. Those are opposite facts. A staged
    verify step is speculative by construction -- `stage_verify` returns ``None`` when
    ``not plan.is_speculative`` -- so its forward ran `gdn.py::_verify_fla`, which assigns
    ``batch.linear_snapshots`` in PYTHON. Arriving here empty means the python never ran.

    ⭐⭐ **AND THAT IS EXACTLY WHAT A CAPTURED STEP DOES (bullet 9q).** `engine.py`'s forward is
    ``self.graph_runner.replay(batch) if use_graph else self.model.forward()``, and
    `graph.py::GraphRunner.replay` refills static buffers and calls ``g.replay()`` -- it never
    calls the model's python forward. `can_use_cuda_graph` admits verify steps, and capture
    happens at bring-up, so the FIRST speculating step is already a replay. Load 8 and load 9 came
    back 256/256 report lines byte-identical with 9p's rewind mounted and verified in the
    container; a rewind that wrote different bytes into a tensor the reference forward reads
    cannot do that. It was reading ``{}``.

    ⛔ One raise, three symptoms: the check-row instrument's rewind becomes a no-op,
    :func:`commit_linear_state` never runs so GDN's recurrent state never advances (bullet 5's
    docstring predicted the shape in the words *"not a crash, it is wrong text"*), and
    :func:`commit_ple_state` never runs either.

    ⚠ ``where`` names the call site rather than the function, because the two sites fail for the
    same reason and are fixed together but are reached under different conditions: the commit runs
    on every speculating step, the rewind only when the check-row instrument is armed.

    ⛔ The caller guards on the POOL, not on a flag: with no `LinearStatePool` there is no commit
    to reach and an absent map says nothing about whether the forward ran.
    """
    snapshots = getattr(batch, "linear_snapshots", None)
    if snapshots:
        return snapshots
    raise MissingVerifySnapshots(
        f"#801: a staged verify step reached {where} with no `batch.linear_snapshots` "
        f"({len(plan.uids)} requests, {plan.num_tokens} rows, tokens_per_req "
        f"{plan.tokens_per_req}). That is NOT 'nothing to commit' -- a staged plan is "
        "speculative, so this step's forward ran gdn.py::_verify_fla and MUST have recorded "
        "itself. An empty map means the python forward never ran, which is what "
        "GraphRunner.replay does on a CAPTURED verify step: it refills the static buffers and "
        "calls g.replay(), and the snapshot assignment is python. Committing nothing here would "
        "leave GDN's recurrent state at its step-entry value forever and the row would serve "
        "wrong text with nothing erroring (#801 round 6, loads 6-9)."
    )


#: The batch attributes a verify forward writes in PYTHON and a captured replay therefore loses.
#: ⛔⛆ **AN ENUMERATION, NEVER A SWEEP (#801 round 6 bullet 9s).** The same forwards also write
#: ``attn_metadata``, ``fla_metadata`` and ``linear_verify_width``, and carrying THOSE across a
#: replay would hand every live step the capture batch's DUMMY addressing. The three named here
#: are the ones whose fields are all static graph-pool tensors that each replay refills -- the
#: GDN intermediate-states buffer (reserved before capture), ``conv_window``'s clone (allocated
#: inside the captured region, so the clone op itself replays), ``conv_inputs``, the persistent
#: ``fla_cu_seqlens``, and ``table_idx``, which `GraphCaptureBuffer.copy_from` refills with the
#: LIVE step's slots on every replay. ⇒ the data is already correct on a replayed step; only the
#: python binding is missing, and that is all this pair restores.
CAPTURED_MEMOS = ("linear_snapshots", "ple_snapshots", "ple_context_snapshot")


def capture_memos(batch: "Batch") -> dict:
    """What a CAPTURED verify forward recorded, to be re-bound on every replay of that graph.

    ⛔⛆ **#801 round 6 bullet 9s, and it is the fix for the tenth silent site.**
    `engine.py`'s forward is ``self.graph_runner.replay(batch) if use_graph else
    self.model.forward()``, and `graph.py::GraphRunner.replay` refills the static buffers and
    calls ``g.replay()`` -- it never enters the model's python. So `gdn.py`'s
    ``batch.linear_snapshots = …`` and `ple.py`'s two assignments run at CAPTURE time and never
    again, and a replayed step reaches its commit carrying nothing (bullet 9q). Bullet 9r made
    that LOUD; this is what makes it stop happening.

    ⛔ Called off the CAPTURED run, never the eager warm-up. `_capture_set` forwards twice per
    size and only the second run's allocations come out of the graph's private pool; the warm-up's
    are ordinary allocations that no replay ever writes again, so binding those would commit the
    dummy request's state on every step -- the same bug one level down.

    ⛔ It RAISES on a verify capture that recorded nothing, because a graph captured without them
    can never carry them: the failure belongs at bring-up, where it costs 75 seconds, rather than
    at the first speculating step, where this round has now paid for it three times.

    ⚠ A missing PLE half is not an error: `model.py` builds PLE metadata only ``if self._ple:``,
    so a model without PLE layers legitimately records none. ``None`` is carried as ``None``.
    """
    if not getattr(batch, "linear_snapshots", None):
        raise MissingVerifySnapshots(
            "#801: a verify graph was captured from a forward that recorded no "
            "`batch.linear_snapshots`. A captured graph can only carry what the capturing "
            "forward left on the batch, so every replay of it would reach the commit empty and "
            "GDN's recurrent state would never advance. Capture is at bring-up, so this is the "
            "cheapest place the round can learn it."
        )
    return {name: getattr(batch, name, None) for name in CAPTURED_MEMOS}


def restore_memos(batch: "Batch", memos: dict) -> None:
    """Re-bind a captured forward's memos onto the live batch a replay just ran.

    ⛔ Every name in :data:`CAPTURED_MEMOS`, including the ``None`` ones, so a live batch can
    never be left half-bound -- a step carrying GDN's snapshots but not PLE's would commit one
    recurrence and silently abandon the other, which is precisely the split `commit_ple_state`
    exists as ONE call to prevent.
    """
    for name, value in memos.items():
        setattr(batch, name, value)


@dataclass(frozen=True)
class LinearSnapshot:
    """What one GDN layer holds across a verify forward so the step can be rolled back.

    Built by `Qwen4ExpGatedDeltaNet.forward` on a T > 1 decode batch and consumed by
    :func:`commit_linear_state` after the sampler has decided `accepted_len`.

    ⭐ The recurrent half is the kernel's own per-step cache: the fused fla kernel already
    supports it (`intermediate_states_buffer` + `disable_state_update`, and a `target_verify`
    mode in its own docstring) -- nothing in this engine passed them. With the state update
    DISABLED the pool still holds the step-ENTRY state, so :func:`commit_linear_state` is the
    only thing that advances it and an abandoned step leaves the pool untouched.

    ⭐ The conv half needs no kernel support at all. `conv_states` is literally the last
    ``kernel - 1`` RAW conv inputs, so the window after ``n`` accepted tokens is the pre-step
    window with this step's first ``n`` inputs shifted in -- reconstructible from what is here.
    """

    #: ``[bs, T, HV, K, V]`` -- the recurrent state after each of this step's tokens. Indexed by
    #: BATCH ROW, not by pool slot: the buffer is this forward's, so it is bs*T wide rather than
    #: num_slots*T (which would be T copies of the whole GDN pool).
    intermediate_states: "torch.Tensor" = None
    #: ``[bs, conv_dim, kernel-1]`` -- each request's conv window BEFORE this step ran.
    conv_window: "torch.Tensor" = None
    #: ``[num_tokens, conv_dim]`` -- this step's RAW (pre-conv) inputs, flat over the batch.
    conv_inputs: "torch.Tensor" = None
    #: ``[bs+1]`` -- the ragged query indptr this step ran with.
    cu_seqlens: "torch.Tensor" = None
    #: ``[bs]`` -- pool slot per request (``FLAMetadata.cache_indices``).
    cache_indices: "torch.Tensor" = None


def commit_linear_state(
    snapshot: LinearSnapshot,
    accepted_len: "torch.Tensor",
    *,
    recurrent_states: "torch.Tensor",
    conv_states: "torch.Tensor",
) -> None:
    """Advance one GDN layer's pool rows by exactly ``accepted_len`` tokens, in place.

    ``recurrent_states [num_slots, HV, K, V]`` and ``conv_states [num_slots, conv_dim, W]`` are
    one layer's views of `LinearStatePool`; ``accepted_len [bs]`` is `VerifyOutcome.accepted_len`
    -- 1 or 2 per request, and 1 for a request that was not speculating at all.

    ⛔⛆ **No host value is read.** `accepted_len` arrives as a device tensor straight off the
    sampler, and reading it here would put a device sync on the decode path (#912's hazard) and
    bake bullet 8's open question into this function. Everything below is a gather.
    """
    import torch

    slots = snapshot.cache_indices.to(torch.int64)
    n = accepted_len.to(torch.int64).reshape(-1)
    rows = torch.arange(n.numel(), device=n.device)

    # ── the recurrent state ────────────────────────────────────────────────────────────────
    # The state after `n` tokens IS the kernel's step-(n-1) intermediate. The pool still holds
    # the step-entry state (`disable_state_update=True`), so this is the whole advance.
    committed = snapshot.intermediate_states[rows, n - 1]
    recurrent_states.index_copy_(0, slots, committed.to(recurrent_states.dtype))

    # ── the conv window ────────────────────────────────────────────────────────────────────
    # Column j of the new window is column j+n of ``cat(old_window, inputs[start : start+n])``:
    # from the OLD window while j + n < W, from this step's inputs after that.
    width = conv_states.shape[-1]
    column = torch.arange(width, device=n.device)
    source = column.unsqueeze(0) + n.unsqueeze(1)                      # [bs, W]
    from_window = source < width
    old = snapshot.conv_window.gather(
        2, source.clamp(max=width - 1).unsqueeze(1).expand(-1, conv_states.shape[1], -1)
    )
    starts = snapshot.cu_seqlens.to(torch.int64)[:-1]
    token_row = starts.unsqueeze(1) + (source - width).clamp(min=0)    # [bs, W]
    fresh = snapshot.conv_inputs[token_row].transpose(1, 2)            # [bs, conv_dim, W]
    window = torch.where(from_window.unsqueeze(1), old, fresh)
    conv_states.index_copy_(0, slots, window.to(conv_states.dtype))


# ⛔⛆ #801 round 6 bullet 7: THE TWO RANKS MUST AGREE ON `accepted_len`, AND NOTHING MAKES THEM.
#   Each rank runs its own process over the same broadcast request stream and samples for itself;
#   `scheduler/io.py`'s patch 0011 is the PUB/SUB slow-joiner HANDSHAKE (#795), not a per-step
#   token relay. Round 5 observed the ranks agreeing on all 26 alpha cells, including the sampled
#   tier, and could not say why.
#
# ⭐⭐ THE MECHANISM, ESTABLISHED BY THIS BULLET. `engine/engine.py::Engine.__init__` calls
#   `torch.manual_seed(42)` before anything else runs, and the triton sampler's own per-device
#   generator (`kernel/triton/sampling.py::_gen_u`, `_UGEN[device]`) is constructed with torch's
#   DEFAULT seed, 67280421310721 -- a constant as well. Both ranks then consume the same
#   generators with the same code over the same shapes. ⇒ agreement is STRUCTURAL, not luck, and
#   this round is entitled to build a verify step on it.
#
# ⛔ AND IT IS FRAGILE IN A WAY NOTHING REPORTS. Lockstep is positional: it survives only while
#   both ranks draw the SAME NUMBER of values in the SAME ORDER. `engine/sample.py` never passes
#   a seed, so nothing re-anchors the streams per step. One rank-asymmetric draw -- an OOM retry,
#   a different graph-versus-eager route, #871's rank-0-only vision failure -- desynchronises them
#   for the rest of the load. Today that costs one token of discarded rank-1 text. In a verify
#   step it desynchronises `device_len`, the two ranks build different batch SHAPES, and the row
#   wedges on NCCL's 60 s watchdog with the cause long scrolled away.
#
# ⭐ THE PRECEDENT IS OURS: `scheduler/rank_agreement.py::any_rank_failed` (patch 0022, written
#   for #871) agrees an outcome across ranks on the HOST, over the gloo `tp_cpu_group`, with the
#   collective INJECTED rather than reached for. Both properties are load-bearing here too -- a
#   gloo group can never touch the NCCL path on any configuration, and an injected collective can
#   be gated without standing up a process group.
#
# ⛔ IT IS A BUDGETED CHECK, NOT A PER-STEP COLLECTIVE. A reduce on every decode step would put a
#   sync stall back on the path #912 spent a round clearing. Same shape as `DRAFTCHECK` and
#   `MASSCHECK`: a COUNT read once from the environment, spent one step at a time, deciding
#   nothing about what is loaded or run.

#: How many requests one fingerprint can carry. ⛔ The row is this wide WHATEVER the batch holds
#: -- see :func:`agree_row`.
# ⛔⛆ #801 round 6 bullet 8b: PLE IS A SECOND PAIR OF RECURRENCES, AND NOTHING OWNED THEM.
#   `models/qwen4_exp/ple.py` never goes through `build_fla_metadata`: it builds its own decode
#   metadata and carries its own per-request short-conv history (`ple_conv`) AND n-gram context
#   (`ple_ngram_ctx`). Both are recurrences in exactly the sense bullet 5 meant -- a rejected
#   token folded into either is there for the rest of the sequence, with nothing erroring.
#
# ⭐ The rollback is the SAME ARITHMETIC as `commit_linear_state`'s conv half, twice: a window of
#   the last W things, advanced by `n` of this step's things. Spelled once in `_advance_window`
#   and applied to the conv history (things = raw conv inputs) and the context (things = token
#   ids). ⛔ `accepted_len` is read as a DEVICE TENSOR here too -- same #912 hazard, same rule.


@dataclass(frozen=True)
class PLESnapshot:
    """What one PLE layer holds across a verify forward so its conv history can be rolled back.

    Built by `PLELayer._verify_conv` and consumed by :func:`commit_ple_state`. ⭐ There is no
    `conv_window` field, and that is the difference from `LinearSnapshot`: the verify conv writes
    NOTHING back, so the pool itself still holds the step-entry window at commit time and a clone
    would be a copy of a thing that never moved. (`LinearSnapshot` needs one because
    `causal_conv1d_varlen` writes the GDN state through its `cache_indices` argument.)
    """

    #: ``[num_tokens, ple_state_width]`` -- this step's RAW (pre-conv) inputs, flat over the batch.
    conv_inputs: "torch.Tensor" = None
    #: ``[bs+1]`` -- the ragged query indptr this step ran with.
    cu_seqlens: "torch.Tensor" = None
    #: ``[bs]`` -- PLE slot per request (`PLEMetadata.state_slots`).
    state_slots: "torch.Tensor" = None


@dataclass(frozen=True)
class PLEContextSnapshot:
    """What the forward holds so the n-gram context can be rolled back. ONE per forward, not one
    per layer: every PLE layer reads the same `ple_ngram_ctx` and `model.py` rolls it once."""

    #: ``[num_tokens]`` int64 -- this forward's token ids, flat over the batch.
    input_ids: "torch.Tensor" = None
    #: ``[bs+1]`` -- the ragged query indptr this step ran with.
    cu_seqlens: "torch.Tensor" = None
    #: ``[bs]`` -- PLE slot per request.
    state_slots: "torch.Tensor" = None


def _advance_window(
    window: "torch.Tensor", fresh: "torch.Tensor", starts: "torch.Tensor", n: "torch.Tensor"
) -> "torch.Tensor":
    """Shift ``n`` of this step's rows into a ``[bs, ..., W]`` window of the last W things.

    Column ``j`` of the result is column ``j + n`` of ``cat(window, fresh[start : start + n])``:
    from the OLD window while ``j + n < W``, from this step's own rows after that. Identical to
    the conv half of :func:`commit_linear_state` -- ⭐ and that is the point, because the PLE conv
    history and the n-gram context are the same kind of object as the GDN conv window.

    ``window`` is ``[bs, C, W]`` (conv) or ``[bs, W]`` (context); ``fresh`` is ``[total, C]`` or
    ``[total]`` to match. ⛔ ``n`` stays a device tensor throughout.
    """
    import torch

    width = window.shape[-1]
    column = torch.arange(width, device=n.device)
    source = column.unsqueeze(0) + n.unsqueeze(1)                       # [bs, W]
    from_window = source < width
    token_row = starts.unsqueeze(1) + (source - width).clamp(min=0)     # [bs, W]
    if window.dim() == 2:
        old = window.gather(1, source.clamp(max=width - 1))
        return torch.where(from_window, old, fresh[token_row])
    channels = window.shape[1]
    old = window.gather(2, source.clamp(max=width - 1).unsqueeze(1).expand(-1, channels, -1))
    return torch.where(from_window.unsqueeze(1), old, fresh[token_row].transpose(1, 2))


def commit_ple_state(batch: "Batch", accepted_len: "torch.Tensor", *, pool) -> None:
    """Advance PLE's two recurrences by exactly ``accepted_len`` tokens, in place.

    ⭐⭐ ONE call, not an open-coded sequence, and that is deliberate: the conv history is
    per-layer and the n-gram context is per-forward, so a call site that rolled one and forgot the
    other would look correct from both helpers' side and would serve a row whose hash window and
    conv window disagree about what it emitted. Bullet 6's rule, and bullet 8's mutation lesson.

    ``pool`` is the `LinearStatePool`; both slot states are reached through it. A batch with no
    verify step staged carries no snapshots and this is a no-op -- today's decode path, untouched.

    ⛔⛆ **No host value is read.** `accepted_len` arrives as a device tensor straight off the
    sampler; a `.tolist()` here would put a device sync on every speculative step (#912).
    """
    import torch

    from .config import PLE_CONV_STATE, PLE_NGRAM_STATE

    n = accepted_len.to(torch.int64).reshape(-1)
    for layer_id, snapshot in (getattr(batch, "ple_snapshots", None) or {}).items():
        conv_states = pool.slot_state(PLE_CONV_STATE, layer_id)
        slots = snapshot.state_slots.to(torch.int64)
        window = conv_states.index_select(0, slots).to(snapshot.conv_inputs.dtype)
        rolled = _advance_window(
            window, snapshot.conv_inputs, snapshot.cu_seqlens.to(torch.int64)[:-1], n
        )
        conv_states.index_copy_(0, slots, rolled.to(conv_states.dtype))

    context = getattr(batch, "ple_context_snapshot", None)
    if context is not None:
        context_pool = pool.slot_state(PLE_NGRAM_STATE)
        slots = context.state_slots.to(torch.int64)
        rolled = _advance_window(
            context_pool.index_select(0, slots).to(torch.int64),
            context.input_ids,
            context.cu_seqlens.to(torch.int64)[:-1],
            n,
        )
        context_pool.index_copy_(0, slots, rolled.to(context_pool.dtype))


AGREE_MAX_REQS = 256


class RankDisagreement(RuntimeError):
    """The TP ranks decided a verify step differently. ⛔ Raised on EVERY rank, never one.

    ⚠ Not a subclass of anything the scheduler catches on purpose: `scheduler.py`'s fail-closed
    handlers exist to keep a row alive through a per-request failure, and this is not one. Two
    ranks that have diverged cannot be brought back into step by refusing a request.
    """


def agree_row(
    accepted_len: "torch.Tensor",
    tokens: "torch.Tensor | None" = None,
    *,
    width: int = AGREE_MAX_REQS,
) -> "torch.Tensor":
    """This rank's verify step as ONE fixed-width host int64 row.

    Layout, ``2 + 3 * width`` long::

        [0]                      number of requests
        [1]                      number of forward rows
        [2 : 2+width]            `VerifyOutcome.accepted_len`, padded with -1
        [2+width : 2+3*width]    `VerifyOutcome.tokens`,       padded with -1

    ⛔⛆ **Fixed width so the check is safe to run on ranks that have ALREADY diverged.** Sized
    from this rank's own request count, two ranks holding different counts would hand gloo two
    different sizes -- and a size-mismatched collective hangs or dies *inside the check whose job
    is to turn a hang into a sentence*. Padded, a disagreement about the count is just another
    field to report.

    ⭐ **The committed TOKENS are in here, not only the lengths.** Two ranks that both reject
    agree on `accepted_len` and can still commit different ids: the lengths match, nothing
    wedges, and the two KV caches hold different text from that step on. That is the worse bug
    and it is the cheaper half of the row.

    ⚠ int64 on the HOST, `rank_agreement.py`'s own rule: gloo takes host tensors, and a device
    tensor would put the agreement back on the card. ⛔ The move off the device IS a sync, and it
    is why this is budgeted rather than run every step.
    """
    import torch

    lengths = accepted_len.reshape(-1).to(device="cpu", dtype=torch.int64)
    n_req = lengths.numel()
    assert n_req <= width, (
        f"#801 speccheck: {n_req} requests exceeds the fingerprint's {width}; truncating would "
        f"make two different batches fingerprint the same"
    )
    row = torch.full((2 + 3 * width,), -1, dtype=torch.int64)
    row[0] = n_req
    row[1] = 0
    row[2 : 2 + n_req] = lengths
    if tokens is not None:
        committed = tokens.reshape(-1).to(device="cpu", dtype=torch.int64)
        n_tokens = committed.numel()
        assert n_tokens <= 2 * width, (
            f"#801 speccheck: {n_tokens} rows exceeds the fingerprint's {2 * width}"
        )
        row[1] = n_tokens
        row[2 + width : 2 + width + n_tokens] = committed
    return row


def _disagreements(ranks: list, width: int, uids: Sequence, limit: int = 8) -> list:
    """Every column of the gathered stack the ranks do not agree on, most significant first.

    ⚠ ALL of them, not the first: a length divergence and a token divergence are different
    findings, and the first differing column is whichever field happens to sit earliest in the
    layout rather than whichever one explains the failure.
    """
    out = []
    for i in range(len(ranks[0])):
        column = [row[i] for row in ranks]
        if len(set(column)) == 1:
            continue
        if i == 0:
            entry = {"field": "n_req"}
        elif i == 1:
            entry = {"field": "n_tokens"}
        elif i < 2 + width:
            entry = {"field": "accepted_len", "index": i - 2}
            if entry["index"] < len(uids):
                entry["uid"] = uids[entry["index"]]
        else:
            entry = {"field": "tokens", "index": i - 2 - width}
        entry["per_rank"] = column
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def agree_accepted(
    accepted_len: "torch.Tensor",
    *,
    tokens: "torch.Tensor | None" = None,
    uids: Sequence = (),
    tp_size: int,
    gather=None,
    tp_rank: int | None = None,
    width: int = AGREE_MAX_REQS,
    step: int | None = None,
) -> dict:
    """Check that every TP rank decided this verify step identically. Raises if not.

    ``gather`` takes one host row and returns ``[tp_size, len(row)]``, rank-major -- built from
    the engine's own gloo group by :func:`gather_over`. It is INJECTED, never reached for: on the
    deployed arm (`--disable-pynccl`) `torch.distributed.group.WORLD` is an **NCCL** group and the
    engine's `tp_cpu_group` is a separate gloo `new_group`, so a function that reached for WORLD
    would be right on one configuration and silently wrong on the one this round measures.

    ⛔⛆ **The verdict is a pure function of the GATHERED rows.** Every rank holds the same stack,
    so every rank reaches the same answer and every rank raises. ``tp_rank`` is carried for the
    log line and is read by NOTHING that decides anything -- a primary-only verdict (the shape
    `logger.info_rank0` and `graph.py`'s `disable=not get_tp_info().is_primary()` use everywhere
    else in this engine) would kill one process and leave its peer in the next collective, which
    is the 60 s watchdog wedge this check exists to replace.

    ⛔ A run that could not happen DECLINES and says so. It must never report the same thing as a
    run that happened and found nothing (#866's trap, and `headcheck.py`'s own rule).
    """
    import torch

    report = {
        "ran": False,
        "declined": None,
        "agreed": None,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "step": step,
    }
    if tp_size <= 1:
        # ⛔ `rank_agreement.py`'s line: a single-card row cannot disagree with itself, and must
        #   not pay -- or risk -- one instruction of distributed machinery for it.
        report["declined"] = f"tp_size {tp_size}: one rank cannot disagree with itself"
        return report
    if gather is None:
        report["declined"] = "no cross-rank gather is wired (see Engine.mtp_set_rank_group)"
        return report

    row = agree_row(accepted_len, tokens, width=width)
    gathered = gather(row)
    assert tuple(gathered.shape) == (tp_size, row.numel()), (
        f"#801 speccheck: the gather returned {tuple(gathered.shape)}, expected "
        f"{(tp_size, row.numel())} -- one row per rank"
    )
    ranks = gathered.to(device="cpu", dtype=torch.int64).tolist()
    # ⭐ Everything reported comes off the GATHERED rank-0 row, so the report itself is identical
    #   on every rank and two logs can be read side by side without diffing noise.
    reference = ranks[0]
    n_req = reference[0]
    report.update(
        {
            "ran": True,
            "agreed": True,
            "n_req": n_req,
            "n_tokens": reference[1],
            "accepted_len": reference[2 : 2 + max(n_req, 0)],
        }
    )
    if all(other == reference for other in ranks[1:]):
        return report

    diffs = _disagreements(ranks, width, uids)
    report["agreed"] = False
    report["disagreements"] = diffs
    message = f"#801 speccheck: TP ranks disagree — {json.dumps(diffs)}"
    # ⛔ Printed BEFORE it is raised. A traceback can be swallowed by a caller's `except` (the
    #   scheduler has several fail-closed ones); a flushed stderr line with a fixed marker cannot,
    #   and it is what an analyser greps for.
    print(f"[#801] speccheck DISAGREEMENT: {json.dumps(report)}", file=sys.stderr, flush=True)
    raise RankDisagreement(message)


def gather_over(group):
    """An all-gather of one host row over ``group``, which must be **gloo**.

    ⛔⛆ Refused loudly on any other backend. `rank_agreement.py` states the reason: the ranks
    reach this holding different answers, and a device collective is precisely the thing that
    wedges when they are out of step. A check that rode the NCCL path would hang exactly when it
    was needed.
    """
    import torch
    import torch.distributed as dist

    backend = dist.get_backend(group)
    if backend != "gloo":
        raise ValueError(
            f"#801 speccheck: the cross-rank check needs a gloo group, got {backend!r}. "
            f"The engine's `tp_cpu_group` is gloo on every configuration; "
            f"`torch.distributed.group.WORLD` is NOT under --disable-pynccl."
        )
    size = dist.get_world_size(group=group)

    def gather(row: "torch.Tensor") -> "torch.Tensor":
        out = [torch.empty_like(row) for _ in range(size)]
        dist.all_gather(out, row, group=group)
        return torch.stack(out, dim=0)

    return gather


def speccheck_armed(model: "Qwen4ExpForCausalLM", plan: VerifyPlan) -> bool:
    """Spend one unit of the `FREETOKEN_MTP801_SPECCHECK` budget on this step, or decline.

    ⛔⛆ **The arming decision is the one thing that can turn this check INTO the wedge.** If one
    rank spends a unit and its peer does not, one process enters a collective the other never
    reaches -- the 60 s watchdog, caused by the instrument. ⇒ this reads exactly two things and
    both are identical on every rank: the countdown (from an environment both ranks are launched
    with, in ONE `docker run`) and the PLAN, which is built from the scheduler's own request list.
    Nothing sampled, nothing rank-local, and `test_agree_801.py` walks the ast to keep it that way.

    ⭐ A step with nothing to verify declines WITHOUT spending a unit (`headcheck.py`'s rule): a
    budget that empties with no check performed reads exactly like a check that passed.
    """
    if not plan.is_speculative:
        return False
    left = int(model._speccheck_left)
    if left <= 0:
        return False
    model._speccheck_left = left - 1
    return True


def speccheck_step(
    model: "Qwen4ExpForCausalLM",
    plan: VerifyPlan,
    outcome: VerifyOutcome,
    *,
    step: int | None = None,
) -> dict | None:
    """Arm, check, report. ``None`` when this step was not checked.

    ⭐ One call, so the verify step's wiring (bullet 8) cannot half-wire it -- in particular
    cannot pass the lengths and forget the committed tokens, which every test of
    :func:`agree_accepted` itself would still pass. *A helper that returns the right thing and a
    call site that ignores it look identical from the helper's side* (round 6 bullet 6).
    """
    if not speccheck_armed(model, plan):
        return None
    report = agree_accepted(
        outcome.accepted_len,
        tokens=outcome.tokens,
        uids=plan.uids,
        tp_size=getattr(model, "_speccheck_tp_size", 1),
        gather=getattr(model, "_speccheck_gather", None),
        tp_rank=getattr(model, "_speccheck_tp_rank", None),
        step=step,
    )
    print(f"[#801] speccheck: {json.dumps(report)}", file=sys.stderr, flush=True)
    return report


# ══ #801 round 6 bullet 9l: THE CHECK-ROW INSTRUMENT ══════════════════════════════════════════
#
# ⛔⛆ **WHAT IT MEASURES, AND WHY IT IS NOT ANOTHER LOAD.** Load 6 served with no traceback, logged
#   like a healthy speculating row, and emitted garbage from CHAR 8 -- decode step ~2-3, which is
#   essentially the first verify step. At α ≈ 1.8 % nearly every step REJECTS, and under greedy a
#   rejection commits the check row's own argmax. ⇒ **the check row is wrong**, and bullet 9k then
#   cleared everything about the verify path a host can reach. What is left is inside the T=2
#   forward, where the output text cannot separate one candidate from another.
#
#   ⇒ this records, per instrumented step, **the T=2 check row's argmax beside what a T=1 forward
#   at the same position produces**. It is the `SPECCHECK` / `MASSCHECK` shape: a COUNT read once
#   from the environment, spent one step at a time, deciding nothing about what is loaded or run.
#
# ⭐⭐ **TWO ARMS, NOT THREE, AND THE ENGINE DECIDED THAT.** The plan wanted a third arm -- a T=1
#   forward taken down the VERIFY path -- to separate *"the verify path is wrong at any width"*
#   from *"the second token corrupts row 0"*. **It does not exist.** Every verify branch here is
#   decided by SHAPE, never by a flag: `ple.py::_is_verify_step` is
#   ``meta.input_ids.shape[0] != len(meta.seq_lens)`` and `gdn.py::_verify_width` is
#   ``max(r.extend_len for r in batch.padded_reqs)``. A width-1 forward takes the plain decode
#   branch by construction, so **a T=1 forward IS plain decode** -- which is also exactly the arm
#   the question is about, since plain decode is what a lossless verify step must reproduce.
#
# ⚠ **THE SECOND FORWARD IS AN INSTRUMENT AND NEVER A DECODE NUMBER** -- banked on its own line,
#   the same rule the host sync (operator decision (a)) and the eager draft carry.
#
# ⛔⛆ **THE REFERENCE FORWARD HAS THREE OBSERVER EFFECTS AND EVERY ONE IS SILENT.** Named here
#   rather than discovered on the box, and each has a gate in `test_verify_wiring_801.py` §12:
#     1. it ADVANCES PLE's two recurrences, and :func:`commit_ple_state` reads the conv history's
#        ENTRY state back OFF THE POOL -- that is why `PLESnapshot` has no ``conv_window`` field.
#        An unrestored pool poisons the commit three lines later. ⇒ saved and restored here.
#     2. it OVERWRITES the tap buffer, which `shadow_step` drafts the NEXT step's token from one
#        call later (bullet 9h). An instrument that changed what the row drafts is not one.
#        ⇒ the buffer and the retained view are both saved and restored.
#     3. at bs > 1 the disk-PLE pinned buffer holds the T=2 fill's row order
#        ``[r0t0, r0t1, r1t0, r1t1]`` while a one-row-per-request forward reads ``pinned[:bs]`` and
#        wants ``[r0t0, r1t0]`` -- bullet 9c's shape, in a file bullet 9c did not open. ⇒ the
#        reference forward stages its OWN host fill through `model.forward_host_ctx`.
#   ⭐ GDN needs NO restore, and that is an ASSUMPTION with its own gate: `commit_linear_state`
#   OVERWRITES both tensors it is handed (two `index_copy_`s) rather than accumulating into them,
#   so the reference forward's scribble is gone by the time the step ends. If that ever becomes a
#   read-modify-write, GDN needs what PLE has here and its own suite would not say so.
#
# ⛔ **IT RUNS BEFORE EVERY COMMIT.** The reference forward must start from the step's ENTRY state
#   -- the state a plain decode step would have run from -- and each commit in :func:`verify_step`
#   advances exactly that. An instrument placed after them measures a forward the engine never
#   takes and reports a disagreement that is its own.


def reference_device_lens(plan: VerifyPlan) -> tuple:
    """Per request, the ``device_len`` a PLAIN decode step would run at this position.

    ⭐ ``device_len`` IS the shape (`stage_verify`'s header): `_make_positions`,
    `attention/linear.py::build_fla_metadata`, `engine/graph.py::_uniform_width` and
    `attention/qsa_sparse.py` all derive T from `Req.extend_len`. So rolling the lengths back by
    what the draft added is the whole of "make this forward one token wide", and at
    ``tokens_per_req == 1`` it is the plan's own length -- today's step, unchanged.
    """
    return tuple(
        length - (tokens - 1)
        for length, tokens in zip(plan.device_lens, plan.tokens_per_req)
    )


def bonus_entry_lens(plan: VerifyPlan) -> tuple:
    """Per request, the ``cached_len`` a PLAIN decode of the BONUS row would run at. #801 r6 b9bt.

    ⛔⛆ **THE OPPOSITE LENGTH TO :func:`reference_device_lens`, AND THAT IS THE WHOLE TRAP.** The
    check row is re-run at the step's ENTRY state, so that function rolls ``device_len`` BACK by
    what the draft added. The bonus row sits ONE TOKEN FURTHER ON -- a plain decode at position
    ``C+1`` has the pair's first token already in KV -- so this leaves ``device_len`` alone and
    moves ``cached_len`` FORWARD instead. ⇒ *do not copy the reference forward's entry-state
    handling; it is tuned for row 0 and is wrong for row 1.*

    ⭐ It is ``plan.positions[bonus_row]`` and nothing else: the bonus row's own position IS the
    count of tokens that precede it. A second derivation would be a second thing to keep in step
    with `plan_verify`.

    ⛔ ``None`` where the request has no bonus row, element for element with
    :attr:`VerifyPlan.bonus_rows` -- answering with the verify row's length instead would hand the
    instrument a request it must not report.

    ⚠ The pairing that makes it load-bearing: ``device_len - this == 1``. `Req.extend_len` IS the
    forward's width (`stage_verify`'s header), and at instrument time ``cached_len`` still holds
    ``C`` while ``device_len`` holds the staged ``C+2`` -- so an instrument that did not move
    ``cached_len`` would run the bonus row at **T=2**, the very width it exists to compare against.
    """
    return tuple(
        (None if row is None else int(plan.positions[row])) for row in plan.bonus_rows
    )


def reference_view(batch: "Batch", plan: VerifyPlan) -> "Batch":
    """A shallow copy of a staged batch narrowed to ONE ROW PER REQUEST, at the verified row.

    ⛔ A VIEW, never a mutation -- :func:`draft_view`'s rule and for the same reason: `engine.py`
    re-enters ``forward_batch(batch)`` with the original object and `graph.py` hands out buffers
    the next replay restages.

    ⛔⛆ **Five things are cleared, and four of them are MEMOS a shallow copy would carry.**
    ``VERIFY_STEP`` is the obvious one -- a view that kept the plan would be a second verify step
    wearing a one-row shape. The other four are written ONTO the batch by the forward that just
    ran: ``linear_verify_width`` (`gdn.py::_verify_width` memoises itself there, so a kept memo
    tells GDN the forward is two tokens wide while handing it one), ``fla_metadata`` (below), and
    the two snapshot bags, which belong to the step being measured and not to the measurement.
    ``attn_metadata`` is nulled rather than kept because the caller re-prepares it: a stale T=2
    metadata surviving into a one-row forward is the one shape that raises nothing.

    ⛔⛆ **``fla_metadata`` IS THE ONE THAT FAULTED THE GPU** (bullet 9, load 7 -- a `Memory Fault`
    in `fused_sigmoid_gating_delta_rule_update_kernel` on both ranks, reaching the log dressed as
    an NCCL watchdog timeout). `attention/linear.py::build_fla_metadata` writes the GDN query
    indptr here and `gdn.py`'s forward rebuilds it **only when it is absent**, so a carried T=2
    ``cu_seqlens = [0, 2]`` hands ``gdn_decode_fla`` one token row with an indptr claiming two.
    ⚠ Every branch test above it is innocent -- ``total != cache_indices.numel()`` is ``1 != 1``
    and ``linear_verify_width`` WAS cleared -- because nothing reads the indptr as a SHAPE until
    the kernel does, and then it reads the row that is not there.

    ⭐ The clearing is safe only because that rebuild is LAZY: it happens inside the reference
    forward, i.e. AFTER :func:`reference_forward` has rolled ``device_len`` back, so the rebuilt
    indptr is the one-token-per-request one. ⛔ `build_fla_metadata` reads ``batch.padded_reqs``
    while the rollback walks ``batch.reqs`` -- the same objects on a real staged batch, and a gate
    that builds its own fixture can make that false without noticing.

    ⛔ The enumeration, not this list, is what keeps the next memo from repeating load 7:
    `test_verify_wiring_801.py::TestEveryMemoTheForwardLeavesBehindIsCleared` asks `gdn.py`,
    `ple.py` and `qsa_sparse.py` what they write onto a batch and requires every name to appear
    here. ⇒ *enumerate what the batch CARRIES, not only what the instrument TOUCHES.*
    """
    import copy

    import torch

    view = copy.copy(batch)
    rows = torch.tensor(plan.verify_rows, dtype=torch.int64)
    view.input_ids = batch.input_ids[rows]
    view.positions = batch.positions[rows]
    view.out_loc = batch.out_loc[rows]
    view.attn_metadata = None
    view.fla_metadata = None
    view.linear_verify_width = None
    view.linear_snapshots = None
    view.ple_snapshots = None
    view.ple_context_snapshot = None
    if hasattr(view, VERIFY_STEP):
        delattr(view, VERIFY_STEP)
    return view


def checkrow_armed(model: "Qwen4ExpForCausalLM", plan: VerifyPlan) -> bool:
    """Spend one unit of the `FREETOKEN_MTP801_CHECKROW` budget on this step, or decline.

    ⭐ `headcheck.py`'s rule, the one :func:`speccheck_armed` already carries: a step with nothing
    to verify declines WITHOUT spending a unit, because *a budget that empties with no check
    performed reads exactly like a check that passed.*
    """
    if not plan.is_speculative:
        return False
    left = int(getattr(model, "_checkrow_left", 0))
    if left <= 0:
        return False
    model._checkrow_left = left - 1
    return True


def checkrow_rows(
    plan: VerifyPlan, logits: "torch.Tensor", reference_logits: "torch.Tensor"
) -> list:
    """One row per request: the two answers, and enough to tell WHICH KIND of wrong this is.

    ⭐ ``reference_rank`` is what separates *"the check row is noise"* from *"the check row is a
    hair off"*: a reference answer sitting at rank 1 behind a 0.01 gap and one buried at rank 900
    are different findings, and both read as ``agree: false``. ``max_abs_diff`` is the same
    question asked of the whole row rather than of the winner.

    ⚠ Every request is reported, INCLUDING one decoding plainly beside a verifying sibling --
    #949's ``mr=4`` shape, and the cheapest place a T=2 batch could corrupt a T=1 request.
    """
    return [
        _row_report(plan, int(plan.verify_rows[i]), int(uid), logits, reference_logits[i])
        for i, uid in enumerate(plan.uids)
    ]


def _row_report(
    plan: VerifyPlan, row: int, uid: int, logits: "torch.Tensor", reference: "torch.Tensor"
) -> dict:
    """ONE row of a two-forward comparison, in the currency bullet 9br settled on.

    ⛔ Shared by :func:`checkrow_rows` and :func:`bonus_report_rows` rather than copied. #801 r6
    b9bt: the two instruments compare DIFFERENT rows against DIFFERENT reference states, but the
    thing they report is the same thing, and `analyse_verify_801.py::checkrow` reads both -- a
    second spelling would be a second set of key names for one reader to keep in step with.

    ⚠ ``row`` is a row of the FLAT token dimension; ``reference`` is the one reference logits row
    that stands opposite it, already selected by the caller. Passing the whole reference tensor
    and an index here is what let the two callers' row orders drift in the first draft.
    """
    import torch

    verify = logits[row].to(torch.float32)
    reference = reference.to(torch.float32)
    v_top = torch.topk(verify, 2)
    r_top = torch.topk(reference, 2)
    verify_id = int(v_top.indices[0])
    reference_id = int(r_top.indices[0])
    return {
        "uid": int(uid),
        "row": int(row),
        "position": int(plan.positions[row]),
        "verify_id": verify_id,
        "reference_id": reference_id,
        "agree": verify_id == reference_id,
        "verify_gap": float(v_top.values[0] - v_top.values[1]),
        "reference_gap": float(r_top.values[0] - r_top.values[1]),
        "reference_rank": int((verify > verify[reference_id]).sum()),
        "max_abs_diff": float((verify - reference).abs().max()),
    }


def _save_ple_rows(batch: "Batch", pool) -> list:
    """The PLE pool rows a second forward can scribble on, cloned. #801 r6 b9bt.

    ⛔⛆ **ONE SPELLING, CALLED BY BOTH ENTRY-STATE FUNCTIONS.** `_enter_reference_state` and
    `_enter_bonus_state` need exactly the same save -- the conv history per layer and the one
    n-gram context -- and they need it for the same reason: `commit_ple_state`'s conv half reads
    the entry window back OFF THE POOL (which is why `PLESnapshot` has no ``conv_window`` field),
    so a pool left scribbled poisons the real commit three lines later.
    ⇒ a second copy would be a second thing to keep in step, in a file every served row runs. The
    two instruments differ in what they do NEXT -- one rewinds GDN, the other advances everything
    by one token -- and that difference stays in the callers, where it is the point.

    ⚠ Returns ``(states, slots, keep)`` triples, which is the shape :func:`_restore_reference_state`
    and :func:`_restore_bonus_state` both put back.
    """
    import torch

    from .config import PLE_CONV_STATE, PLE_NGRAM_STATE

    rows = []
    if pool is None:
        return rows
    for layer_id, snapshot in (getattr(batch, "ple_snapshots", None) or {}).items():
        states = pool.slot_state(PLE_CONV_STATE, layer_id)
        slots = snapshot.state_slots.to(torch.int64)
        rows.append((states, slots, states.index_select(0, slots).clone()))
    context = getattr(batch, "ple_context_snapshot", None)
    if context is not None:
        states = pool.slot_state(PLE_NGRAM_STATE)
        slots = context.state_slots.to(torch.int64)
        rows.append((states, slots, states.index_select(0, slots).clone()))
    return rows


def _enter_reference_state(
    model: "Qwen4ExpForCausalLM", batch: "Batch", plan: VerifyPlan, pool
) -> dict:
    """Clone everything the reference forward will scribble on -- and REWIND what this step's own
    forward already advanced. Both halves, in one function, because they cannot be separated.

    ⛔⛆ **#801 round 6 bullet 9p, AND LOAD 8 WAS SPENT ON THE HALF THAT WAS MISSING.** An
    instrument has two premises, not one. 9l gated *"does the reference forward leave a trace"* --
    `TestTheReferenceForwardLeavesNoTrace`, three observer effects, all in the WRITE direction. It
    never asked the READ direction: *does the step being measured reach the reference forward
    first?* For `gdn.py`'s conv pool it does. `_conv_verify` hands `pool.conv_states[li]` straight
    to `causal_conv1d_varlen`, whose own module header says the slot is "read as the left context
    ... **then refreshed in place with each request tail**" -- which is the entire reason
    `_conv_verify` clones `window` before the call and `LinearSnapshot.conv_window` is documented
    as the window BEFORE this step ran. ⇒ by the time `checkrow_step` runs, every GDN layer's conv
    window already ends at this step's LAST token, draft included, and a plain decode forward that
    reads it is reading a history shifted forward by ``T`` in 36 of the model's layers. Load 8
    reported 122 of 128 steps disagreeing; that is what it was measuring.

    ⭐⭐ **THE TWO GDN HALVES NEED OPPOSITE TREATMENT, AND THAT IS THE WHOLE TRAP.**
    ``recurrent_states`` is left at the step-ENTRY state by ``disable_state_update=True`` (bullet
    9k read the kernel's own store), so it is ALREADY what a plain decode forward should read and
    rewinding it would move it AWAY. ``conv_states`` gets no such protection -- different kernel,
    different semantics -- and 9l carried the recurrent half's correctness onto GDN as a whole.
    `test_the_recurrent_state_is_left_exactly_as_the_verify_forward_left_it` pins the other side so
    this cannot be "fixed" into symmetry later.

    ⚠ The rewind target is not derived here: ``LinearSnapshot.conv_window`` IS the entry window and
    ``.cache_indices`` IS the slot list, both already on the batch because the rollback needs them.
    A second derivation would be a second thing to keep in step with `_conv_verify`.

    ⛔ PLE's two pools take the save half only. `ple.py`'s verify path advances NEITHER of them
    (bullet 8b) -- `commit_ple_state` is the only advance -- so they, too, already hold what a
    plain decode forward should read.
    """
    import torch

    ple = _save_ple_rows(batch, pool)
    gdn = []
    if pool is not None:
        # ⛔⛆ #801 bullet 9r: LOUD, never `or {}`. The rewind below is this instrument's whole
        #   correctness and it has nothing to rewind FROM without the snapshots -- which is how
        #   load 9 reproduced load 8 byte for byte with 9p's fix mounted.
        required = require_linear_snapshots(batch, plan, where="the reference forward's rewind")
        for layer_id, snapshot in required.items():
            states = pool.conv_states[pool.local_index(layer_id)]
            slots = snapshot.cache_indices.to(torch.int64)
            gdn.append((states, slots, states.index_select(0, slots).clone()))
            states.index_copy_(0, slots, snapshot.conv_window.to(states.dtype))
    buffer = getattr(model.model, "multi_stream_buffer", None)
    return {
        "ple": ple,
        "gdn": gdn,
        "tap": None if buffer is None else (buffer, buffer.clone()),
        "tap_view": getattr(model, "_multi_stream", None),
    }


def _restore_reference_state(model: "Qwen4ExpForCausalLM", saved: dict) -> None:
    """Put back exactly what :func:`_enter_reference_state` took, including the rewind.

    ⛔⛆ The GDN rows are restored to the ADVANCED window, not left at the entry one. It is true
    that `commit_linear_state` overwrites this pool three lines later and would paper over a
    missing restore -- and `TestTheAssumptionThatLetsGDNGoUnrestored` already carries that
    dependency ONCE, for the reference forward's own `_conv_decode` write. A second thing leaning
    on it is how the first one stopped being checked.
    """
    for states, slots, keep in saved["ple"]:
        states.index_copy_(0, slots, keep)
    for states, slots, keep in saved["gdn"]:
        states.index_copy_(0, slots, keep)
    if saved["tap"] is not None:
        buffer, keep = saved["tap"]
        buffer.copy_(keep)
    model._multi_stream = saved["tap_view"]


def reference_forward(
    model: "Qwen4ExpForCausalLM", batch: "Batch", plan: VerifyPlan, *, linear_pool=None
) -> "torch.Tensor":
    """Run the PLAIN decode forward this step's requests would have taken, and leave no trace.

    ⛔⛆ Everything it touches is restored in a ``finally``, including the requests' ``device_len``
    -- the commits three lines later advance from the PLAN's lengths, so a request left at the
    reference length would be advanced twice and the row would skip a position with nothing
    erroring.

    ⚠ `Engine.forward_batch`'s own re-entry rule (`mtp_shadow_step`'s comment): ``ctx.forward_batch``
    has already EXITED by the time :func:`verify_step` runs, so it is re-entered here. ⭐ Unlike
    the head's, this forward IS the backbone and it DOES ship PLE tensors, so
    `forward_host_ctx` -- the disk-PLE prefetch -- is entered alongside it rather than
    deliberately skipped.
    """
    from freetoken.core import get_global_ctx

    ctx = get_global_ctx()
    view = reference_view(batch, plan)
    reqs = list(batch.reqs)
    staged_lens = tuple(r.device_len for r in reqs)
    saved = _enter_reference_state(model, batch, plan, linear_pool)
    try:
        for req, length in zip(reqs, reference_device_lens(plan)):
            req.device_len = length
        ctx.attn_backend.prepare_metadata(view)
        host_ctx = getattr(model, "forward_host_ctx", None)
        if host_ctx is None:
            with ctx.forward_batch(view):
                return model.forward()
        with ctx.forward_batch(view), host_ctx(view, False):
            return model.forward()
    finally:
        for req, length in zip(reqs, staged_lens):
            req.device_len = length
        _restore_reference_state(model, saved)


# ══ #801 round 6 bullet 9bt: THE BONUS ROW ════════════════════════════════════════════════════
#
# ⭐⭐⭐ **LOAD 27 EXONERATED ROW 0, AND THAT IS WHY THIS EXISTS.** `CHECKROW` compared the T=2
#   check row against a plain T=1 decode over 1,200 consecutive speculating steps, five requests,
#   both ranks, positions 66 -> 13,494: **1,198 of 1,200 rows BIT-IDENTICAL, 0 argmax
#   disagreements.** Under greedy that settles two of the three tokens a verify step can emit --
#   on a rejection the committed token IS row 0's argmax, and on an acceptance the draft was
#   accepted precisely BECAUSE it equals row 0's argmax. ⇒ every token row 0 decides is right,
#   and the only emitted token left is the **BONUS token from row 1**.
#
# ⛔⛆ **THE BONUS ROW HAS NO T=1 COUNTERPART ANYWHERE ON THE ARM.** A handoff proposed reading it
#   for free off the arm's own next step; `plan_verify`'s arithmetic refuses that, and
#   `test_verify_801.py::TestWhatAPlainDecodeOfTheBonusRowWouldRunAt` carries the refusal:
#
#     * on an ACCEPT the step commits two, ``cached_len`` goes C -> C+2, and the next step's
#       row 0 is at C+2 -- position C+1 is never decoded plainly;
#     * on a REJECT the next step's row 0 IS at C+1, but its input token is the COMMITTED one,
#       which by construction is not the draft the bonus row was fed.
#
#   ⇒ there is no free reading, the instrument runs a forward of its own, and the round's blind
#   spot has a shape: *the one row the arm never re-decodes is the one nothing has measured.*
#
# ⭐⭐ **THE ENTRY STATE IS THE DEPLOYED COMMIT PATH AT n = 1, NOT A SECOND REWIND.** A plain
#   decode of the bonus row runs from the state AFTER the pair's first token is committed, which
#   is exactly what :func:`commit_linear_state` and :func:`commit_ple_state` write when
#   ``accepted_len`` is all ones. ⇒ the instrument calls the production functions rather than
#   re-deriving their arithmetic, so it cannot drift from what the engine actually commits.
#   ⛔ Neither of them reads the pool it writes for the half that matters here -- GDN's recurrent
#   row comes from ``intermediate_states[:, 0]`` and its conv window from the snapshot's ENTRY
#   window plus this step's inputs -- so the advance is correct however `_conv_verify` left the
#   pool. PLE's conv half DOES read the pool, and bullet 8b established that a verify forward
#   advances neither PLE recurrence, so it reads the entry state it needs.
#
# ⛔⛆ **IT SAVES BOTH GDN HALVES, WHICH `_enter_reference_state` DOES NOT.** That one leans on
#   `commit_linear_state` overwriting the pool three lines later, a dependency
#   `TestTheAssumptionThatLetsGDNGoUnrestored` carries ONCE -- and whose own docstring says a
#   second thing leaning on it is how the first stopped being checked. This instrument leans on
#   nothing: it saves what it touches and puts it back.
#
# ⚠ **SAME BANNER AS `CHECKROW`, AND FOR THE SAME REASON.** The second forward writes the KV
#   cache and QSA's index caches at the bonus row's position. ⇒ on a BONUSROW load the served
#   text, byte identity, alpha and the decode figure are NOT evidence. The launcher says so on
#   stdout so it lands in the bank (the round's own rule, 9p).


def is_mixed_batch(plan: VerifyPlan) -> bool:
    """Did ANY request in this step forward a single token? #801 r6 b9bt.

    ⛔ ONE spelling, asked by :func:`bonus_armed` and :func:`bonus_step` both. The same question
    written twice is how the decline and the notice that explains it drift apart -- and this round
    has already banked a survivor of exactly that shape one function away
    ([[feedback_a_mutation_must_replace_not_duplicate]], in its one-concept-two-spellings form).
    """
    return any(t < 2 for t in plan.tokens_per_req)


def bonus_armed(model: "Qwen4ExpForCausalLM", plan: VerifyPlan) -> bool:
    """Spend one unit of the `FREETOKEN_MTP801_BONUSROW` budget on this step, or decline.

    ⭐ `headcheck.py`'s rule, the one :func:`speccheck_armed` and :func:`checkrow_armed` already
    carry: a step with nothing to measure declines WITHOUT spending, because *a budget that
    empties with no check performed reads exactly like a check that passed.*

    ⛔⛆ **IT ALSO DECLINES ON A MIXED BATCH, AND THAT IS A SCOPE LIMIT, NOT A FILTER.** A batch
    where some request forwarded one token has no bonus row for that request, so the one-row-per-
    request view cannot be built without DROPPING requests -- a different forward shape, with a
    different slot mapping, that this instrument does not gate. It is also the shape where the
    entry advance would be wrong: ``n = 0`` is not a value :func:`commit_linear_state` supports
    (``intermediate_states[rows, n - 1]`` would index the pair's LAST state), and advancing a
    plain sibling by one would leave it reading a history it never had.
    ⇒ #949's ``mr=4`` shape is out of scope and :func:`bonus_step` SAYS SO in the bank rather
    than reporting nothing -- a load that measured nothing must not read like a load that was
    clean (#866's shape).
    """
    # ⛔⛆ THE TWO GUARDS ARE NOT THE SAME GUARD, and a mutation sweep is what said so. For every
    #   NON-EMPTY plan `not is_speculative` implies `is_mixed_batch` (all t == 1 makes any t < 2
    #   true), so neutering this line changed nothing the gates could construct and it SURVIVED.
    #   They part on the EMPTY plan: `is_speculative` refuses it, `is_mixed_batch` does not -- and
    #   without this line an empty plan would SPEND a unit and build a zero-row view.
    #   ⚠ `stage_verify` refuses a non-speculative plan today, so nothing reaches here empty. The
    #   guard is kept and PINNED anyway (`test_an_empty_plan_spends_nothing`) rather than banked
    #   as an equivalence: an unfalsifiable guard is one nobody notices going wrong.
    if not plan.is_speculative:
        return False
    if is_mixed_batch(plan):
        return False
    left = int(getattr(model, "_bonusrow_left", 0))
    if left <= 0:
        return False
    model._bonusrow_left = left - 1
    return True


def bonus_view(batch: "Batch", plan: VerifyPlan) -> "Batch":
    """A shallow copy of a staged batch narrowed to ONE ROW PER REQUEST, at the BONUS row.

    ⭐ It DELEGATES the memo clearing to :func:`reference_view` and re-points only the three
    per-row tensors it overwrites wholesale. The memos -- ``VERIFY_STEP``,
    ``linear_verify_width``, ``fla_metadata`` (⛔⛆ **the one that FAULTED THE GPU** on load 7) and
    the two snapshot bags -- are enumerated from what the forward CARRIES rather than from what an
    instrument touches, and load 7's lesson is that the defect was an INCOMPLETE LIST, not a wrong
    line. ⇒ there is exactly ONE list, `TestEveryMemoTheForwardLeavesBehindIsCleared` gates it,
    and a memo added there reaches this view by construction.

    ⛔ The only difference is the row selector, and it is the whole instrument:
    :attr:`VerifyPlan.bonus_rows` rather than :attr:`VerifyPlan.verify_rows`. Every element is an
    int because :func:`bonus_armed` has already declined the mixed batch that would make one
    ``None``.
    """
    import torch

    rows = plan.bonus_rows
    assert all(row is not None for row in rows), (
        "bonus_view was handed a MIXED batch -- bonus_armed declines those, and a view built "
        f"from {rows} would silently drop a request from the forward"
    )
    # ⛔⛆ **BUILT ON :func:`reference_view` RATHER THAN COPIED FROM IT, AND LOAD 7 IS WHY.** The
    #   memo list is the part that has already cost this round a GPU fault, and its own gate says
    #   the defect was an INCOMPLETE LIST rather than a wrong line. A second copy of it here would
    #   be a second list to forget -- so the clears happen ONCE, and this only re-points the three
    #   per-row tensors it overwrites wholesale.
    view = reference_view(batch, plan)
    index = torch.tensor([int(row) for row in rows], dtype=torch.int64)
    view.input_ids = batch.input_ids[index]
    view.positions = batch.positions[index]
    view.out_loc = batch.out_loc[index]
    return view


def _enter_bonus_state(
    model: "Qwen4ExpForCausalLM", batch: "Batch", plan: VerifyPlan, pool
) -> dict:
    """Clone everything the bonus forward will scribble on, then ADVANCE the recurrences by ONE.

    ⛔⛆ **THE ADVANCE IS THE OPPOSITE OF :func:`_enter_reference_state`'s REWIND, AND COPYING THAT
    FUNCTION HERE IS THE NAMED TRAP.** The check row is re-run at the step's ENTRY state, so that
    one rewinds GDN's conv window to ``LinearSnapshot.conv_window`` and deliberately leaves the
    recurrent state alone (``disable_state_update=True`` already parked it at entry). The bonus
    row is a plain decode ONE TOKEN LATER, so BOTH halves have to move forward by exactly one --
    and so do PLE's two.

    ⭐⭐ **AND THE THING THAT MOVES THEM IS THE ENGINE'S OWN COMMIT, AT ``accepted_len = 1``.**
    :func:`commit_linear_state` writes ``intermediate_states[:, 0]`` into the recurrent pool and
    the entry window plus this step's first input into the conv pool; :func:`commit_ple_state`
    rolls PLE's conv history and n-gram context by one. That IS "after the pair's first token",
    by the same code the engine runs three lines later -- so the instrument cannot drift from the
    state the row actually goes on to serve from. ⚠ A hand-written advance here would be a second
    copy of that arithmetic and the round has already banked what those cost.

    ⛔⛆ **BOTH GDN HALVES ARE SAVED, UNLIKE THE REFERENCE FORWARD'S.** `_restore_reference_state`
    restores only the conv rows and leans on `commit_linear_state` overwriting the recurrent pool
    afterwards; `TestTheAssumptionThatLetsGDNGoUnrestored` carries that dependency once and warns
    in as many words that a second leaner is how the first stops being checked. This one saves
    what it writes.
    """
    import torch

    ple = _save_ple_rows(batch, pool)
    gdn = []
    buffer = getattr(model.model, "multi_stream_buffer", None)
    saved = {
        "ple": ple,
        "gdn": gdn,
        "tap": None if buffer is None else (buffer, buffer.clone()),
        "tap_view": getattr(model, "_multi_stream", None),
    }
    if pool is None:
        return saved
    # ⛔⛆ LOUD, never `or {}` -- bullet 9r's rule. The advance below IS this instrument's
    #   correctness and it has nothing to advance FROM without the snapshots.
    required = require_linear_snapshots(batch, plan, where="the bonus forward's advance")
    # ⛔⛆ **ON THE FORWARD'S DEVICE, AND EVERY CPU GATE IN THIS ROUND IS BLIND TO IT.**
    #   `commit_linear_state` and `_advance_window` both build their index tensors with
    #   ``device=n.device``, and `VerifyOutcome`'s own header says the real `accepted_len` is "on
    #   the logits' own device". A host-side ``torch.ones`` therefore indexes a GPU pool with a
    #   CPU index and raises -- on the box, ~75 s into a load, and NOWHERE in a CPU-only
    #   container, where every tensor is already on the same device. ⇒ the round's standing rule,
    #   *a CPU suite can be green while the real path is broken*, with a concrete instance.
    one = torch.ones(len(plan.uids), dtype=torch.int64, device=batch.input_ids.device)
    try:
        for layer_id, snapshot in required.items():
            local = pool.local_index(layer_id)
            recurrent = pool.recurrent_states[local]
            conv = pool.conv_states[local]
            slots = snapshot.cache_indices.to(torch.int64)
            # ⭐ SAVED BEFORE THE LAYER IS ADVANCED, so a raise halfway through the loop still has
            #   every row it has touched in `gdn` for the `except` below to put back.
            gdn.append((recurrent, slots, recurrent.index_select(0, slots).clone()))
            gdn.append((conv, slots, conv.index_select(0, slots).clone()))
            commit_linear_state(
                snapshot, one, recurrent_states=recurrent, conv_states=conv
            )
        commit_ple_state(batch, one, pool=pool)
    except BaseException:
        # ⛔⛆ **A PARTIAL ADVANCE IS NOT A NO-OP.** `_enter_reference_state` may raise before it
        #   writes anything; this one writes 36 layers in a loop, so a raise in the middle would
        #   leave the pool advanced for some layers and not others -- and `bonus_forward`'s
        #   ``finally`` never runs, because the exception escapes before its ``try`` is entered.
        #   The row would then serve from a GDN state one token ahead of its own text, with
        #   nothing erroring. ⚠ `BaseException`, not `Exception`: a KeyboardInterrupt or a
        #   CUDA OOM must not leave the pool half-advanced either.
        _restore_bonus_state(model, saved)
        raise
    return saved


def _restore_bonus_state(model: "Qwen4ExpForCausalLM", saved: dict) -> None:
    """Put back exactly what :func:`_enter_bonus_state` took, including the one-token advance.

    ⛔ The PLE rows are restored BEFORE `commit_ple_state` runs for real, and that is not
    optional: its conv half reads the entry window back OFF THE POOL (which is why `PLESnapshot`
    has no ``conv_window`` field), so a pool left advanced would be rolled a second time and the
    row would serve from a history one token ahead of its own text.
    """
    for states, slots, keep in saved["ple"]:
        states.index_copy_(0, slots, keep)
    for states, slots, keep in saved["gdn"]:
        states.index_copy_(0, slots, keep)
    if saved["tap"] is not None:
        buffer, keep = saved["tap"]
        buffer.copy_(keep)
    model._multi_stream = saved["tap_view"]


def bonus_host_ids(batch: "Batch", plan: VerifyPlan) -> tuple:
    """Per request, the ``(position, token id)`` the bonus forward's HOST fill will ask for and
    the host does not yet know. ``None`` where the request has no bonus row. #801 r6 b9bu.

    ⛔⛆ **LOAD 28 DIED HERE, ON BOTH RANKS, AT THE FIRST SPECULATING STEP**, and no CPU gate could
    have seen it. `ple_disk._decode_runs` takes its context ``req.device_len - req.extend_len``
    back — which for a ONE-ROW forward whose ``device_len`` still holds the staged ``C+2`` is
    position ``C+1`` — and `_undrained_context` then asks for the two ids BEFORE it: ``C-1`` and
    ``C``. Position ``C`` is this step's ROW 0 INPUT, and at host-fill time it is in neither
    place `host_token_id` looks:

    * ``req.input_ids`` holds ``cached_len`` ids (``0 … C-1``) — `overlap_loop` forwards BEFORE it
      drains, so the token row 0 is forwarding has not been appended yet;
    * the undrained record is written by :func:`commit_verify`, which runs **27 lines AFTER**
      :func:`bonus_step` in `verify_step` — deliberately, because the instrument must measure the
      state BEFORE the real commits.

    ⇒ ``RuntimeError: #801: req 1 has no host id at position 66 — input_ids holds 66 and the
    undrained record holds []; cached_len=67, device_len=68``.

    ⭐ **The id is not missing, only unsynced: it is the token the batch is forwarding at row 0**,
    so this reads it from ``batch.input_ids`` rather than re-deriving it. ⚠ ONE device sync for
    the whole batch (a gather, then a single ``tolist``) — which this instrument has already
    bought: :func:`bonus_step` syncs ``outcome.accepted`` a few lines later and runs an entire
    extra backbone forward, and its load is a DIAGNOSTIC whose decode figure is not evidence.

    ⛔ :attr:`VerifyPlan.verify_rows` is row 0's index and :attr:`VerifyPlan.positions` is where
    its input token sits — the plan's own arithmetic, not a second copy of it.
    """
    # ⛔ Local, like every other torch user in this file: a module-level ``import torch`` breaks
    #   host importability, which `conftest.py`'s ``_TORCH_ONLY`` list gates on purpose (bullet 2).
    import torch

    rows = plan.verify_rows
    wanted = [row for row, bonus in zip(rows, plan.bonus_rows) if bonus is not None]
    if not wanted:
        return tuple(None for _ in plan.bonus_rows)
    index = torch.tensor(wanted, dtype=torch.int64, device=batch.input_ids.device)
    ids = batch.input_ids.index_select(0, index).tolist()
    out: list = []
    seen = 0
    for row, bonus in zip(rows, plan.bonus_rows):
        if bonus is None:
            out.append(None)
            continue
        out.append((int(plan.positions[row]), int(ids[seen])))
        seen += 1
    return tuple(out)


def bonus_forward(
    model: "Qwen4ExpForCausalLM", batch: "Batch", plan: VerifyPlan, *, linear_pool=None
) -> "torch.Tensor":
    """Run the PLAIN decode forward the BONUS row's position would have taken, and leave no trace.

    ⛔⛆ **IT MOVES ``cached_len``, NOT ``device_len``, AND THAT IS THE WHOLE SHAPE.**
    `Req.extend_len` is ``device_len - cached_len`` and it IS the forward's width. At this point
    ``cached_len`` still holds ``C`` (the commits run three lines later) and ``device_len`` holds
    the staged ``C+2`` -- so leaving both alone would run the bonus row at **T=2**, the very width
    the instrument exists to compare against. :func:`bonus_entry_lens` is the length each request
    takes instead, and it is the bonus row's own position.

    ⛔ Everything is restored in a ``finally``, ``cached_len`` included: :func:`commit_verify`
    advances FROM it a few lines later and asserts ``device_len == cached_len + 1`` at the next
    step's entry, so a request left advanced here would skip a position with nothing erroring.
    """
    from freetoken.core import get_global_ctx

    ctx = get_global_ctx()
    view = bonus_view(batch, plan)
    reqs = list(batch.reqs)
    staged_cached = tuple(r.cached_len for r in reqs)
    # ⛔⛆ #801 r6 b9bu: BOTH reads are taken BEFORE anything is written, and the undrained record
    #   is saved as an OBJECT-or-``None`` rather than a dict, so "the request had no record" and
    #   "the request had an empty record" restore differently. `commit_verify` writes ``{}`` on a
    #   rejecting commit and that is a value, not an absence.
    staged_undrained = tuple(getattr(r, UNDRAINED_IDS, None) for r in reqs)
    host_ids = bonus_host_ids(batch, plan)
    saved = _enter_bonus_state(model, batch, plan, linear_pool)
    try:
        for req, length in zip(reqs, bonus_entry_lens(plan)):
            req.cached_len = length
        # ⛔⛆ MERGED, never replaced. A request whose PREVIOUS step accepted two carries the
        #   accepted draft at ``C-1`` in this same record (`commit_verify`), and that is one of
        #   the two ids this forward's host fill is about to ask for. Dropping it would trade
        #   load 28's raise at ``C`` for the same raise at ``C-1`` on exactly the accepted steps
        #   the instrument exists to read. ⇒ [[feedback_a_mutation_must_replace_not_duplicate]]
        #   does not apply here: this is one record with two writers, not one concept twice.
        for req, prior, entry in zip(reqs, staged_undrained, host_ids):
            if entry is None:
                continue
            position, token = entry
            setattr(req, UNDRAINED_IDS, {**(prior or {}), position: token})
        ctx.attn_backend.prepare_metadata(view)
        host_ctx = getattr(model, "forward_host_ctx", None)
        if host_ctx is None:
            with ctx.forward_batch(view):
                return model.forward()
        with ctx.forward_batch(view), host_ctx(view, False):
            return model.forward()
    finally:
        for req, length, prior in zip(reqs, staged_cached, staged_undrained):
            req.cached_len = length
            # ⛔ An absence is restored as an ABSENCE. `host_token_id` reads the record with
            #   ``getattr(req, UNDRAINED_IDS, {})``, so leaving a ``{}`` behind serves the same
            #   ids -- but `commit_verify` is not the only writer any more, and a request that
            #   leaves this instrument carrying an attribute it did not arrive with is a
            #   difference the NEXT bullet would have to rule out. Put back exactly what was taken.
            if prior is None:
                if hasattr(req, UNDRAINED_IDS):
                    delattr(req, UNDRAINED_IDS)
            else:
                setattr(req, UNDRAINED_IDS, prior)
        _restore_bonus_state(model, saved)


def bonus_report_rows(
    plan: VerifyPlan,
    logits: "torch.Tensor",
    reference_logits: "torch.Tensor",
    accepted: "list[bool]",
) -> list:
    """One row per request: the bonus row's answer, a plain decode's, and whether it was SERVED.

    ⭐⭐ ``accepted`` is the field :func:`checkrow_rows` has no use for and this one cannot do
    without. A bonus row on a REJECTED step is sampled and written -- that is what keeps the
    scatter shape fixed for a captured graph -- and then dropped by `published_tokens`, so a
    disagreement there never reached a client. A disagreement on an ACCEPTED step is served text.
    ⇒ two populations in one report, separated at read time rather than filtered at write time
    (bullet 9be's rule: *demote, do not drop*).
    """
    assert len(accepted) == len(plan.uids), (
        f"{len(accepted)} acceptance flag(s) for {len(plan.uids)} request(s)"
    )
    rows = []
    for i, uid in enumerate(plan.uids):
        row = _row_report(
            plan, int(plan.bonus_rows[i]), int(uid), logits, reference_logits[i]
        )
        row["accepted"] = bool(accepted[i])
        rows.append(row)
    return rows


def bonus_step(
    model: "Qwen4ExpForCausalLM",
    batch: "Batch",
    plan: VerifyPlan,
    logits: "torch.Tensor",
    outcome: "VerifyOutcome",
    *,
    linear_pool=None,
    step: int | None = None,
) -> dict | None:
    """Arm, run the bonus forward, report. ``None`` when this step was not instrumented.

    ⭐ ONE call from :func:`verify_step`, for the reason bullet 7 gave for :func:`speccheck_step`
    and bullet 9l repeated for :func:`checkrow_step`: a call site that armed the budget and
    skipped the forward, or ran the forward after the commits, would look correct from every
    helper's side.

    ⛔⛆ **A MIXED BATCH IS ANNOUNCED, ONCE, AND NOT SILENTLY SKIPPED.** :func:`bonus_armed`
    declines it without spending -- but a load whose every step is mixed (#949's ``mr=4``) would
    then bank NOTHING, which reads exactly like a load that ran clean. The notice is latched so it
    costs one line rather than one per step, and it names the shape it saw.
    """
    if not plan.is_speculative:
        return None
    if is_mixed_batch(plan) and not getattr(model, "_bonusrow_mixed_notice", False):
        model._bonusrow_mixed_notice = True
        if int(getattr(model, "_bonusrow_left", 0)) > 0:
            print(
                "[#801] bonusrow: "
                + json.dumps(
                    {
                        "step": step,
                        "skipped": "mixed batch",
                        "tokens_per_req": list(plan.tokens_per_req),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    if not bonus_armed(model, plan):
        return None
    reference = bonus_forward(model, batch, plan, linear_pool=linear_pool)
    accepted = [bool(x) for x in outcome.accepted.reshape(-1).tolist()]
    rows = bonus_report_rows(plan, logits, reference, accepted)
    report = {
        "step": step,
        "n": len(rows),
        "agree_n": sum(1 for row in rows if row["agree"]),
        "accepted_n": sum(1 for row in rows if row["accepted"]),
        "rows": rows,
    }
    print(f"[#801] bonusrow: {json.dumps(report)}", file=sys.stderr, flush=True)
    return report


#: ⚠ #801 r6 b9w: a bf16 kernel against an fp32 reference, so ``agree`` is a CONVENIENCE and the
#: number to read is ``row0_worst``. ⛔ The banked 9n rule, restated because this is its second
#: instrument: **read the magnitude, not the boolean.**
GDNCHECK_ATOL = 1e-2


def gdncheck_layer(
    model,
    *,
    layer_id: int,
    local_layer: int = 0,
    step: int,
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    g: "torch.Tensor",
    beta: "torch.Tensor",
    entry_state: "torch.Tensor",
    cu_seqlens: "torch.Tensor",
    fused_out: "torch.Tensor",
    t1_out: "torch.Tensor | None" = None,
    atol: float = GDNCHECK_ATOL,
) -> dict | None:
    """What the fla kernel's verify mode RETURNS, against the pure-torch oracle. #801 r6 b9w.

    ⛔⛆ **BULLET 9t FINDING 1 IS WHY THIS EXISTS.** `gdn_reference.recurrent_gated_delta_rule`
    returns ``(output, state)``; every gate in this round is ``_, state = …``. The round has
    checked what a verify forward STORES and never what it RETURNS -- and loads 6/10 and 9t's
    check row all say the returned row 0 is wrong.

    ⭐⭐ **SOUND WHERE `CHECKROW` IS NOT.** ``disable_state_update=True`` leaves the pool holding
    the step-ENTRY state, so the caller passes ``rec[slots]`` and the reference runs on the same
    inputs the kernel just consumed. ⇒ **no second backbone forward, no rewind, and no KV or
    QSA-index write** -- 9p's still-unaffirmed judgement call does not apply to this instrument.

    ⭐ **Row 0 is pooled separately and that is the whole point.** Under greedy a REJECTED step
    commits row 0's own argmax, so row 0 decides the served text; the draft row only ever supplies
    a bonus token. A single pooled number could not tell the round which half it has.

    ⛔ A BUDGET, not a boolean -- the sixth of that shape. The reference is a python-loop recurrence
    over every token of every request, so an unbudgeted call would land inside the decode figure
    bullet 9 exists to produce.
    """
    if int(local_layer) not in gdncheck_layers(model):
        return None

    # ⛔⛆ #801 r6 b9aa: THE BUDGET COUNTS SPECULATING STEPS, NOT LAYER-CALLS.
    #   `GDNCHECK=4` has meant "four speculating steps" since 9x, and every bank says so. With a
    #   layer SET a per-call spend would turn a four-layer set into ONE step measured four ways,
    #   and the reports would look exactly like four steps -- bullet 9o's mistake with a different
    #   denominator. ⭐ Charging the step once also means the set can be widened without
    #   re-reading what any existing bank's N meant.
    step = int(step)
    if getattr(model, "_gdncheck_step", None) != step:
        left = getattr(model, "_gdncheck_left", 0) or 0
        if left <= 0:
            return None
        model._gdncheck_left = left - 1
        model._gdncheck_step = step

    from .gdn_reference import recurrent_gated_delta_rule

    # ⛔⛆ #801 r6 b9z: THE RANK CONTRACT, STATED ONCE, WHERE THE SPEND IS.
    #   Load 11 died inside warmup because `gdn.py` handed this ``o[0][0]`` -- one strip too
    #   many, because the fla kernel SQUEEZES its ``NK`` axis away on its last line and 9x read
    #   only the ``q.new_empty(NK, *v.shape)`` allocation. The subtraction below then right-
    #   aligned ``[Hv, Dv]`` against ``[T, Hv, Dv]`` and raised a bare broadcast error 20 lines
    #   from the mistake, with no name on it.
    #   ⭐ Checked against ``q``, which this function already trusts for the per-request slices
    #   and which the CALL SITE has already GQA-expanded to the value-head count -- so a fixture
    #   cannot satisfy this by agreeing with itself. ⛔ It must RAISE, not clamp or reshape: a
    #   silent fix-up here would have turned load 11's loud failure into a quiet wrong number,
    #   which is this round's most expensive recurring shape.
    #   ⛔⛆ #801 r6 b9ac: THE LAST AXIS IS **Dv**, AND IT IS CHECKED AGAINST ``v``, NOT ``q``.
    #   The model has ``head_k_dim == head_v_dim == 128`` and every fixture in this round built a
    #   SQUARE geometry, so a Dk-wide return and a correct one were the same shape to this guard,
    #   to the hardware and to all 73 gates. ⇒ the value geometry is checked against the tensor
    #   that carries it. (The same conflation was live in the ``t1_out`` guard below, where it
    #   REJECTED the correct shape the moment the two dims differed.)
    if (
        fused_out.dim() != 3
        or fused_out.shape[:2] != q.shape[:2]
        or fused_out.shape[2] != v.shape[2]
    ):
        raise ValueError(
            "#801 gdncheck: fused_out must be the kernel's flat per-token output "
            f"[total, Hv, Dv] == {tuple(q.shape[:2]) + (v.shape[2],)}, "
            f"got {tuple(fused_out.shape)}. "
            "The fla kernel squeezes NK before returning, so the verify branch owes this "
            "`o[0]`, never `o[0][0]`."
        )

    # ⛔⛆ #801 r6 b9ab: THE CONTROL'S SHAPE, STATED WHERE THE SPEND IS -- 9z's rule, second use.
    #   `t1_out` is the SAME kernel re-run at T=1 on each request's first token, so it carries one
    #   row PER REQUEST and the same [Hv, Dv] geometry `q` does. Checked against ``q``, which this
    #   function already trusts, so a fixture cannot satisfy it by agreeing with itself. ⛔ It must
    #   RAISE: a silent reshape here would turn the one comparison that has no oracle in it into a
    #   quiet wrong number, which is this round's most expensive recurring shape.
    #   ⛔⛆ **9ac CORRECTS THE TENSOR IT IS CHECKED AGAINST.** ``[Hv, Dv]`` was compared with
    #   ``q.shape[1:]``, which is ``[Hv, Dk]``: on a square model those are the same tuple, so the
    #   guard both accepted a Dk-wide control AND would have rejected the correct one on any
    #   geometry where they differ. ``v`` carries the value geometry and is equally not derived
    #   from ``t1_out``, so the 9z property (a fixture cannot satisfy it by agreeing with itself)
    #   is unchanged.
    if t1_out is not None and (
        t1_out.dim() != 3
        or t1_out.shape[0] != cu_seqlens.numel() - 1
        or t1_out.shape[1:] != v.shape[1:]
    ):
        raise ValueError(
            "#801 gdncheck: t1_out must be the T=1 control's per-request output "
            f"[bs, Hv, Dv] == ({cu_seqlens.numel() - 1},) + {tuple(v.shape[1:])}, "
            f"got {tuple(t1_out.shape)}. The fla wrapper squeezes NK before returning, so the "
            "control owes this `o[0]`, never `o[0][0]`."
        )

    bounds = [int(x) for x in cu_seqlens.tolist()]
    requests, worst, row0_worst, row0_ref = [], 0.0, 0.0, 0.0
    row0_t1_worst, row0_t1_ref, row0_self_worst, row0_self_ref = 0.0, 0.0, 0.0, 0.0
    # ⭐⭐⭐ #801 r6 b9ac: THE ENTRY STATE IS READ BOTH WAYS ROUND, AND NEITHER READING IS ASSUMED.
    #   Load 14 showed the oracle cannot read CLEAN even against the SERVED decode call (the same
    #   kernel at T=1: 0 of 288 cells inside tolerance), so loads 11-13 measured the instrument.
    #   The kernel's own pointer arithmetic names the cause --
    #   `kernel/fla/fused_sigmoid_gating_recurrent.py:106-110` addresses element (k, v) at
    #   ``v*K + k``, so ``k`` is the FASTEST axis and the buffer is ``[V, K]`` to the kernel; it
    #   stores back through the identical expression, so the served path round-trips and is fine.
    #   `gdn_reference.recurrent_gated_delta_rule` contracts ``dim=-2`` against ``k`` and so needs
    #   ``[Dk, Dv]`` ⇒ this oracle has handed the reference a TRANSPOSED state on every load.
    #   ⛔⛆ **DERIVED FROM SOURCE, NEVER MEASURED** -- which is why the twin columns are EMITTED
    #   BESIDE the originals rather than the call being swapped: a swap bets the load on the
    #   hypothesis and silently changes what every existing bank's `row0_rel` means.
    #   ⛔ Only when ``K == V``. A non-square state has exactly ONE shape-legal orientation (the
    #   reference broadcasts ``state * k_t.unsqueeze(-1)``), and it is the plain one -- and that
    #   squareness is precisely what made this invisible for four loads.
    square = entry_state.shape[-1] == entry_state.shape[-2]
    worst_t, row0_worst_t, row0_ref_t = 0.0, 0.0, 0.0
    row0_t1_worst_t, row0_t1_ref_t = 0.0, 0.0
    for req in range(len(bounds) - 1):
        lo, hi = bounds[req], bounds[req + 1]
        if hi <= lo:  # a padded row forwards no token; it has no reference to compare against
            continue
        ref, _ = recurrent_gated_delta_rule(
            q[lo:hi].unsqueeze(0), k[lo:hi].unsqueeze(0), v[lo:hi].unsqueeze(0),
            g[lo:hi].unsqueeze(0), beta[lo:hi].unsqueeze(0),
            initial_state=entry_state[req : req + 1],
        )
        ref0 = ref[0].float()
        diff = (fused_out[lo:hi].float() - ref0).abs()
        per_token = [float(diff[t].max()) for t in range(hi - lo)]
        # ⛔⛆ #801 r6 b9aa: THE SCALE TRAVELS WITH THE NUMBERS IT JUDGES -- load 12 is why.
        #   That load measured row 0 at 0.1305 and row 1 at 0.1281 and could say the RATIO was
        #   1.008 (scale-free, and that was the result) but could NOT say whether either was
        #   clean: a bf16 kernel against an fp32 reference is ordinary rounding at 0.13 if the
        #   outputs run to ~30, and a 25 % error if they run to ~0.5. ⭐ Same principle as `atol`
        #   in 9y: derived in the analyser it would apply THIS reader's reference to THAT load's
        #   figures, so it is emitted beside the difference it scales.
        per_token_ref = [float(ref0[t].abs().max()) for t in range(hi - lo)]
        worst = max(worst, *per_token) if per_token else worst
        row0_worst = max(row0_worst, per_token[0])
        row0_ref = max(row0_ref, per_token_ref[0])
        entry = {"req": req, "tokens": hi - lo, "per_token": per_token,
                 "per_token_ref": per_token_ref}
        # ⭐⭐ the SAME reference, the SAME inputs, the entry state read the other way round.
        if square:
            ref_t, _ = recurrent_gated_delta_rule(
                q[lo:hi].unsqueeze(0), k[lo:hi].unsqueeze(0), v[lo:hi].unsqueeze(0),
                g[lo:hi].unsqueeze(0), beta[lo:hi].unsqueeze(0),
                initial_state=entry_state[req: req + 1].transpose(-1, -2),
            )
            ref0_t = ref_t[0].float()
            diff_t = (fused_out[lo:hi].float() - ref0_t).abs()
            per_token_t = [float(diff_t[t].max()) for t in range(hi - lo)]
            per_token_ref_t = [float(ref0_t[t].abs().max()) for t in range(hi - lo)]
            worst_t = max(worst_t, *per_token_t)
            row0_worst_t = max(row0_worst_t, per_token_t[0])
            row0_ref_t = max(row0_ref_t, per_token_ref_t[0])
            entry["per_token_t"] = per_token_t
            entry["per_token_ref_t"] = per_token_ref_t
        else:
            entry["per_token_t"] = None
            entry["per_token_ref_t"] = None
        if t1_out is not None:
            # ⭐⭐ #801 r6 b9ab: THE SAME REFERENCE, ON ONE TOKEN, against the same kernel re-run
            #   at T=1 -- the known-good decode call. This is the arm that says whether the ORACLE
            #   can read clean at all, which nothing in loads 11-13 established.
            ref1, _ = recurrent_gated_delta_rule(
                q[lo:lo + 1].unsqueeze(0), k[lo:lo + 1].unsqueeze(0),
                v[lo:lo + 1].unsqueeze(0), g[lo:lo + 1].unsqueeze(0),
                beta[lo:lo + 1].unsqueeze(0),
                initial_state=entry_state[req: req + 1],
            )
            r1 = ref1[0][0].float()
            k1 = t1_out[req].float()
            entry["t1_worst"] = float((k1 - r1).abs().max())
            entry["t1_ref"] = float(r1.abs().max())
            # ⭐⭐⭐ NO ORACLE IN THIS ONE. The rule is CAUSAL, so token 0's output cannot depend
            #   on token 1: the T=2 call's first row and the T=1 call's row are the same kernel
            #   computing the same thing. A disagreement here is a statement about the hardware
            #   that survives whatever turns out to be true of `gdn_reference`.
            entry["self_worst"] = float((fused_out[lo].float() - k1).abs().max())
            entry["self_ref"] = float(k1.abs().max())
            row0_t1_worst = max(row0_t1_worst, entry["t1_worst"])
            row0_t1_ref = max(row0_t1_ref, entry["t1_ref"])
            row0_self_worst = max(row0_self_worst, entry["self_worst"])
            row0_self_ref = max(row0_self_ref, entry["self_ref"])
            # ⭐⭐ the control's own twin. ⛔ `self_*` gets NONE: it compares the kernel with
            #   itself, so there is no reference in it for an orientation to be wrong about and a
            #   twin would be a byte-identical copy -- a column that cannot differ cannot be read.
            if square:
                ref1_t, _ = recurrent_gated_delta_rule(
                    q[lo:lo + 1].unsqueeze(0), k[lo:lo + 1].unsqueeze(0),
                    v[lo:lo + 1].unsqueeze(0), g[lo:lo + 1].unsqueeze(0),
                    beta[lo:lo + 1].unsqueeze(0),
                    initial_state=entry_state[req: req + 1].transpose(-1, -2),
                )
                r1t = ref1_t[0][0].float()
                entry["t1_worst_t"] = float((k1 - r1t).abs().max())
                entry["t1_ref_t"] = float(r1t.abs().max())
                row0_t1_worst_t = max(row0_t1_worst_t, entry["t1_worst_t"])
                row0_t1_ref_t = max(row0_t1_ref_t, entry["t1_ref_t"])
        requests.append(entry)
    return {
        "step": int(step),
        "layer": int(layer_id),
        # ⛔ BOTH numbers, because the round has only ever seen them equal: `layer` is the GLOBAL
        #   id a log line names, `local_layer` the GDN-local index the state pool is keyed by and
        #   the one the layer SET selects on. The first GDN layer is global layer 0 too, so every
        #   report before 9aa read `"layer": 0` for both and the distinction never showed.
        "local_layer": int(local_layer),
        # ⛔⛆ #801 r6 b9aa: THE RANK, BECAUSE THE TWO RANKS MEASURE DIFFERENT HEAD SHARDS.
        #   `speccheck` carries this so `per_event` can IGNORE it and make a disagreement loud --
        #   the two ranks there check the same thing. Here they do not: at TP=2 each rank owns its
        #   own heads, so its |Δ| is its own measurement and load 12 proves they differ (0.0962
        #   vs 0.1313 on step 4). Without this field the analyser collapses them as rank copies,
        #   calls every step divergent, and keeps ONE rank's numbers.
        "tp_rank": getattr(model, "_speccheck_tp_rank", None),
        "requests": requests,
        "worst": worst,
        "row0_worst": row0_worst,
        "row0_ref": row0_ref,
        "row0_rel": (row0_worst / row0_ref) if row0_ref > 0 else None,
        # ⭐⭐ #801 r6 b9ab: THE TWO COLUMNS THAT SAY WHETHER THE INSTRUMENT CAN BE BELIEVED.
        #   `row0_t1_rel` is the same reference against the SAME kernel re-run at T=1 -- the
        #   served decode call -- so it is the oracle's own control: ~1e-2 means it can read
        #   clean and the T=2 return is the defect, ~1.0 means the marshalling is wrong and
        #   loads 11/12/13 measured the instrument.
        #   `row0_self_rel` has NO oracle in it: a causal rule cannot let token 1 change token
        #   0's output, so the T=2 first row and the T=1 row are the same kernel computing the
        #   same thing. ⛔ `None` when the control did not run -- `is None`, never truthiness,
        #   because 0.0 is the reading this column exists to be able to report.
        "row0_t1_worst": row0_t1_worst if t1_out is not None else None,
        "row0_t1_ref": row0_t1_ref if t1_out is not None else None,
        "row0_t1_rel": (row0_t1_worst / row0_t1_ref) if row0_t1_ref > 0 else None,
        "row0_self_worst": row0_self_worst if t1_out is not None else None,
        "row0_self_ref": row0_self_ref if t1_out is not None else None,
        "row0_self_rel": (row0_self_worst / row0_self_ref) if row0_self_ref > 0 else None,
        # ⭐⭐⭐ #801 r6 b9ac: THE SAME TWO COLUMNS, WITH THE ENTRY STATE READ ``[V, K]``.
        #   The reading is fixed before the load: ``row0_t1_rel_t`` ≈ 1e-2 CONFIRMS the transpose
        #   -- the oracle works, loads 11-13's question re-opens with a sound instrument, and
        #   `row0_self_rel` becomes readable. ``row0_t1_rel_t`` ≈ 1.4, like its un-transposed
        #   twin, REFUTES it and makes the T=1 control call itself the suspect.
        #   ⛔ `None` when the state is not square, never 0.0 -- there is no second orientation to
        #   report, and 0.0 is this column's best possible reading.
        "worst_t": worst_t if square else None,
        "row0_worst_t": row0_worst_t if square else None,
        "row0_ref_t": row0_ref_t if square else None,
        "row0_rel_t": (row0_worst_t / row0_ref_t) if (square and row0_ref_t > 0) else None,
        "row0_t1_worst_t": row0_t1_worst_t if (square and t1_out is not None) else None,
        "row0_t1_ref_t": row0_t1_ref_t if (square and t1_out is not None) else None,
        "row0_t1_rel_t": (
            (row0_t1_worst_t / row0_t1_ref_t)
            if (square and t1_out is not None and row0_t1_ref_t > 0)
            else None
        ),
        # ⛔ THE TOLERANCE TRAVELS WITH THE NUMBERS (#801 r6 b9y). `analyse_verify_801.py` has
        #   to say whether ROW 0 sat inside it, and `agree` is pooled over every row so it cannot.
        #   A constant in the analyser would be a second source for `GDNCHECK_ATOL` -- and it
        #   would apply the reader's value to the load's numbers, so an old bank would change
        #   meaning the day the tolerance is recalibrated.
        "atol": float(atol),
        "agree": worst <= atol,
    }


#: ⭐⭐⭐ #801 r6 b9ca: which integer width a float's BITS are read as. Keyed by the dtype's NAME,
#: not by a `torch.dtype` object, and that is this module's own rule rather than a style choice:
#: bullet 2 moved `import torch` out of this file's top so the host suite (python 3.14, numpy and
#: pytest, NO torch) can import it, and a dict literal keyed `torch.float32: torch.int32` would
#: drag it back in at import time and take `test_verify_801.py` out of collection.
#: ⛔⛆ **A FLOAT DTYPE MISSING HERE RAISES** -- see :func:`ft801_hidden_fingerprint`. 9bw's copy
#: falls through to a hash BY VALUE instead and names that in a comment as the one thing the
#: instrument may not do; this one makes it a stop, because a value hash reports two arms holding
#: DIFFERENT bits as agreeing, which is a vacuous pass that reads as a finding (#866's shape).
_FT801_HIDDEN_BITS = {
    "torch.float64": "int64",
    "torch.float32": "int32",
    "torch.bfloat16": "int16",
    "torch.float16": "int16",
}


#: ⭐⭐⭐ #801 r6 b9cb: the SEVEN tensors of a decoder layer's block, in the order layer 0 writes
#: them. ⛔⛆ **THE ORDER IS THE VERDICT**, not a label: `mixer_out` is CANDIDATE A (the GDN mixer's
#: output row at T=2) and `mlp_out` is CANDIDATE B (the MoE/MLP at two rows), and the reader names
#: whichever differs FIRST while everything before it agrees. ⚠ `in` differing at all contradicts
#: load 30 -- it says the residual was carried INTO the block rather than born in it -- so 9cd
#: makes that a hard stop rather than a verdict.
#: ⭐ Here rather than in `model.py` so `analyse_hiddencheck_801.py` can import the order on the
#: HOST, where there is no torch: this module is importable without it and `model.py` is not.
FT801_HIDDENCHECK_NAMES = (
    "in",          # the hidden as it ARRIVES at the layer
    "post_ple",    # after the PLE add (== "in" on a layer with no PLE)
    "mixer_in",    # after attn_hyper_connection.mix
    "mixer_out",   # ⭐ CANDIDATE A
    "attn_resid",  # after attn_hyper_connection.combine
    "mlp_out",     # ⭐ CANDIDATE B
    "block_out",   # what the layer hands to the next one
)

#: ⭐⭐⭐ #801 #983: the decoder layers HIDDENCHECK arms, in the order their fingerprints are
#: written into the buffer -- layer `FT801_HIDDENCHECK_LAYERS[i]` owns slots
#: `i*len(NAMES) .. (i+1)*len(NAMES)`. ⛔⛆ **STACK ORDER IS LOAD-BEARING**: the verdict is
#: POSITIONAL, so the reader walks the layers in this order and, within a layer, the names above.
#: A list out of stack order would report a LATER stage as the birthplace of a divergence an
#: EARLIER one already carries -- 9cd's inverted verdict, one level out.
#:
#: ⭐⭐ **WHY LAYER 1 JOINED, AND IT IS THE WHOLE OF #983.** Through #981 the round had measured
#: the M dial three times and no two measurements overlapped: CHECKROW ran it INTRA-arm on the
#: WHOLE forward with SERVED data (1/1,200), #980 ran it on the gap's SIX STAGES with
#: `torch.randn` (0/512), and HIDDENCHECK covered LAYER 0 only, CROSS-arm. ⇒ nothing anywhere
#: measured layer 1 on served data, which is exactly where STATECHECK says the arms' committed
#: state first differs. Arming layer 1 lands the seven existing fingerprints either side of the
#: gap, on the row the engine actually runs.
#:
#: ⚠ `reserve_hiddencheck`'s old docstring said "LAYER 0 ALONE ... instrumenting 48 layers would
#: multiply the emit by 48". That reasoning is intact and this is not a repeal of it: the emit is
#: multiplied by the LENGTH OF THIS TUPLE, and it is 2.
FT801_HIDDENCHECK_LAYERS = (0, 1)


def ft801_hidden_fingerprint(tensor: "torch.Tensor") -> "torch.Tensor":
    """One exact integer per ROW of an activation, as a DEVICE tensor. #801 r6 b9ca.

    ``tensor`` is any ``[T, *rest]`` activation; the result is ``[T]`` int64, one hash per row,
    with the trailing dimensions folded into the row. Returns a tensor on ``tensor``'s own device
    and performs **no host transfer** -- see below.

    ⭐⭐⭐ **WHY THIS EXISTS.** Load 30 localised the two arms' first divergence to INSIDE layer 0's
    block on the forward that commits 11,470 -> 11,471: layer 0's own committed GDN state is
    bit-identical on both arms while layer 1's differs, so layer 0 received identical input, wrote
    identical state, and handed layer 1 something different. The two candidates -- the GDN mixer's
    OUTPUT row at T=2, and the MoE/MLP at two rows -- are hidden states, and
    `Engine._ft801_state_fingerprint` can only reduce a `LinearStatePool`. This reduces the
    activation itself.

    ⛔⛆ **NO HOST TRANSFER, AND IT IS THE PROPERTY THE WHOLE MEASUREMENT RESTS ON.** 9cb arms this
    at BUILD time, so layer 0's write is captured into the CUDA graph and REPLAYS -- which is what
    keeps the instrumented path the CAPTURED one load 30 measured. GDNCHECK cannot do that: it has
    to make `can_use_cuda_graph` decline, because `GraphRunner.replay` never runs the model's
    python (9q, three loads), so a load that sets it measures the EAGER path. A ``.tolist()`` here
    would be a host read inside a capturing stream -- the class of call that killed this round's
    first box load -- so the sync stays in `engine.py`, outside the graph, where STATECHECK's
    already is. `test_verify_wiring_801.py` gates that at the source, not just at the return type.

    ⛔⛆ **THE CURRENCY IS BITS, NOT CLOSENESS**, exactly as in 9bw: the reduction asserts nothing
    about the numbers and reports only whether two arms hold the SAME BITS. ``-0.0`` differs from
    ``0.0`` and a NaN agrees with itself, both deliberately, and both gated so nobody "fixes" this
    into a float compare -- a tolerance here would hide precisely the accumulation being hunted.

    ⛔ **NOT A SUM**, for 9bw's reason: a plain sum is blind to a PERMUTATION, which is the shape a
    chunked verify path against a per-token decode path can take. The second, position-weighted
    moment is what makes an element's POSITION part of the hash. Both products wrap in int64 --
    two's complement, deterministic, a hash rather than an arithmetic claim.

    ⛔ **THE ROW IS THE UNIT, NEVER THE TENSOR.** The arms forward different row counts at the same
    position -- T=1 on the control arm, T=2 on the verify arm -- so a reduction over the whole
    tensor would differ on every position of a clean load. Only row 0 is comparable across arms
    anyway (9bt: the bonus row has no T=1 counterpart anywhere on the arm).

    ⚠ **A DELIBERATE DUPLICATE of `Engine._ft801_state_fingerprint`'s math, kept as an invariant
    rather than a refactor.** 9bw's lives on `Engine` because `test_load_801.py` forbids the engine
    overlay any top-level name, and a layer cannot reach `Engine` without the model importing the
    engine or the reverse. Merging them means editing a file every served row runs, for no
    measurement gain. ⇒ the two are gated to produce the SAME integer for the same bits.
    """
    import torch

    flat = tensor.reshape(tensor.shape[0], -1)
    if flat.dtype.is_floating_point:
        width = _FT801_HIDDEN_BITS.get(str(flat.dtype))
        if width is None:
            raise TypeError(
                f"[#801] ft801_hidden_fingerprint: no bit view for floating dtype {flat.dtype}. "
                "Hashing a float BY VALUE would report two arms holding different bits as "
                "agreeing; add the width to _FT801_HIDDEN_BITS rather than falling through."
            )
        # ⛔⛆ **NO `contiguous()` HERE, AND A SURVIVING MUTATION IS WHY.** 9ca's first draft
        #   called it, reasoning that `.view(dtype)` raises on a strided slice -- which is FALSE
        #   for a SAME-WIDTH view, and every entry in the table above is same-width by a gate
        #   (`test_every_bit_view_matches_its_dtypes_element_size`). Measured in the image: a
        #   `[4, 6]` at stride `(12, 2)` -- a non-unit stride in the LAST dimension, the harshest
        #   layout a block can hand -- views to `int32` and to `int16` without complaint. ⇒ the
        #   call was dead code that a mutation could delete with every gate still green, and a
        #   copy of the activation is not free on the decode path. It is the ELEMENT SIZE
        #   CHANGING that requires contiguity, and the width gate is what forbids that.
        flat = flat.view(getattr(torch, width))
    bits = flat.to(torch.int64)
    weight = torch.arange(1, bits.shape[1] + 1, dtype=torch.int64, device=bits.device)
    return bits.sum(-1) * 0x100000001B3 + (bits * weight).sum(-1)


#: ⭐⭐ #801 r6 b9x: where bullet 9w's oracle finds the budget it spends.
#:
#: ⛔⛆ **`gdn.py` HAS THE TENSORS AND NO ROUTE TO THE MODEL.** A GDN mixer is handed a pool and a
#: batch and nothing else, and `q`/`k`/`v`/`g`/`beta` exist nowhere but inside its forward -- 9w
#: established that `spec.verify_step`'s snapshots do not carry them. ⇒ the same carrier the plan
#: itself takes: an attribute on the `Batch`, so this round still adds no `core.py` overlay.
#:
#: ⭐⭐ **AND IT IS WHAT MAKES THE CAPTURE PASS SAFE.** `engine/graph.py::_capture_pass` forwards
#: its own `Batch(reqs=[dummy_req] * bs)` down this very branch while a CUDA stream is capturing,
#: and :func:`gdncheck_layer` opens with a host read (``cu_seqlens.tolist()``) -- the class of call
#: that killed this round's first box load. :func:`stage_verify` is the only writer and it never
#: sees that batch, so the oracle declines there without a unit being spent.
GDNCHECK_OWNER = "ft801_gdncheck_owner"


def gdncheck_layers(owner) -> frozenset:
    """Which LOCAL GDN layers the oracle is spent on. Default: the first alone. #801 r6 b9aa.

    ⭐⭐ **LOAD 12 IS WHY THIS IS A SET.** 9w chose the first local layer deliberately -- *"a
    layer-0 disagreement is the kernel or its arguments; a clean layer 0 under a garbage logit row
    is something that grows on the way up"*. Layer 0 has now answered: row 0 is wrong by the same
    margin as row 1 there (median ratio **1.008**), so 9t's row-0-specific defect is introduced
    further up, and the live question is WHERE. One load over several layers localises it.

    ⛔ **LOCAL indices**, never the global `layer_id`: `li = pool.local_index(self.layer_id)` is
    what the state pool is keyed by and what "first local layer" has always meant. They coincide
    at 0, which is the only value the round has used, so nothing has ever exercised the
    difference.

    ⛔ The default is the first local layer ALONE, so every bank written before 9aa keeps its
    meaning and a load that names no set measures exactly what loads 11 and 12 measured.
    """
    want = getattr(owner, "_gdncheck_layers", None)
    return frozenset(want) if want else frozenset({0})


def gdncheck_wants_layer(batch: "Batch", local_layer: int) -> bool:
    """Could the oracle spend a unit on THIS layer of this batch? The CHEAP question. #801 r6 b9aa.

    ⛔⛆ **ONE QUESTION, ASKED BEFORE ANY TENSOR WORK** -- 9x's capture-pass rule, unchanged and
    now carrying the layer test too. `gdn.py` used to ask ``li == 0 and gdncheck_instrumented(..)``;
    with a set the layer test needs the owner, and splitting it across the call site would put
    half the condition where the capture pass can reach the other half.
    """
    if not gdncheck_instrumented(batch):
        return False
    return int(local_layer) in gdncheck_layers(getattr(batch, GDNCHECK_OWNER, None))


def gdncheck_instrumented(batch: "Batch") -> bool:
    """Could the oracle spend a unit on this batch? The CHEAP question. #801 r6 b9x.

    ⛔⛆ **IT IS ASKED BEFORE ANY TENSOR WORK, AND THE CAPTURE PASS IS WHY.**
    `engine/graph.py::_capture_pass` forwards its own `Batch(reqs=[dummy_req] * bs)` down the
    verify branch INSIDE ``torch.cuda.graph(...)``. Everything `gdn.py` would build for the oracle
    -- `_gate_params`, the GQA `repeat_interleave`, the ``rec`` gather -- is device work, so it
    would not fault; it would be BAKED INTO THE GRAPH and paid on every replay for the life of the
    load, for a report nothing ever reads. An instrument that costs the served row something on
    the steps it does not measure is the shape this round keeps banking.

    ⛔ NOT an arm, and deliberately not named like one: :func:`speccheck_armed` and
    :func:`checkrow_armed` SPEND a unit when they answer True. This only peeks -- the spend stays
    in :func:`gdncheck_layer`, in one place, so a call site cannot drain the budget and skip the
    comparison.
    """
    owner = getattr(batch, GDNCHECK_OWNER, None)
    if owner is None:
        return False
    if int(getattr(owner, "_gdncheck_left", 0) or 0) > 0:
        return True
    # ⛔⛆ #801 r6 b9aa: A STEP ALREADY CHARGED IS STILL IN FLIGHT, AND ITS REMAINING LAYERS
    #   MUST BE REACHED. With a layer set the budget can fall to zero on the FIRST layer of a step
    #   whose later layers have not run yet. Declining here would report layer 0 and silently drop
    #   layers 12 and 24 OF THE SAME STEP -- and the bank would read as though those layers had
    #   been measured and were clean, which is this round's most expensive shape.
    #   ⭐ The step number is derived exactly as `gdncheck_batch` derives it, from the same
    #   counter, so the two cannot drift apart.
    return getattr(owner, "_gdncheck_step", None) == int(
        getattr(owner, "_verify_steps", 0) or 0
    ) + 1


def gdncheck_batch(batch: "Batch", **fields) -> dict | None:
    """Spend one unit of the GDN-output oracle on this step, report it, or decline. #801 r6 b9x.

    ⭐ ONE call from `gdn.py`'s verify branch, for the reason bullet 7 gave for
    :func:`speccheck_step` and bullet 9l repeated for :func:`checkrow_step`: a call site that
    armed the budget and skipped the comparison, or compared and dropped the report, would look
    correct from the helper's side.

    ⛔⛆ **THE STEP NUMBER IS ``_verify_steps + 1``, AND THAT IS NOT AN OFF-BY-ONE.**
    `model.mtp_verify_step` increments the counter, and it runs AFTER the forward -- so while this
    runs the counter still holds the PREVIOUS step's count. Reporting it raw would label every
    line with the step before its own and the analyser would pair it with the wrong `checkrow`.
    """
    owner = getattr(batch, GDNCHECK_OWNER, None)
    if owner is None:
        return None
    report = gdncheck_layer(
        owner, step=int(getattr(owner, "_verify_steps", 0) or 0) + 1, **fields
    )
    if report is None:
        return None
    print(f"[#801] gdncheck: {json.dumps(report)}", file=sys.stderr, flush=True)
    return report


# ══ #801 round 6 bullet 9af: ONE LIVE CELL, ON DISK ════════════════════════════════════════════
#
# ⭐⭐⭐ **WHY A DUMP AND NOT A SEVENTH SWEEP.** Bullet 9ae called the served verify kernel directly
#   on random tensors — T=2, the intermediate buffer, ``disable_state_update=True``,
#   ``num_stages=3`` — and every absolute deviation came back at 2⁻¹⁰, the bf16 quantum. The kernel
#   is EXACT to the format and load 15's *"the fla kernel's T>1 path IS wrong"* does not survive
#   it. What is left is the live DATA, the live ENVIRONMENT, or the INSTRUMENT — and six GDNCHECK
#   loads have now measured the same number, so nothing in that family separates them.
#   ⇒ dump ONE cell's tensors and replay them through `repro_gdn_t2_801.py --from`, which splits
#   all three at once. The reading is fixed in that script's `replay_verdict`, BEFORE the load.
#
# ⛔⛆ **IT IS A COUNT, AND THE HARNESS OWNS THE PATH.** `model.py` reads six environment
#   instruments and every one is a COUNT, deciding nothing about what is loaded or run; bullet 9aa
#   deliberately rode the GDN layer SET on an existing count rather than add a seventh read. A
#   ``GDNDUMP=/some/dir`` would be the first DIAL in that set. So ``FREETOKEN_MTP801_GDNDUMP=N``
#   says how many cells, :data:`GDN_DUMP_DIR` says where, and the arm script mounts it — the same
#   shape `headcheck.py` uses, which takes its path as an argument rather than reading the
#   environment for it.
#
# ⛔ **A DUMPED CELL IS ALWAYS A REPORTED CELL.** The dump is spent from the return of
#   :func:`gdncheck_batch`, not from the decision to enter the instrument: a cell whose report the
#   budget declined has no line in the bank, and a replay of one would have nothing to be compared
#   against. Pairing them by construction is what makes "the replay reads clean" mean something.

#: ⛔ NOT an environment read. The arm script mounts a writable host directory here; a load that
#: arms the dump without the mount fails LOUDLY at bring-up rather than serving for 25 minutes and
#: writing nothing (#866's shape: a vacuous pass that looks like a result).
GDN_DUMP_DIR = "/ft801-dump"
GDN_DUMP_VERSION = 1
GDN_DUMP_MANIFEST = "manifest.json"
GDN_DUMP_TENSOR_FILE = "tensors.pt"

#: ⛔⛆ A CAP, NOT A HOPE — and hitting it is a FINDING, not a formatting problem. `torch.save` on a
#: VIEW writes the whole underlying storage, and ``a``/``b`` reach the kernel as `torch.split`
#: views of the projection. At the deployed geometry the whole cell is single-digit MB, so a run
#: that reaches half a gigabyte is telling you one of these tensors is not what this round thinks
#: it is. It raises, on the box, with the offender named.
GDN_DUMP_MAX_BYTES = 512 * 1024 * 1024

#: Every tensor the replay needs, in the two groups the bisect reads them as. ⛔ Kept in step with
#: `repro_gdn_t2_801.DUMP_*` by `test_gdn_verify_801`, which validates a manifest built HERE with
#: the reader's own `validate_manifest` — the two ends are in different interpreters and cannot
#: import each other at runtime, so the gate is the only thing that stops them drifting.
GDN_DUMP_KERNEL_TENSORS = (
    "q", "k", "v", "a", "b", "A_log", "dt_bias", "entry_state", "cu_seqlens", "cache_indices",
)
GDN_DUMP_ORACLE_TENSORS = ("oracle_q", "oracle_k", "oracle_v", "oracle_g", "oracle_beta")
GDN_DUMP_OUTPUT_TENSORS = ("fused_out", "t1_out")
GDN_DUMP_TENSORS = GDN_DUMP_KERNEL_TENSORS + GDN_DUMP_ORACLE_TENSORS + GDN_DUMP_OUTPUT_TENSORS


def gdn_dump_armed(owner) -> bool:
    """Is there a cell left to dump? The CHEAP question, asked before any tensor work.

    ⛔ NOT an arm, and not named like one: it only peeks. The spend is in :func:`gdn_dump_cell`, in
    one place, so a call site cannot drain the budget and skip the write — the same rule
    :func:`gdncheck_wants_layer` keeps against :func:`gdncheck_layer`.
    """
    return int(getattr(owner, "_gdndump_left", 0) or 0) > 0


def gdn_dump_describe(tensor) -> dict:
    """One tensor, as the plain JSON the manifest carries.

    ⭐ **THE STRIDE IS IN HERE BECAUSE IT IS A CANDIDATE.** ``a`` and ``b`` arrive as `torch.split`
    views whose row stride is the projection's width and not ``HV``; ``q``/``k``/``v`` come through
    `_conv_verify`'s transpose+reshape. `torch.save` preserves view metadata so the replay sees the
    same strides — and stating them in plain JSON is what lets the HOST gate the claim with no
    torch and no container.

    ⛔ ``values`` only for small integer tensors. ``cu_seqlens`` and ``cache_indices`` decide how
    the replay slices and what it can and cannot reproduce, so a host reader must be able to see
    them without loading the ``.pt``; a float tensor's contents are the measurement and do not
    belong in a manifest.
    """
    storage = getattr(tensor, "untyped_storage", None)
    nbytes = (
        storage().nbytes() if storage is not None
        else tensor.numel() * tensor.element_size()
    )
    described = {
        "shape": [int(x) for x in tensor.shape],
        "stride": [int(x) for x in tensor.stride()],
        "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
        "storage_bytes": int(nbytes),
        "storage_offset": int(tensor.storage_offset()),
    }
    if not tensor.dtype.is_floating_point and tensor.numel() <= 64:
        described["values"] = [int(x) for x in tensor.flatten().tolist()]
    return described


def gdn_dump_manifest(
    *, layer_id: int, local_layer: int, step: int, tp_rank, call: dict,
    tensors: dict, pool: dict, cu_seqlens: Sequence, report: dict,
) -> dict:
    """The manifest, assembled from values that are already plain python. #801 r6 b9af.

    ⛔⛆ **PURE, AND THAT IS THE POINT.** It touches no tensor — :func:`gdn_dump_describe` has
    already turned each one into a dict — so the HOST suite can build one, hand it to
    `repro_gdn_t2_801.validate_manifest`, and prove the two ends of the contract agree before a
    load is ever spent on it. The two live in different interpreters and neither can import the
    other on the box; a gate that runs both is the only thing that keeps them in step.

    ⛔ ``report`` is embedded WHOLE, not summarised. The replay's whole value is being readable
    beside the numbers the live cell itself produced, and a summary written here would be a second
    source for ratios :func:`gdncheck_layer` already emits — including its ``None``s, which mean
    *"this column could not be read"* and must not become zeros.
    """
    missing = [name for name in GDN_DUMP_TENSORS if name not in tensors]
    if missing:
        raise ValueError(f"#801 gdn dump: no description for {missing}")
    return {
        "version": GDN_DUMP_VERSION,
        "bullet": "9af",
        "layer_id": int(layer_id),
        "local_layer": int(local_layer),
        "step": int(step),
        "tp_rank": None if tp_rank is None else int(tp_rank),
        "geometry": {
            "hk": int(tensors["q"]["shape"][2]),
            "hv": int(tensors["v"]["shape"][2]),
            "dk": int(tensors["q"]["shape"][3]),
            "dv": int(tensors["v"]["shape"][3]),
        },
        "call": dict(call),
        "tensors": {name: dict(tensors[name]) for name in GDN_DUMP_TENSORS},
        "pool": dict(pool),
        "cu_seqlens": [int(x) for x in cu_seqlens],
        "report": report,
    }


def gdn_dump_cell(
    owner, *, tensors, pool, call: dict, layer_id: int, local_layer: int, step: int,
    report: dict, dest: str | None = None,
) -> "str | None":
    """Spend one unit of the dump budget on this cell, or decline. #801 r6 b9af.

    Returns the directory written, or ``None`` when nothing was armed.

    ⛔⛆ **THE RANK COMES FROM THE ENGINE'S REGISTRY AND IS CHECKED AGAINST THE REPORT.** A TP=2 arm
    runs both ranks under ONE ``docker run`` with one environment, so an env-derived rank would
    label both directories ``rank0`` and one would overwrite the other — `headcheck.py` names that
    trap. The report carries its own ``tp_rank`` from a different source, so disagreeing with it is
    a finding and raises rather than picking one.

    ⛔ The write is ONE directory per cell, named by rank, step and layer. Two ranks and several
    steps have to coexist, and a flat file would silently keep whichever wrote last — which is
    exactly how load 14's control nearly went unread.
    """
    import json
    import os

    import torch
    from freetoken.distributed import get_tp_info

    if not gdn_dump_armed(owner):
        return None

    rank = int(get_tp_info().rank)
    reported = report.get("tp_rank")
    if reported is not None and int(reported) != rank:
        raise RuntimeError(
            f"#801 gdn dump: the engine's registry says rank {rank}, the report says {reported}. "
            "Two sources for the rank that names the directory is one too many."
        )

    described = {name: gdn_dump_describe(tensors[name]) for name in GDN_DUMP_TENSORS}
    total = sum(d["storage_bytes"] for d in described.values())
    if total > GDN_DUMP_MAX_BYTES:
        worst = max(described.items(), key=lambda kv: kv[1]["storage_bytes"])
        raise RuntimeError(
            f"#801 gdn dump: {total} bytes of backing storage exceeds {GDN_DUMP_MAX_BYTES}; "
            f"the largest is {worst[0]} at {worst[1]['storage_bytes']} bytes with shape "
            f"{worst[1]['shape']} and stride {worst[1]['stride']}. `torch.save` writes a VIEW's "
            "whole base storage — at the deployed geometry this cell is single-digit MB, so this "
            "is a finding about one of these tensors, not a formatting problem."
        )

    manifest = gdn_dump_manifest(
        layer_id=layer_id, local_layer=local_layer, step=step, tp_rank=rank,
        call=call, tensors=described, pool=pool,
        cu_seqlens=tensors["cu_seqlens"].tolist(), report=report,
    )

    root = dest or GDN_DUMP_DIR
    out = os.path.join(root, f"rank{rank}-step{int(step)}-layer{int(layer_id)}")
    os.makedirs(out, exist_ok=True)
    # ⛔ Tensors FIRST, manifest second. `repro_gdn_t2_801.load_inputs` refuses a manifest whose
    #   `.pt` does not match it; writing the manifest last means a run killed mid-write leaves a
    #   directory the reader declines, never one it silently half-reads.
    torch.save({name: tensors[name] for name in GDN_DUMP_TENSORS},
               os.path.join(out, GDN_DUMP_TENSOR_FILE))
    with open(os.path.join(out, GDN_DUMP_MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    owner._gdndump_left = int(getattr(owner, "_gdndump_left", 0) or 0) - 1
    print(f"[#801] gdndump: wrote {out} ({total} bytes of backing storage), "
          f"{owner._gdndump_left} cell(s) left", file=sys.stderr, flush=True)
    return out


def checkrow_step(
    model: "Qwen4ExpForCausalLM",
    batch: "Batch",
    plan: VerifyPlan,
    logits: "torch.Tensor",
    *,
    linear_pool=None,
    step: int | None = None,
) -> dict | None:
    """Arm, run the reference forward, report. ``None`` when this step was not instrumented.

    ⭐ ONE call from :func:`verify_step`, for the reason bullet 7 gave for :func:`speccheck_step`:
    a call site that armed the budget and skipped the forward, or ran the forward after the
    commits, would look correct from every helper's side.
    """
    if not checkrow_armed(model, plan):
        return None
    reference = reference_forward(model, batch, plan, linear_pool=linear_pool)
    rows = checkrow_rows(plan, logits, reference)
    report = {
        "step": step,
        "n": len(rows),
        "agree_n": sum(1 for row in rows if row["agree"]),
        "rows": rows,
    }
    print(f"[#801] checkrow: {json.dumps(report)}", file=sys.stderr, flush=True)
    return report


# ══ #801 round 6 bullet 8: THE WIRING ═════════════════════════════════════════════════════════
#
# ⭐⭐ Bullets 2-7 built every piece of a verify step and gated each one in isolation. NONE of them
#   was reachable by a served row, because nothing called them. This section is the calls, and it
#   is deliberately ONE place: three call sites (`scheduler._prepare_batch`, `Engine.forward_batch`,
#   `scheduler._process_last_data`) each make exactly one, so neither every-served-row file grows
#   #801 logic and neither can half-wire the step. Round 6 bullet 6's lesson, applied: *a helper
#   that returns the right thing and a call site that ignores it look identical from the helper's
#   side.*
#
# ⛔⛆ **THE HOST SYNC, AND IT IS AN OPERATOR DECISION (2026-09-12), NOT AN OVERSIGHT.**
#   `Engine.forward_batch` advances every request BEFORE the sampler, which is legal only because
#   today's advance is `+1` and needs no token values. A verify step's advance is `accepted_len` --
#   a device tensor off the sampler -- and `scheduler.py::overlap_loop` PREPARES batch N+1 before
#   it DRAINS batch N, so the advance cannot be deferred to the drain. Three options were put to
#   the operator: (a) sync on speculative steps only, (b) move the bookkeeping onto the GPU,
#   (c) run the arm with `ENV.DISABLE_OVERLAP_SCHEDULING`. **(a) was chosen**, on three grounds:
#     - the sync is paid ONLY on steps that actually speculate, so a plain decode batch, a
#       flag-off row and a no-draft row keep today's `complete_one()` loop byte-identically --
#       gated by `test_verify_wiring_801.py::TestThePlainPathIsUntouched`;
#     - round 5's shadow arm ALREADY syncs every decode step (`sampled_ids.tolist()` in
#       :func:`shadow_step`), so round 5's α was measured under a per-step sync and bullets 9/10
#       stay apples-to-apples with it. The arm was never overlap-clean;
#     - #912 priced a comparable lever at ~4.4 % decode, against a projected +20-33 %.
#   ⛔ It is therefore a REAL COST OF SPECULATION, not a research artifact. Bullet 9's bank must
#   carry it as its own line rather than absorb it into the tok/s number.
#
# ⛔ `commit_linear_state` runs BEFORE the sync, on purpose: it reads no host value (it is a
#   gather), so running it first keeps the GPU busy across the sync instead of behind it.

#: Where the staged step rides. ⭐ An ATTRIBUTE on the `Batch`, not a `core.py` field -- the same
#: carrier `gdn.py` already uses for `batch.linear_snapshots`, and for the same reason: this round
#: adds no `core.py` overlay (bullet 2 made that call for `Req`).
VERIFY_STEP = "ft801_verify_step"


@dataclass
class VerifyStep:
    """One staged verify forward: the plan, and what it turned out to commit.

    ⚠ NOT frozen, unlike :class:`VerifyPlan`. The plan is fixed at staging time and crosses five
    consumers that must all read the same tuples; ``committed`` cannot exist until the sampler has
    run, and the drain reads it one scheduler iteration later.
    """

    plan: VerifyPlan
    #: host integers, one per request, filled by :func:`verify_step` after the sync. ⛔ Empty
    #: until then -- a drain that finds it empty is draining a batch whose forward never committed.
    committed: tuple[int, ...] = ()


def verify_step_of(batch: "Batch") -> "VerifyStep | None":
    """The step staged on ``batch``, or ``None``. ⛔ Every call site selects its plain branch with
    an `is None` on this, so a no-draft step costs nothing anywhere."""
    return getattr(batch, VERIFY_STEP, None)


def spec_rows(staged: "VerifyStep") -> "tuple[list[int], list[int]]":
    """Per request of a staged step: (the row it was VERIFIED at, the row it DRAFTS from).

    ⭐ **No new arithmetic.** `VerifyPlan.cu_seqlens` is already the prefix sums of
    ``tokens_per_req``, so request ``i`` starts at ``cu_seqlens[i]`` and the last token it
    actually committed is ``committed[i] - 1`` rows further on. A plain decode request
    (``tokens_per_req == 1``, ``committed == 1``) gives back its own index, twice, which is
    what every pre-8c caller assumed for every request.

    ⛔ The two are DIFFERENT rows on an accepted step, and conflating them is the whole of
    bullet 8c: the held draft is judged at the position it was a prediction FOR (the verified
    row), while the next draft must continue from the last token the step committed (the
    drafting row).

    ⛔⛆ ``committed`` is empty until :func:`verify_step` has run. A staged batch reaching here
    without it is a forward that never committed, not a batch to guess about.
    """
    plan = staged.plan
    assert len(staged.committed) == len(plan.uids), (
        f"#801: {len(plan.uids)} request(s) staged, {len(staged.committed)} committed -- "
        f"the verify step did not run before the draft step"
    )
    verified = [plan.cu_seqlens[i] for i in range(len(plan.uids))]
    drafting = [start + n - 1 for start, n in zip(verified, staged.committed)]
    return verified, drafting


def draft_view(
    batch: "Batch", staged: "VerifyStep | None", draft_rows: list, picked
) -> "Batch":
    """``batch`` as the MTP head's own forward must see it: ONE row per request.

    ⛔⛆ **#801 round 6 bullet 9e -- THE SEVENTH SILENT SITE, and load 4 died on it.** Bullet 8c
    selected the head's ROWS and left the BATCH alone. The head stores its KV at ``batch.out_loc``,
    which on a verify step is one row per TOKEN while the head forwards one per REQUEST::

        qsa_sparse.py:401  store_kv(k, v, batch.out_loc, layer_id)
        store.cu:88  Tensor match failed -- Size mismatch for L(shape#0): expected 1 but got 2

    ⭐⭐ **:func:`shadow_step`'s own docstring names this crash** for the PREFILL shape (round 4
    bullet 7's first load, *"expected 1 but got 53"*) and then states the premise that made decode
    safe: *"a decode batch's `out_loc`/`positions` are ALSO one per request, so the shapes agree."*
    Bullet 5 made that false and nobody re-read the sentence that depended on it.

    ⛔ **THREE fields, not one.** `qsa_sparse.qsa_forward` plus `_plan_index_writes` read exactly
    ``attn_metadata``, ``out_loc`` and ``positions`` off the batch. The store is where it surfaced;
    the other two would have been wrong in silence -- ``positions`` ropes the indexer query and
    drives the ring row, so the head would have drafted from the right hidden state at the wrong
    place on every accepted step.

    ⛔ **A VIEW, never a mutation.** `engine.py` re-enters ``forward_batch(batch)`` with this same
    object and `graph.py` hands out buffers the next replay restages, so the one-row shapes go on a
    shallow copy. ⭐ And ``VERIFY_STEP`` comes off it: the head runs a plain one-token decode, so
    the batch it is handed must say so rather than carry a plan nothing on that path can apply.

    ⛔⛆ **RETURNED UNCHANGED when nothing is staged** -- ``is``-identical, the discriminator every
    other call site in this module uses. Bullet 9d's surviving mutation is the precedent: a shallow
    copy compares equal and shares every ``data_ptr``, so only object identity catches one extra op
    landing on the decode path this box actually serves.

    ⚠ **The eager path is the only one that reaches here.** A verify step is already ineligible for
    the captured draft graph (``staged is None`` is in :func:`shadow_step`'s capture test), so the
    copy is never on the captured arm's cost.
    """
    if staged is None:
        return batch
    import copy

    from freetoken.core import get_global_ctx

    view = copy.copy(batch)
    view.out_loc = batch.out_loc[picked]
    view.positions = batch.positions[picked]
    # ⛔⛆ ``position + 1`` per DRAFTING row, and it is the operator's scope decision for this
    #   bullet (2026-09-12), not a shape fix. The backbone forwarded ``device_len``; on a REJECTED
    #   step only ``cached_len + 1`` of that is real AND the head's own KV layer was never written
    #   at the rejected slot -- the head writes one row per step, at the position it drafts from.
    #   `QSASparseMetadata.seq_lens` feeds the visible-block count, so leaving it at the backbone's
    #   length lets that slot compete for the head's top-k budget holding whatever the pool last
    #   left there. Nothing raises; α just reads low. ⭐ On an ACCEPTED step, and on every plain
    #   decode step, this is ``device_len`` exactly -- which is why it is not a blanket ``- 1``.
    view.attn_metadata = get_global_ctx().attn_backend.draft_metadata(
        batch.attn_metadata, [staged.plan.positions[row] + 1 for row in draft_rows]
    )
    if hasattr(view, VERIFY_STEP):
        delattr(view, VERIFY_STEP)
    return view


def publish_drafts(model: "Qwen4ExpForCausalLM", ids: dict, logits: dict | None) -> None:
    """Hold this step's drafts for the NEXT one to verify.

    ⭐ Keyed by ``Req.uid`` and REPLACED wholesale each step, the convention
    `_pending_draft_logits` already uses: a request absent from this batch has no draft at the
    next one, so a membership change can never hand request A's draft to request B.
    """
    model._pending_draft_ids = dict(ids)
    if logits is not None:
        model._pending_draft_logits = logits


def stage_verify(
    model: "Qwen4ExpForCausalLM", batch: "Batch", token_pool: "torch.Tensor"
) -> "VerifyPlan | None":
    """Lay this decode batch out as a verify step, or decline. Called from `_prepare_batch`.

    Three things happen here and nowhere else: the plan is built from the drafts held since the
    previous step, each request's ``device_len`` is advanced to what the plan says, and each draft
    token is written into ``token_pool`` at the position `_make_input_tuple` will read it from.

    ⛔⛆ **`device_len` IS the verify shape.** `_make_positions`, `attention/linear.py::
    build_fla_metadata`, `engine/graph.py::_uniform_width` and `attention/qsa_sparse.py` all derive
    T from `Req.extend_len`. A plan built but not applied is a T=1 forward that believes it
    verified -- and every one of those five files would agree with it.

    ⛔ BEFORE `pad_batch`, which is the first thing `_prepare_batch` does: `pad_batch` asks
    `can_use_cuda_graph`, which reads `extend_len` to decide whether this is a verify step at all.

    ⛔ ``None`` when there is nothing to verify -- not an all-ones plan. `plan_verify` is
    READ-ONLY and this function is the one place a request is advanced, so a declined step leaves
    the batch exactly as the engine built it.
    """
    if not model._verify_run or not batch.is_decode:
        return None
    drafts = getattr(model, "_pending_draft_ids", None)
    if not drafts:
        return None
    plan = plan_verify(batch.reqs, drafts)
    if not plan.is_speculative:
        return None
    for req, device_len in zip(batch.reqs, plan.device_lens):
        req.device_len = device_len
    for req in batch.reqs:
        # ⛔ #801 bullet 9j: cleared on EVERY request first. A request that drafted last step and
        #   decodes plainly this one must not hand its stale draft to this step's commit.
        setattr(req, STAGED_DRAFT, None)
    for index, position, token_id in plan.draft_writes:
        # ⛔ `table_idx`, not the request's index: `token_pool` is keyed by the page-table row the
        #   request owns, which is what `_make_input_tuple` pairs with the position.
        token_pool[batch.reqs[index].table_idx, position] = token_id
        # ⭐ The same (position, id) the host will be missing one step from now, kept where
        #   `commit_verify` can promote it if this draft is ACCEPTED. See `host_token_id`.
        setattr(batch.reqs[index], STAGED_DRAFT, (position, token_id))
    setattr(batch, VERIFY_STEP, VerifyStep(plan=plan))
    # ⭐ #801 r6 b9x: bullet 9w's oracle spends a budget that lives on the MODEL, from a call site
    #   (`gdn.py`'s verify branch) that is handed only a pool and a batch. Same carrier, same
    #   line, and staged HERE so that the only batches carrying it are served speculating steps --
    #   never `engine/graph.py`'s capture batch, where the oracle's host read would be illegal.
    setattr(batch, GDNCHECK_OWNER, model)
    return plan


def verify_step(
    model: "Qwen4ExpForCausalLM",
    batch: "Batch",
    logits: "torch.Tensor",
    args: "BatchSamplingArgs",
    *,
    page_size: int,
    linear_pool=None,
    step: int | None = None,
) -> "torch.Tensor | None":
    """Verify, check, commit. Called from `Engine.forward_batch` in place of the sampler.

    Returns the token to write at EVERY row of the plan (`_forward` scatters it into `token_pool`
    at the write tuple, which bullet 8 widened to one row per token), or ``None`` when this batch
    was never staged -- in which case `forward_batch` runs today's `complete_one()` + sampler
    path, untouched.

    ⛔⛆ The ONE call `forward_batch` makes, for the reason bullet 7 gave for `speccheck_step`: a
    call site that passed the lengths and forgot the committed tokens, or committed the requests
    and forgot the GDN state, would look correct from every helper's side.
    """
    import torch

    staged = verify_step_of(batch)
    if staged is None:
        return None
    plan = staged.plan
    device = logits.device
    verify_uids = [plan.uids[i] for i, t in enumerate(plan.tokens_per_req) if t > 1]
    held_ids = model._pending_draft_ids
    draft_ids = torch.tensor([held_ids[uid] for uid in verify_uids], dtype=torch.int64,
                             device=device)
    # ⚠ ``None`` under greedy, where the rule needs no ``q`` at all (`verify_and_commit` asserts
    #   it is present only on the sampled branch). A held map missing ONE of the verifying uids is
    #   not a greedy step -- it is a bug -- so this is all-or-nothing rather than a partial gather.
    held_logits = getattr(model, "_pending_draft_logits", None) or {}
    draft_logits = (
        torch.cat([held_logits[uid] for uid in verify_uids], dim=0)
        if all(uid in held_logits for uid in verify_uids) and verify_uids
        else None
    )
    outcome = verify_and_commit(plan, logits[: plan.num_tokens], draft_logits, draft_ids, args)
    speccheck_step(model, plan, outcome, step=step)
    # ── the check-row instrument (bullet 9l), and it MUST be before every commit ─────────
    # ⛔⛆ It runs a PLAIN decode forward for these same requests at these same positions, and a
    #   plain decode forward starts from the step's ENTRY state. Each commit below advances exactly
    #   that state, so an instrument placed after them measures a forward the engine never takes
    #   and reports a disagreement that is its own. ⚠ A no-op unless `FREETOKEN_MTP801_CHECKROW`
    #   is non-zero, and its second forward is never inside a decode figure.
    checkrow_step(model, batch, plan, logits, linear_pool=linear_pool, step=step)
    # ── the bonus-row instrument (bullet 9bt), and it MUST be before every commit too ────
    # ⛔⛆ Its forward runs from the state AFTER the pair's FIRST token -- which it produces by
    #   calling the two commits below at ``accepted_len = 1`` and then putting the pools back.
    #   Placed after the real commits it would advance a second time, from a state the engine had
    #   already moved, and report a disagreement that is its own. ⚠ A no-op unless
    #   `FREETOKEN_MTP801_BONUSROW` is non-zero, and its second forward is never inside a decode
    #   figure.
    bonus_step(model, batch, plan, logits, outcome, linear_pool=linear_pool, step=step)
    if linear_pool is not None:
        # ⛔⛆ The half of a verify step that cannot be re-derived (bullet 5). `gdn.py` ran the
        #   recurrence with `disable_state_update=True`, so the pool still holds the step-ENTRY
        #   state and this is the ONLY thing that advances it. A verify load without this call
        #   serves from a GDN state that never moves -- which is not a crash, it is wrong text.
        # ⛔⛆ #801 bullet 9r: LOUD, never `or {}`. An absent map is "the forward was never
        #   recorded", not "nothing to commit", and the difference is a row that serves garbage.
        for layer_id, snapshot in require_linear_snapshots(
            batch, plan, where="the GDN commit"
        ).items():
            local = linear_pool.local_index(layer_id)
            commit_linear_state(
                snapshot,
                outcome.accepted_len,
                recurrent_states=linear_pool.recurrent_states[local],
                conv_states=linear_pool.conv_states[local],
            )
    if linear_pool is not None:
        # ⛔⛆ PLE's own two recurrences (bullet 8b), and they are NOT covered by the loop above:
        #   `ple.py` never goes through `build_fla_metadata`, so its conv history and n-gram
        #   context have their own snapshots and their own commit. A verify load without this call
        #   serves from a PLE state that never moves -- again not a crash, again wrong text.
        commit_ple_state(batch, outcome.accepted_len, pool=linear_pool)
    # ── the sync, and it is the whole of operator decision (a) ──────────────────────────────
    committed = [int(n) for n in outcome.accepted_len.tolist()]
    for req, accepted_len in zip(batch.reqs, committed):
        commit_verify(req, accepted_len, page_size=page_size)
    staged.committed = tuple(committed)
    return outcome.tokens


def published_tokens(batch: "Batch", next_tokens_cpu: "torch.Tensor") -> list:
    """Per request, the tokens this drained forward actually committed.

    ⛔⛆ **THE DRAIN'S OWN BUG, AND IT IS SILENT.** `_process_last_data` reads
    ``next_tokens_cpu[i]`` with ``i`` the REQUEST's position. On a verify step that tensor is one
    row per TOKEN, so ``[i]`` is the wrong element for every request after the first, and an
    accepted second token is never shipped at all.

    ⭐ A rejected request's BONUS row is dropped here. It was sampled and written unconditionally
    -- that is what keeps the scatter shape fixed for a captured graph -- but it lands at a
    position ``cached_len`` will never reach, and publishing it would ship a token the model never
    committed to.

    ⚠ Returns a list per request whatever the batch is, so the drain has ONE shape to loop over.
    A plain decode batch (no step staged) is one token each, which is today's behaviour spelled as
    a list of length one.
    """
    staged = verify_step_of(batch)
    if staged is None:
        return [next_tokens_cpu[i : i + 1] for i in range(len(batch.reqs))]
    plan = staged.plan
    assert staged.committed, (
        "a staged verify batch reached the drain with no committed lengths -- its forward "
        "never called spec.verify_step"
    )
    return [
        next_tokens_cpu[plan.cu_seqlens[i] : plan.cu_seqlens[i] + accepted_len]
        for i, accepted_len in enumerate(staged.committed)
    ]


__all__ = [
    "agree_accepted",
    "AGREE_MAX_REQS",
    "agree_row",
    "ALLOC_PAGE_END",
    "bonus_armed",
    "bonus_entry_lens",
    "bonus_forward",
    "bonus_host_ids",
    "bonus_report_rows",
    "bonus_step",
    "bonus_view",
    "checkrow_armed",
    "checkrow_rows",
    "checkrow_step",
    "commit_linear_state",
    "commit_ple_state",
    "commit_verify",
    "draft_view",
    "gather_over",
    "gdn_dump_armed",
    "gdn_dump_cell",
    "gdn_dump_describe",
    "GDN_DUMP_DIR",
    "gdn_dump_manifest",
    "GDN_DUMP_TENSORS",
    "gdncheck_batch",
    "gdncheck_instrumented",
    "gdncheck_layer",
    "GDNCHECK_OWNER",
    "host_token_id",
    "LinearSnapshot",
    "owned_page_end",
    "pages_for_step",
    "plan_verify",
    "PLEContextSnapshot",
    "PLESnapshot",
    "publish_drafts",
    "published_tokens",
    "RankDisagreement",
    "reference_device_lens",
    "reference_forward",
    "reference_view",
    "shadow_step",
    "spec_rows",
    "speccheck_armed",
    "speccheck_step",
    "stage_verify",
    "STAGED_DRAFT",
    "UNDRAINED_IDS",
    "verify_and_commit",
    "VERIFY_STEP",
    "verify_step",
    "verify_step_of",
    "VerifyOutcome",
    "VerifyPlan",
    "VerifyStep",
]
