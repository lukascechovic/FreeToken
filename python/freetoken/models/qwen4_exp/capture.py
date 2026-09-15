"""`t_draft`, captured — a dedicated small CUDA graph for the head's own forward, llm-server #801
round 5 bullet 6.

⭐ **A NEW engine file**, `models/qwen4_exp/capture.py`. There is no `.orig` beside it, same reason
`draft.py`/`shadow.py`/`timing.py` have none — nothing at this path ships in image
`llm-server/freetoken-gfx1201:2026-09-09-agree-0022`. ⛔ BIND-MOUNTED, not patched — in no image,
in no Dockerfile ladder.

⭐ **What this captures.** `draft.py::draft_next_token_ids`'s own chain — `head.forward` (the
head's one QSA decoder layer, real KV writes, real MoE) → `lm_head.forward` → sampling — as ONE
`torch.cuda.graph`, replayed each eligible decode step instead of run eagerly. Bullet 5's
`RollingStats`/`time_draft_call` wrap EITHER path identically (`model.py::mtp_shadow_step` picks
which); this file only supplies the captured half.

⛔⛆ **SCOPE, DELIBERATELY NARROW — round 5's own plan, approved before this file was written:**

* **bs = 1 ONLY.** This round's shadow-mode loads run `--max-running-requests 1`; a per-bs
  `graph_map` (`engine/graph.py::GraphRunner`'s own shape) is real complexity this bullet's own
  question — does capture beat eager at all — does not need yet. `replay()` asserts it.
* **GREEDY ONLY.** `draft.py::sample_ids`'s top_k/top_p branch calls `torch.multinomial`; an RNG draw
  baked into a captured graph replays the SAME random draw every time unless a graph-safe generator
  is plumbed through — a separate engineering problem, orthogonal to "does capture beat eager," and
  deferred rather than solved here. Greedy (`torch.argmax`) is deterministic and capture-safe with
  no extra machinery, and matches this round's own gate discipline (`top_k:1,top_p:1` pinned
  elsewhere for exactly this reason).
* **Captured at ENGINE BRING-UP**, not lazily on the first real shadow step — see :meth:`capture`'s
  own docstring for why (keeps bullet 5's `t_draft` EAGER timing clean, and matches
  `Qwen4ExpModel.reserve_multi_stream`'s "nothing may allocate during a capture" ordering rule).

⛔⛆ **THE LOAD-BEARING ASSUMPTION THIS FILE MAKES, UNVERIFIED OFF A BOX.**
`attention/qsa_sparse.py::QSASparseAttnBackend`'s capture-mode scratch (`init_capture_graph`'s
`self._graph` dict — block table, kvlen, index selection) is SHARED, address-stable,
per-forward-transient storage, refilled by `prepare_for_capture`/`prepare_for_replay`
(`_stage_decode`) and read by every QSA layer regardless of which `Batch` object currently wraps
it. `replay()` below never calls either of those itself: it relies on the BACKBONE's own decode
replay (`engine/graph.py::GraphRunner.replay`, which runs earlier in the same step, inside
`engine.py`'s `forward_batch`) having already called `prepare_for_replay(batch)` for THIS exact
request — so by the time this graph replays, the shared scratch already holds the correct, fresh,
per-step attention-selection data, and duplicating that call here would just be a second write of
the same content. This is sound **only while `batch.padded_size == 1`** end to end (this round's
own arms) — a row that ever schedules more than one concurrent decode request would have the
backbone's `prepare_for_replay` stage row-0-of-N data that need not be THIS request's, and this
file has no way to tell. `replay()` asserts `padded_size == 1` for exactly that reason: if the
assumption is ever wrong, it fails loudly instead of drafting silently from a stale window. ⛔ The
one thing this cannot make safe by construction, and the one thing a box load must confirm before
any number from the captured arm is banked — the same discipline `headcheck.py`'s own docstring
states: "the real correctness gate ... a box load, not off-GPU theatre."

⛔ `out_loc`/`positions` are NOT part of that shared scratch — they address where THIS forward's
OWN K/V write and rope angle land, one layer further up than the backbone (the head's own
dedicated layer id). Those two `replay()` DOES refresh itself, every call, from the live batch —
same `copy_from`-shaped idiom `engine/graph.py::GraphCaptureBuffer` already uses for the backbone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.core import Batch, Req
    from freetoken.layers import BaseOP

    from .mtp import Qwen4ExpMTPHead


class DraftGraphRunner:
    """Captures and replays the head's forward + `lm_head` + greedy argmax, at a fixed bs=1.

    ⛔ One instance per model; :meth:`capture` runs exactly once (engine bring-up), :meth:`replay`
    on every eligible decode step after that.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self._batch: "Batch | None" = None
        # ⛔⛆ Own static buffers, at final size, allocated HERE — before any capture — for the two
        # things that are not already a stable address by the time `capture()` runs: the draft's
        # input token (fresh every step, unlike `R` — see `Qwen4ExpModel.reserve_multi_stream`,
        # already static) and this graph's own KV/rope addressing and output.
        self.sampled_ids = torch.zeros(1, dtype=torch.int32, device=device)
        self.out_loc = torch.zeros(1, dtype=torch.int32, device=device)
        self.positions = torch.zeros(1, dtype=torch.int32, device=device)
        self.predicted_ids = torch.zeros(1, dtype=torch.int32, device=device)
        # ⭐ Allocated once `capture()` knows `vocab_size` (`lm_head`'s own output width) — before
        # ANY capture, same rule as the four buffers above. `None` until then.
        self.logits: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        return self.graph is not None

    def capture(
        self,
        head: "Qwen4ExpMTPHead",
        lm_head: "BaseOP",
        attn_backend: "BaseAttnBackend",
        stream: torch.cuda.Stream,
        dummy_req: "Req",
        R: torch.Tensor,
    ) -> None:
        """Warm up eagerly, then capture, on a dedicated dummy bs=1 decode batch.

        ⛔ Called from engine bring-up (`engine.py`'s `mtp_capture_draft_graph` hook), AFTER
        `GraphRunner._capture_graphs` has finished capturing the BACKBONE's own decode graphs —
        the same ordering `reserve_multi_stream`'s own docstring requires, for the same reason: an
        allocation made DURING a `torch.cuda.graph(...)` block comes out of THAT graph's private
        pool, so every buffer above is allocated in `__init__`, never here. ``stream``: the SAME
        `engine.py::self.stream` `GraphRunner` itself captures on — matching precedent rather than
        capturing on whatever the default stream happens to be.

        ``dummy_req``: the SAME placeholder `GraphRunner` itself captures on (its own `dummy_req`)
        — its `table_idx` addresses scratch, never a real request's KV. ``R``:
        `Qwen4ExpModel.multi_stream_buffer` — the tap's own static buffer; capture only needs ONE
        valid row of it (real content does not matter yet — only its address and dtype do).
        """
        from freetoken.core import Batch, get_global_ctx

        assert self.graph is None, "#801: draft graph captured twice"
        batch = Batch(reqs=[dummy_req], phase="decode")
        batch.padded_reqs = batch.reqs
        batch.input_ids = self.sampled_ids
        batch.out_loc = self.out_loc
        batch.positions = self.positions
        attn_backend.prepare_for_capture(batch)
        self._batch = batch  # kept alive: its `attn_metadata` is what the captured kernels read

        def _step() -> tuple[torch.Tensor, torch.Tensor]:
            hidden, _ = head.forward(self.sampled_ids, R[:1], batch)
            logits = lm_head.forward(hidden)
            return logits, torch.argmax(logits, dim=-1).to(torch.int32)

        with get_global_ctx().forward_batch(batch):
            logits, ids = _step()  # eager warm-up -- discovers vocab_size, before ANY capture
            self.logits = torch.empty_like(logits)
            self.logits.copy_(logits)
            self.predicted_ids.copy_(ids)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                logits, ids = _step()
                self.logits.copy_(logits)
                self.predicted_ids.copy_(ids)

    def replay(self, sampled_ids: torch.Tensor, batch: "Batch") -> torch.Tensor:
        """Refill the input buffers from THIS step's real values, replay, return a fresh tensor.

        ⛔⛆ ``batch`` must be THIS step's already-processed decode batch (`mtp_shadow_step`'s own
        argument) — see this module's docstring for why its attention-SELECTION scratch is not
        re-derived here. ⛔ `padded_size` must be 1 — the one precondition this file cannot
        enforce by construction, so it is checked, loudly, every call, same as `sampled_ids`'.
        """
        assert self.graph is not None, "#801: draft graph replay before capture"
        assert sampled_ids.shape[0] == 1, (
            f"#801: draft graph is bs=1 only, got {sampled_ids.shape[0]} sampled id(s)"
        )
        assert batch.padded_size == 1, (
            f"#801: draft graph is bs=1 only, batch padded to {batch.padded_size}"
        )
        self.sampled_ids.copy_(sampled_ids)
        self.out_loc.copy_(batch.out_loc[:1])
        self.positions.copy_(batch.positions[:1])
        self.graph.replay()
        return self.predicted_ids.clone()

    def diff_against_eager(
        self,
        head: "Qwen4ExpMTPHead",
        lm_head: "BaseOP",
        sampled_ids: torch.Tensor,
        R: torch.Tensor,
        batch: "Batch",
    ) -> dict:
        """Bullet 6's own correctness gate: this step's CAPTURED replay against a fresh EAGER call
        on the identical inputs — the engine's own two implementations against each other, same
        discipline `headcheck.py` already established for round 4's kernels, never a third oracle.

        ⛔ Runs the eager path a SECOND time this step (`head.forward` + `lm_head.forward`)
        purely for the diff; the caller decides whether that cost is worth paying on any given
        step (`model.py`'s own countdown dial, mirroring `_mtp_check_left`).

        ⛔⛆ **THE CALLER MUST ALREADY BE INSIDE `ctx.forward_batch(batch)`, AND THIS METHOD MUST
        NOT OPEN ONE.** The head's real MoE forward reads `get_global_ctx().batch`
        unconditionally, which is bullet 4's finding — but bullet 4 fixed it at the ENGINE CALL
        SITE (`build_engine_overlay_801.py` edit 7 wraps the whole `mtp_shadow_step` call), so by
        the time this runs the context is already open and `core.py:206` asserts ``Nested
        forward_batch is not allowed``. ⚠ This method DID open one until #801 round 5 bullet 7d's
        first box load, which died on exactly that assert on its first decode step — bullet 6
        reasoned from bullet 4's LESSON ("wrap it") without checking bullet 4's IMPLEMENTATION
        (already wrapped, one level up). ⭐ `replay()` above is the counter-example that was right
        all along: it opens no context either. ⇒ the invariant is "the caller owns the context",
        and `test_shadow_wiring_801.py::TestDiffAgainstEagerOpensNoContext` is what holds it.
        """
        eager_hidden, _ = head.forward(sampled_ids, R, batch)
        eager_logits = lm_head.forward(eager_hidden)
        eager_ids = torch.argmax(eager_logits, dim=-1).to(torch.int32)
        captured_ids = self.replay(sampled_ids, batch)
        diff = (self.logits.float() - eager_logits.float()).abs()
        return {
            "ids_match": bool(torch.equal(captured_ids, eager_ids)),
            "captured_id": int(captured_ids.item()),
            "eager_id": int(eager_ids.item()),
            "logits_max_abs_diff": float(diff.max().item()),
            "logits_mean_abs_diff": float(diff.mean().item()),
        }


__all__ = ["DraftGraphRunner"]
