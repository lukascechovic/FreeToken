"""The draft head's real invocation — llm-server #801 round 5 bullet 2.

⭐ **A NEW engine file**, `models/qwen4_exp/draft.py`. There is no `.orig` beside it, same reason
`mtp.py` has none: image `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` ships nothing at this
path. ⛔ BIND-MOUNTED, not patched — in no image, in no Dockerfile ladder.

⭐ **Why this is not `mtp.py`.** `mtp.py`'s own module docstring draws the line on purpose: "the
head is constructed and fed, never scheduled: there is no draft/verify/commit path here." Round 4's
`_draft_hidden` (inside `Qwen4ExpModel.forward`, fed `batch.input_ids` as a stand-in) exists only to
exercise kernels/pools/graph sizing. This module is the first REAL draft step: the head fed the
token this step's real sampler *actually chose*, called from `engine.py` (bullet 4) after
`next_tokens_gpu = self.sampler.sample(...)`, using the SAME `R` the tap already captures on every
forward when a head is loaded.

⛔⛆ **"No engine imports" is a real constraint, not a style note.** `Sampler.sample` (the real
serving path, `engine/sample.py`) is greedy-else-`sample_impl`, and `sample_impl`'s top_k/top_p
branch dispatches to `flashinfer.sampling` or `freetoken.kernel.triton.sampling` — both GPU-only
kernels that cannot run in the CPU-only container every gate in this round runs in
(`check_draft_801.sh`). So this module does NOT import `freetoken.engine.sample` at runtime: it
reimplements greedy/top_k/top_p sampling in plain torch, mirroring `head_ref_801.py`'s own
discipline of writing primitives that can be tested off-GPU rather than importing the thing being
approximated. `BatchSamplingArgs` is imported under `TYPE_CHECKING` only, for its shape — the real
dataclass the engine already builds each step (`temperatures`/`top_k`/`top_p`, each `None` or a
per-row tensor) is accepted structurally, so bullet 4's `engine.py` wiring hands this function the
SAME `args` object `forward_batch` built for the real sampler, with no conversion step of its own
to drift from it.

⚠ **This is the head's OWN draft guess, not a claim of bitwise agreement with the served kernel.**
The α this round measures is "did the head's guess match what the real sampler independently
chose" — a comparison the shadow tracker (bullet 3) makes on TOKEN IDS, not on logits or
probabilities. A few ULPs of difference between this module's nucleus-filtering and flashinfer's
own do not change which token wins often enough to matter at the 15-point gate bullet 7 evaluates
against, and are the price of a gate that can run at all off a GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.layers import BaseOP

    from .mtp import Qwen4ExpMTPHead


def is_greedy(args: "BatchSamplingArgs") -> bool:
    """Is this batch greedy? ``args.temperatures is None`` — the engine's OWN spelling.

    ⭐ `engine/sample.py::Sampler.sample` branches on exactly this, and it is a WHOLE-BATCH
    property there, not a per-row one (`index_sampling_args`' docstring says the same). Named
    once because round 5 grew three literal copies of the test — here, in :func:`sample_ids`, and in
    `model.py::mtp_shadow_step`, where it decides capture eligibility, which window a `t_draft`
    sample lands in, and whether the acceptance mass is α or a cross-check. Three copies of the
    definition of "greedy" is three places for the engine's own spelling to drift away from ours.
    """
    return getattr(args, "temperatures", None) is None


def _apply_top_k(probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    """Zero every row's mass outside its OWN top-``k`` — ``top_k`` is one int per row.

    ⚠ Ties at the cutoff are broken by ``sort``'s own (stable, descending) order, same as
    `sample_impl`'s kernels resolve them arbitrarily too — nothing here or there promises a
    particular winner among exactly-equal logits.
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    rank = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0)
    keep = rank < top_k.unsqueeze(-1).to(rank.dtype)
    sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def _apply_top_p(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Zero every row's mass outside its OWN nucleus — ``top_p`` is one float per row.

    ⭐ The kept prefix is "every token whose OWN probability arrives before the running total
    reaches ``top_p``" (i.e. keep while ``cumulative - own_prob < top_p``), which always keeps at
    least the single highest-probability token even when ``top_p`` is smaller than it.
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    keep = cumulative_before < top_p.unsqueeze(-1)
    sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def filtered_probs(logits: torch.Tensor, args: "BatchSamplingArgs") -> torch.Tensor:
    """``logits [T, vocab] -> probs [T, vocab]``: the distribution the sampler actually draws from.

    ⭐ Temperature, then top-k, then top-p, then renormalise — the order `sample_ids` below already
    used and the order `engine/sample.py::sample_impl` uses. Factored out for
    :func:`acceptance_mass` (#801 round 5 bullet 7b), which needs the DISTRIBUTIONS rather than a
    draw from them.

    ⚠ Greedy (``args.temperatures is None``) has no filter to apply and is returned as the
    one-hot on the argmax, so the function is total: every caller gets a real distribution and
    none has to special-case a ``None``. ⛔ That branch allocates a vocab-wide row, which is why
    `mtp_shadow_step` does not call :func:`acceptance_mass` on a greedy arm — there, token
    equality already IS the acceptance.
    """
    if is_greedy(args):
        ids = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, ids, 1.0)
    probs = torch.softmax(logits.float() / args.temperatures.unsqueeze(-1), dim=-1)
    if args.top_k is not None:
        probs = _apply_top_k(probs, args.top_k)
    if args.top_p is not None:
        probs = _apply_top_p(probs, args.top_p)
    return probs / probs.sum(dim=-1, keepdim=True)


class _RowSampling:
    """The three fields :func:`filtered_probs` reads, for a SUBSET of a batch's rows.

    ⛔ Not a `BatchSamplingArgs`: that dataclass is the engine's and carries more than this, and
    `draft.py` deliberately does not import it at runtime (see the module docstring). Duck-typed
    on purpose — the same structural acceptance the real args object already gets here.
    """

    __slots__ = ("temperatures", "top_k", "top_p")

    def __init__(self, temperatures, top_k, top_p) -> None:
        self.temperatures = temperatures
        self.top_k = top_k
        self.top_p = top_p


def index_sampling_args(args: "BatchSamplingArgs", rows: "list[int]") -> "_RowSampling":
    """``args`` restricted to ``rows`` (#801 round 5 bullet 7b).

    ⭐ Needed because the acceptance mass is computed on a SUBSET of a decode batch: only the
    requests that had a draft held over from the previous step can be scored, and a request on
    its first sighting has none. ⚠ A ``None`` field stays ``None`` — that is the engine's own
    spelling for "the whole batch is greedy", and it is not per-row.
    """
    # ⛔⛆ Greedy first, and by :func:`is_greedy` -- the same spelling every other branch in this
    #   module and in `model.py::mtp_shadow_step` uses. Two reasons it cannot be an attribute
    #   read here: greedy is a WHOLE-BATCH absence, so there is nothing per-row to index and the
    #   rows argument is meaningless; and this function is now reached from the greedy arm at all
    #   (round 5's `/code-review` wired `FREETOKEN_MTP801_MASSCHECK`), where the engine hook is
    #   gated with `getattr` and the CPU gates hand it a bare stand-in. Before that, only the
    #   sampled arm ever got here and the inconsistency could not show.
    if is_greedy(args):
        return _RowSampling(None, None, None)
    pick = lambda t: None if t is None else t[rows]  # noqa: E731
    return _RowSampling(pick(args.temperatures), pick(args.top_k), pick(args.top_p))


def acceptance_mass(
    target_logits: torch.Tensor, draft_logits: torch.Tensor, args: "BatchSamplingArgs"
) -> torch.Tensor:
    """``Σ_x min(p(x), q(x))`` per row: the probability a LOSSLESS verify step accepts the draft.

    ⭐⭐ **Why this exists at all, and why token equality is not it** (#801 round 5 bullet 7b).
    Issue bullet 5's verify path is exact rejection sampling: the draft's token is accepted with
    probability ``min(1, p/q)``, so over a draw from ``q`` the acceptance rate is
    ``Σ_x q(x)·min(1, p(x)/q(x)) = Σ_x min(p(x), q(x))``. Shadow mode cannot run that path, but it
    can compute its rate exactly, from the two distributions it already has. What the shadow
    tracker measures instead — did the two independent draws happen to land on the same id — is a
    DIFFERENT and systematically SMALLER number under a sampler, because two draws from the same
    distribution disagree most of the time even when the distributions are identical. ⇒ reading α
    off token equality on the served-sampler arm would understate it, and α is exactly what the
    round's ≥ +15 % gate is evaluated against.

    ⛔⛆ BOTH distributions are filtered by the SAME ``args``. The issue body's own condition for
    losslessness is that the draft's ``q`` is taken AFTER the same top-k/top-p filter as the
    target's ``p``; comparing an unfiltered ``q`` against a filtered ``p`` measures a scheme
    nobody is proposing to build.

    ⭐ Under greedy both rows are one-hot, so this returns exactly ``1.0`` on a match and ``0.0``
    otherwise — identical to token equality. `analyse_alpha_801.py` prints that agreement as a
    cross-check rather than anything here assuming it.
    """
    p = filtered_probs(target_logits, args)
    q = filtered_probs(draft_logits, args)
    return torch.minimum(p, q).sum(dim=-1)


def sample_ids(logits: torch.Tensor, args: "BatchSamplingArgs") -> torch.Tensor:
    """``logits [T, vocab] -> token_ids [T]``, greedy-else-filtered-multinomial.

    ⚠ It was ``_sample`` until #801 round 6 bullet 3. `spec.py::verify_and_commit` draws every
    non-verifying row of a verify step with exactly this function -- a plain decode row and a
    bonus row are ordinary sampler draws and must not be a second spelling of one -- and reaching
    across modules for a private name is how two spellings start.

    ⭐ Greedy is EXACTLY `engine/sample.py::Sampler.sample`'s own greedy branch
    (``args.temperatures is None`` -> ``torch.argmax``) — the one path this module need not
    approximate, since it is deterministic and needs no kernel either side of the GPU line. ⚠ It
    does NOT route through :func:`filtered_probs`' one-hot branch: that would allocate a
    vocab-wide row on every decode step of the arm this round measures `t_draft` on.
    """
    if is_greedy(args):
        return torch.argmax(logits, dim=-1)
    return torch.multinomial(filtered_probs(logits, args), num_samples=1).squeeze(-1)


def draft_next_token_ids(
    head: "Qwen4ExpMTPHead",
    lm_head: "BaseOP",
    sampled_ids: torch.Tensor,
    R: torch.Tensor,
    batch: "Batch",
    sampling_args: "BatchSamplingArgs",
) -> torch.Tensor:
    """The head's own guess at the NEXT token, given the token this step's real sampler just chose.

    ``sampled_ids [T]`` is `engine.py`'s ``next_tokens_gpu`` (bullet 4 supplies it, post-sample);
    ``R [T, hc_count*hidden]`` is the SAME step's tapped pre-final-mixer multi-stream
    (`Qwen4ExpForCausalLM.multi_stream`, valid synchronously until the next replay — see
    `mtp.py`'s module docstring for why that staleness is safe here). ``batch`` is the live batch
    the KV/rope side effects of `head.forward`'s ``self_attn`` land against.

    ⛔ Never fed back into sampling — the caller (bullet 3's tracker) only ever COMPARES this
    against the real next step's own ``sampled_ids``.
    """
    return draft_next_token_ids_and_logits(
        head, lm_head, sampled_ids, R, batch, sampling_args
    )[0]


def draft_next_token_ids_and_logits(
    head: "Qwen4ExpMTPHead",
    lm_head: "BaseOP",
    sampled_ids: torch.Tensor,
    R: torch.Tensor,
    batch: "Batch",
    sampling_args: "BatchSamplingArgs",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """:func:`draft_next_token_ids`, plus the head's own ``logits [T, vocab]`` (#801 round 5
    bullet 7b).

    ⭐ ONE forward, same as the sibling — which simply drops the second element. The logits are
    what :func:`acceptance_mass` needs for ``q``; nothing else reads them, and the caller keeps
    the mass computation OUTSIDE its `timing.py::time_draft_call` window so `t_draft` stays the
    draft's own cost and not the measurement's.

    ⛔ There is no captured twin: `capture.py::DraftGraphRunner` is greedy-only and bakes the
    argmax into the graph, and the sampled arm — the only one that needs a mass — always runs
    eager for exactly that reason.
    """
    sample_hidden, _ = head.forward(sampled_ids, R, batch)
    logits = lm_head.forward(sample_hidden)
    return sample_ids(logits, sampling_args), logits


__all__ = [
    "acceptance_mass",
    "sample_ids",
    "index_sampling_args",
    "draft_next_token_ids",
    "draft_next_token_ids_and_logits",
    "filtered_probs",
]
