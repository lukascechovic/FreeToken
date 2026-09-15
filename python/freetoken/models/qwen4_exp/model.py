"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer
from .spec import FT801_HIDDENCHECK_LAYERS, FT801_HIDDENCHECK_NAMES

# ── #801 overlay marker ──────────────────────────────────────────────────────────────
# This file is `models/qwen4_exp/model.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 489c549617a864bedb93c79c23d4cb89,
# 254 lines) BIND-MOUNTED over the installed package, plus this block, the multi-stream tap on
# `Qwen4ExpModel` and the `mtp` attribute on `Qwen4ExpForCausalLM`. ⛔ It is NOT a patch in the
# Dockerfile ladder and is in NO image.
#
# ⚠ WHY A MARKER EXISTS AT ALL. A `-v` that silently does not take -- wrong path inside the image,
#   a typo'd source, a file the container cannot read -- leaves the row running the IMAGE's model
#   while every log line looks exactly like the arm we think we launched (#866). ⛔ And this
#   overlay's whole claim is that the flag-off path is byte-identical to that image file, so
#   "which one is loaded" is the one thing a reader must never have to infer.
#
# ⚠ stderr, not a logger: `docker logs` captures it, and it lands before the engine's logging is up.
import sys as _ft801_sys

# ⚠ #801 round 6 bullet 1: the `json` import stood here for round 5's α lines. Its only consumers
#   moved to `spec.py` with the step, so it is GONE -- this file's differential against the
#   image's copy is what the flag-off path is argued on, and an import with no reader is one more
#   line to argue about.


print(
    "[#801] overlay ACTIVE: models/qwen4_exp/model.py bind-mounted from the repo "
    f"(pid {os.getpid()}, base md5 489c549617a864bedb93c79c23d4cb89, "
    f"FREETOKEN_LOAD_MTP={os.getenv('FREETOKEN_LOAD_MTP', '<unset>')})",
    file=_ft801_sys.stderr,
    flush=True,
)

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models.config import ModelConfig

    from .timing import RollingStats


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )
        # ⭐ #801 r6 b9cb: this layer's slice of the hidden-state fingerprint buffer, or None --
        #   set on the layers in `spec.FT801_HIDDENCHECK_LAYERS` and on no others. Leading underscore
        #   for `_multi_stream_out`'s reason: a bare tensor attribute enters `BaseOP.state_dict`
        #   and becomes a checkpoint key nothing ships. Set by `Qwen4ExpModel.reserve_hiddencheck`
        #   BEFORE any capture, never here -- there is no device or dtype to size it from yet.
        self._ft801_hiddencheck: torch.Tensor | None = None

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        # ⭐⭐ #801 r6 b9cb: ONE ATTRIBUTE READ per layer per forward when the dial is off, and
        #   `None` on every layer outside `spec.FT801_HIDDENCHECK_LAYERS` even when it is on --
        #   `commitcheck`'s standard.
        #   ⛔⛆ The branch is taken at CAPTURE time and baked: the buffer is reserved BEFORE
        #   `GraphRunner._capture_graphs`, so an armed load's writes are INSIDE the graph and
        #   REPLAY, and an unarmed load captures a graph with no writes in it at all.
        _ft801_hc = self._ft801_hiddencheck
        if _ft801_hc is not None:
            return self._ft801_forward_hiddencheck(hidden, batch, _ft801_hc)
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)

    def _ft801_forward_hiddencheck(
        self, hidden: torch.Tensor, batch: Batch, buf: torch.Tensor
    ) -> torch.Tensor:
        """:meth:`forward`, plus a per-row fingerprint of each of the block's seven tensors.

        ⛔⛆ **A SECOND BODY, AND THE ALTERNATIVE WAS WORSE.** Naming the intermediates in
        :meth:`forward` itself keeps `post_ple` and the attention `block_input` alive to the end of
        the function -- up to two extra ``[T, hc*hidden]`` tensors per layer -- and that is paid on
        EVERY SERVED ROW, dial off, on a 48-layer backbone. An instrument may not cost the row it
        is not measuring. ⇒ the served body above is untouched and this one is only ever entered by
        the layers `spec.FT801_HIDDENCHECK_LAYERS` names, on an armed load -- two of forty-eight.
        ⛔ The drift that buys is gated:
        `test_load_801.py::test_the_instrumented_block_computes_the_same_calls_in_the_same_order`
        compares the two bodies' call sequences, so a change to one alone goes red.

        ⭐⭐⭐ **WHY SEVEN AND WHY THESE.** Load 30 localised the arms' first divergence to inside
        layer 0's block on one forward: layer 0's own committed GDN state is bit-identical on both
        arms while layer 1's differs ⇒ layer 0 got identical input, wrote identical state, and
        handed layer 1 something different. `mixer_out` and `mlp_out` are the two candidates; the
        other five exist so that *"the instrument found nothing"* is impossible unless the boundary
        moved -- they cover both hyper-connection combines and the PLE add.

        ⭐⭐⭐ **AND #983 ARMED LAYER 1 WITH THE SAME SEVEN.** By then the block had been cleared:
        9ce read layer 0 identical cross-arm through 11,565, #980 found the gap's six bf16 stages
        M-invariant (0/256, both dtypes) and #981 could not break a premise -- yet STATECHECK
        still says layer 1's committed state differs. ⇒ the seven land EITHER SIDE of the gap on
        served data: `in` re-confirms the handover (it must equal layer 0's `block_out`),
        `mixer_in` is the output of #980's five `hc` stages ON THE ROW THE ENGINE RUNS, and
        `mixer_out` is the GDN's own.

        ⛔⛆ **AND `post_ple` IS WHERE IT LANDED, WHICH NOBODY EXPECTED.** #981 wrote *"layer 1
        carries no PLE, `ple_layer_ids` is [2]"* into the round's premise chain off the
        CHECKPOINT's config. `models/qwen4_exp/config.py` reads that field as **ONE-INDEXED** and
        subtracts one -- ``[2]`` is model layer **1**. ⇒ layer 1 DOES carry the PLE, it sits
        UPSTREAM of the gap #976 and #978 drew and #980 swept, and #983 measured the arms' first
        divergence exactly there.
        """
        from .spec import ft801_hidden_fingerprint

        # ⛔⛆ **#801 r6 b9cf: THE GUARD THAT TURNS 9cb's CRASH INTO A SENTENCE.** All seven tensors
        #   are `[T, *rest]` at this forward's own T, and the buffer is reserved at bring-up. If T
        #   ever exceeds it, torch's own message names two integers and no cause -- which is what
        #   9cb would have cost load 31 at the first prefill chunk. ⚠ It is python, so a CAPTURE
        #   bakes it out at the captured width; it protects the EAGER prefill, which is exactly the
        #   path that was broken, and costs one integer compare per armed forward.
        if hidden.shape[0] > buf.shape[1]:
            raise RuntimeError(
                f"#801: HIDDENCHECK's buffer holds {buf.shape[1]} row(s) and this forward is "
                f"{hidden.shape[0]} row(s) wide -- `mtp_reserve_graph_buffers` is sized by the "
                f"scheduler's TOKEN budget (max_extend_tokens), not by the request count, and "
                f"this one is short. See bullet 9cf."
            )

        # ⛔⛆ **THE LOCALS ARE NAMED FOR `spec.FT801_HIDDENCHECK_NAMES`, ELEMENT FOR ELEMENT, AND
        #   A GATE READS THEM.** Nothing else ties the order these are WRITTEN to the order the
        #   reader NAMES them, and a swap of `mixer_out` and `mlp_out` would hand CANDIDATE B's
        #   numbers to CANDIDATE A -- a confident, inverted verdict, which is worse than no
        #   instrument at all. ⚠ `hidden_in` and not `in`, which is a keyword.
        hidden_in = hidden
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        post_ple = hidden
        mixer_in, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            mixer_out = self.linear_attn.forward(mixer_in)
        else:
            mixer_out = self.self_attn.forward(mixer_in, batch)
        hidden = self.attn_hyper_connection.combine(hidden, mixer_out, inject)
        attn_resid = hidden
        mlp_in, inject = self.mlp_hyper_connection.mix(hidden)
        mlp_out = self.mlp.forward(mlp_in)
        block_out = self.mlp_hyper_connection.combine(hidden, mlp_out, inject)
        # ⛔ DEVICE WORK ONLY -- no `.tolist()`, no `.item()`. The host read is `engine.py`'s,
        #   outside the graph, beside STATECHECK's. A sync here would be illegal on a capturing
        #   stream: the class of call that killed this round's first box load.
        # ⚠ `[:rows]`, never the whole buffer: it is sized for the largest batch the capture
        #   resolver reserved, and a forward narrower than that must not overwrite a stale row
        #   with garbage -- the reader slices by the request's own `device_len`.
        for _slot, _tensor in enumerate(
            (hidden_in, post_ple, mixer_in, mixer_out, attn_resid, mlp_out, block_out)
        ):
            buf[_slot, : _tensor.shape[0]] = ft801_hidden_fingerprint(_tensor)
        return block_out


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self._image_token_id = config.image_token_id
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)
        # #801: the multi-stream tap's static output buffer, allocated on first use or by
        # `reserve_multi_stream`. Leading underscore, for the reason the line above gives: a bare
        # tensor attribute enters `BaseOP.state_dict` and becomes a checkpoint key nothing ships.
        self._multi_stream_out: torch.Tensor | None = None
        # #801 bullet 7: the head's DISCARDED output, and the full multi-stream it was fed,
        # stashed rather than returned so the tap's return arity -- and every gate written against
        # it -- is unchanged. ⚠ Holding the stream costs one [T, hc*hidden] tensor across the gap
        # between forwards (~80 MiB at the deployed --max-prefill-length 4096, and it is freed on
        # the next forward). Only a head-bearing research arm ever pays it.
        self._draft_hidden: torch.Tensor | None = None
        self._draft_stream: torch.Tensor | None = None
        # ⭐⭐ #801 r6 b9cb, widened by #983: the ARMED LAYERS' hidden-state fingerprint buffer,
        #   stacked in `spec.FT801_HIDDENCHECK_LAYERS` order. Same underscore rule and the same
        #   ordering rule as `_multi_stream_out` above -- see `reserve_hiddencheck`.
        self._hiddencheck_out: torch.Tensor | None = None

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scatter precomputed image soft tokens over the ``image_token_id`` placeholders.

        ⚠ Runs on the ``[T, hidden]`` embedding, BEFORE it is repeated across the ``hc_count``
        hyper-connection streams -- the image features are text-hidden-sized, not stream-sized.
        Only prefill batches carrying images reach the scatter; decode batches never do.
        """
        mm_embeds = getattr(get_global_ctx().batch, "mm_embeds", None)
        if mm_embeds is None or self._image_token_id is None:
            return x
        mask = input_ids == self._image_token_id
        n_slots = int(mask.sum().item())
        assert n_slots == mm_embeds.shape[0], (
            f"image-token slots ({n_slots}) != vision features ({mm_embeds.shape[0]}); "
            "the scheduler must hand each prefill chunk exactly its own soft-token rows"
        )
        return x.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(x.dtype))

    @property
    def multi_stream_buffer(self) -> torch.Tensor | None:
        """The tap's static output buffer, or ``None`` while the tap has never been asked for."""
        return self._multi_stream_out

    @property
    def hiddencheck_buffer(self) -> torch.Tensor | None:
        """The armed layers' fingerprint buffer, or ``None`` while HIDDENCHECK has never been
        armed. ⚠ ONE buffer for all of `spec.FT801_HIDDENCHECK_LAYERS`, stacked in that order --
        `engine.py` copies it whole and the READER is what splits it back into layers.

        ⚠ An accessor rather than a reach into `_hiddencheck_out` from the causal LM: the leading
        underscore is `BaseOP.state_dict`'s rule (see `__init__`), not an invitation to read the
        attribute from another class.
        """
        return self._hiddencheck_out

    def reserve_multi_stream(self, max_rows: int) -> torch.Tensor:
        """Size the tap's static buffer for up to ``max_rows`` requests. Never shrinks.

        ⛔⛆ A captured decode graph writes into the addresses it was captured with, and an
        allocation made DURING capture comes out of the graph's private pool -- right on the eager
        warm-up, wrong on every replay. So the buffer must exist, at its final size, BEFORE
        `engine/graph.py::GraphRunner._capture_graphs` runs. This is the seam that says so, rather
        than leaning on the capture loop happening to visit the largest batch size first.

        ⭐ Shape and dtype come from the final mixer's own weight: the buffer is exactly as wide as
        the multi-stream that mixer consumes, in the weights' own dtype. ⛔ That dtype is
        load-bearing -- `LinearReplicated` in the draft head is a bare `F.linear` with no cast, so
        a widened ``R`` raises rather than being promoted.
        """
        held = self._multi_stream_out
        if held is None or held.shape[0] < max_rows:
            template = self.hyper_connection_mixer.input_mix_weight_down.weight
            self._multi_stream_out = torch.empty(
                (max_rows, template.shape[1]), dtype=template.dtype, device=template.device
            )
        return self._multi_stream_out

    def reserve_hiddencheck(self, max_tokens: int) -> torch.Tensor:
        """Size the fingerprint buffer for up to ``max_tokens`` rows, and hand each armed layer
        its own slice of it.

        ⛔⛆ **THE UNIT IS TOKENS, NOT REQUESTS, AND 9cb GOT IT WRONG.** An armed layer writes one integer
        per ROW OF THE FORWARD, and a prefill chunk is the scheduler's whole token budget --
        `--max-prefill-length 4096` on the deployed arm, against a `max_rows` of ~160. The caller
        (`Qwen4ExpForCausalLM.mtp_reserve_graph_buffers`) is where the two currencies are named.

        ⛔⛆ **BEFORE ANY CAPTURE, for `reserve_multi_stream`'s reason, and here it decides the
        whole measurement.** A captured decode graph writes into the addresses it was captured
        with, and an allocation made DURING capture comes out of the graph's private pool -- right
        on the eager warm-up, wrong on every replay. ⭐ Reserving here is also what lets layer 0's
        write be captured INSIDE the graph and REPLAY, which is why this dial -- unlike GDNCHECK --
        never makes `can_use_cuda_graph` decline. That matters more than it sounds: GDNCHECK's
        budgeted steps run EAGER (9q, three loads), and load 30's boundary was measured on the
        CAPTURED path, so a dial that forced eager could move or erase the thing being measured.

        ⛔ **THE LAYERS ARE `spec.FT801_HIDDENCHECK_LAYERS`, AND THE LIST IS SHORT ON PURPOSE.**
        9bz put the residual inside layer 0's block, so 9cb armed layer 0 alone; #983 added layer
        1, because that is where STATECHECK says the arms' committed state first differs and
        nothing in this round had ever measured it ON SERVED DATA. The emit is multiplied by the
        LENGTH of that tuple -- arming all 48 would multiply it by 48 for the 46 that are not the
        question, which is 9cb's rule and is not repealed.

        ⛔⛆ **EACH ARMED LAYER GETS A VIEW, NEVER ITS OWN TENSOR, AND THE CAPTURE IS WHY.** One
        allocation means one set of addresses for `GraphRunner._capture_graphs` to bake and one
        host copy for `engine.py` to slice -- which is what keeps the emit's tensor count
        `len(NAMES) * len(LAYERS)` with no second buffer to keep in step. ⚠ `buf[slot, :T] = ...`
        through a view writes the parent's storage at a fixed offset, so an armed layer's write
        is captured and replays exactly as 9cb's did.

        ⚠ Shape ``[7*L, max_tokens]`` int64 -- one integer per block tensor per ROW, layer
        `LAYERS[i]` owning slots ``i*7 .. (i+1)*7``. ⛔⛆ **STACK ORDER, because the verdict is
        POSITIONAL**: the reader walks slots in ascending order and the first that differs names
        the stage, so a buffer laid out against the stack would name a LATER layer as the birth of
        a divergence an EARLIER one already carried. Device comes from the final mixer's own
        weight, exactly as the tap's buffer does; the DTYPE does not, because a fingerprint is an
        integer and the weights are not.
        """
        held = self._hiddencheck_out
        if held is None or held.shape[1] < max_tokens:
            template = self.hyper_connection_mixer.input_mix_weight_down.weight
            width = len(FT801_HIDDENCHECK_NAMES)
            self._hiddencheck_out = torch.zeros(
                (width * len(FT801_HIDDENCHECK_LAYERS), max_tokens),
                dtype=torch.int64,
                device=template.device,
            )
            # ⛔ EVERY armed layer is re-pointed on a GROW, not just the ones that had no buffer:
            #   a layer still holding a view of the OLD allocation would write somewhere the host
            #   read no longer looks at, and the emit would carry seven zeros that read exactly
            #   like "these two arms agree" (#866's shape, in the one place it would be believed).
            for _i, _layer_id in enumerate(FT801_HIDDENCHECK_LAYERS):
                self.layers.op_list[_layer_id]._ft801_hiddencheck = (
                    self._hiddencheck_out[_i * width : (_i + 1) * width]
                )
        return self._hiddencheck_out

    @staticmethod
    def multi_stream_rows(batch: Batch) -> torch.Tensor | None:
        """Which rows of a forward the tap takes: an index tensor, or ``None`` for "every row".

        ⭐ **The one spelling of the rule.** `_tap_multi_stream` reads it, and so does whatever
        feeds the head the token that goes with each tapped ``R`` -- llm-server #801 bullet 7's
        `Qwen4ExpForCausalLM.forward` does exactly that. ⛔ A second spelling would look identical
        today and be a second thing to keep in step with the scheduler's ragged layout; bullet 6
        already refused one inside this file, and a caller outside it is the same mistake at a
        longer distance.

        ⚠ It is `layers/embedding.py::ParallelLMHead.forward`'s accessor, on the same metadata
        object -- not a restatement of what that accessor does.
        """
        return batch.attn_metadata.get_last_indices(batch.size) if batch.is_prefill else None

    def _tap_multi_stream(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        """The pre-final-mixer multi-stream ``R``, at the row the LM head reads for each request.

        ⭐ "The last position of each request" is not defined here. It is whatever
        `layers/embedding.py::ParallelLMHead.forward` selects, read through the same accessor on
        the same metadata object -- a second spelling of the scheduler's ragged layout would be a
        second thing to keep in step with it. A decode forward has one row per padded request and
        the LM head selects nothing, so there every row is already a last position (and the
        padding rows come back too: `GraphRunner.replay` is what trims to ``batch.size``).

        ⛔ The rows are COPIED into the static buffer rather than returned as a view of ``hidden``,
        which is the whole point -- see :meth:`reserve_multi_stream`.
        """
        selector = self.multi_stream_rows(batch)
        rows = hidden if selector is None else hidden[selector]
        out = self.reserve_multi_stream(rows.shape[0])[: rows.shape[0]]
        out.copy_(rows)
        return out

    @property
    def draft_hidden(self) -> torch.Tensor | None:
        """The last forward's DISCARDED draft output (llm-server #801 bullet 7), or ``None``.

        ⚠ Same staleness as :attr:`Qwen4ExpForCausalLM.multi_stream`, and for a stronger reason: a
        captured decode graph allocated it from the graph's PRIVATE POOL and refills it there on
        every replay. ⛔ It is read only off a prefill (`headcheck.py` refuses decode outright), and
        nothing downstream consumes it -- bullet 7 consumes no draft.
        """
        return self._draft_hidden

    @property
    def draft_stream(self) -> torch.Tensor | None:
        """The full pre-final-mixer ``R`` the head was last fed -- every row of that forward.

        ⛔ NOT :attr:`Qwen4ExpForCausalLM.multi_stream`, which is the tap: one row per request, for
        a draft step to consume. The difference is what killed bullet 7's first load."""
        return self._draft_stream

    def forward(
        self,
        input_ids: torch.Tensor,
        batch: Batch,
        *,
        want_multi_stream: bool = False,
        draft_head: BaseOP | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """The collapsed hidden ``[T, hidden]``; with ``want_multi_stream``, also the draft head's
        input ``[bs, hc_count*hidden]`` (llm-server #801).

        ⛔ Both extras are keyword-only and default OFF, and with both off this returns exactly what
        it returned before they existed, computed by exactly the same statements.

        ⛔⛆ **``draft_head`` RUNS THE HEAD ON THIS FORWARD'S OWN ROWS, AND THE TAP'S ROWS ARE A
        DIFFERENT SET.** Found by loading, #801 bullet 7: a head fed the TAP's rows (one per
        request) on a 53-token prefill batch dies in
        ``kvcache/mha_pool.py::store_kv`` -> ``store.cu:88`` with *"Size mismatch for L(shape#0):
        expected 1 but got 53"*, because `QSASparseAttnBackend` addresses its slab with
        ``batch.out_loc`` and its rope with ``batch.positions`` -- both ``[T]``. ⇒ the tap is what a
        DRAFT STEP CONSUMES; a head that actually runs is a layer, and a layer runs on the batch.
        ⚠ No CPU gate could see this: bullets 3-6 fed the head a batch whose token count happened
        to equal ``R``'s rows.
        """
        embedded = self._merge_multimodal(input_ids, self.embed_tokens.forward(input_ids))
        hidden = embedded.repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            # #801 bullet 8b: `batch=` is REQUIRED on a verify step -- the context is a
            # recurrence, so rolling it here would fold this step's rejected drafts in
            # permanently. It records the step instead; `spec.commit_ple_state` rolls it.
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None), batch=batch)
        if draft_head is not None:
            # ⛔ ``hidden`` is the PRE-final-mixer multi-stream at every row of this forward, which
            #   is exactly what the head's `fuse_input` wants -- and `batch` is the same object the
            #   backbone's 48 layers just used, so the head's KV lands at the same ``out_loc`` and
            #   its rope at the same ``positions``, one layer further up.
            # ⚠ `batch.input_ids` is a STAND-IN for the token a real draft step would fuse (that
            #   one is sampled after `lm_head`, and does not exist yet). Bullet 7 runs the head for
            #   its kernels, its pools and its graph, never for its predictions.
            self._draft_stream = hidden
            self._draft_hidden = draft_head.forward(input_ids, hidden, batch)[0]
        collapsed = self.hyper_connection_mixer.mix(hidden)[0]
        if not want_multi_stream:
            return collapsed
        return collapsed, self._tap_multi_stream(hidden, batch)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        if config.is_multimodal:
            from .vision import Qwen4ExpVisionModel

            self.visual = Qwen4ExpVisionModel(config.vision_config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        # ⛔⛆ #801: THE DRAFT HEAD HANGS OFF THIS MODULE AS `mtp`, AND NOTHING ELSE.
        #   `weight.py::_rename` returns the checkpoint's 29 head keys UNCHANGED at `mtp.`, and
        #   `layers/base.py::BaseOP.load_state_dict` derives every lookup from the ATTRIBUTE NAME
        #   it recursed through -- so this attribute name IS the checkpoint prefix. Renaming it
        #   does not fail loudly; it fails as `KeyError: 'pre_fc_norm_embedding.weight'`.
        #   ⚠ `None` unless the operator asked for it. `mtp_load_enabled()` is the LOADER's own
        #   predicate (one spelling of `FREETOKEN_LOAD_MTP`, not a second one here): keeping the
        #   keys and building the module that receives them must never disagree.
        #   ⚠ One layer, the checkpoint's own `mtp_num_hidden_layers`. `config.py::parse_config`
        #   parses none of the checkpoint's `mtp` block, so nothing here can read it -- see
        #   `mtp.py`'s module docstring for what that costs and what it does not.
        from .weight import (
            mtp_draft_alternate_enabled,
            mtp_draft_capture_enabled,
            mtp_load_enabled,
            mtp_run_enabled,
            mtp_shadow_enabled,
            mtp_verify_enabled,
        )

        self.mtp = None
        self._multi_stream: torch.Tensor | None = None
        # #801 bullet 7: a resident head is FORWARDED unless the operator says otherwise. Read
        # once, here, so a captured graph and an eager step can never disagree about it.
        # ⛔ `_draft_hidden` is the head's collapsed output and is DISCARDED: bullet 7 consumes no
        #   draft. Leading underscore, for `BaseOP.state_dict`'s sake (bullet 6).
        self._mtp_run = False
        # round 6 bullet 6: whether this load runs a real VERIFY step. Read once, here, same
        # reason as `_mtp_run` -- and sharper: this dial is what decides whether a SECOND set of
        # CUDA graphs is captured at all, so a later read could not change the answer anyway.
        self._verify_run = False
        # round 5 bullet 4: the REAL draft step's tracker, or `None` -- read once, same reason as
        # `_mtp_run` above. ⛔ Independent of `_mtp_run`: this is not round 4's discarded stand-in,
        # it is `draft.py`'s real invocation, called from `engine.py` after the real sampler
        # (`mtp_shadow_step` below).
        self._shadow_tracker = None
        # round 5 bullet 4's box load (#801): the FIRST time `mtp_shadow_step` actually returns,
        # so the load gate has something on stderr to grep for besides "it did not crash" -- a
        # dial that is read but never fires (a `hasattr` miss, a phase-gate that never opens) would
        # otherwise pass the byte-identical check VACUOUSLY, the same trap #866 already found once.
        self._shadow_fired = False
        # round 5 bullet 5: t_draft, eager -- how long bullet 2's real draft call takes, measured
        # with a `torch.cuda.Event` pair around it (`timing.py::time_draft_call`) and kept as a
        # rolling window (`timing.py::RollingStats`), same one-per-concern reasoning that keeps this
        # OFF `_shadow_tracker`: that tracker is uid-keyed and per-request, this is one scalar per
        # step, batch-wide. Same dial as the tracker -- there is nothing to time when nothing drafts.
        self._draft_timing = None
        self._draft_timing_fired = False
        # #801 bullet 7: how many ELIGIBLE forwards the loose gate checks (0 = never). ⚠ A count,
        # not a flag, and deliberately not `weight.py`'s business: it gates a research probe, not
        # what is loaded. `headcheck.py` declines batches it cannot check and does not spend one.
        self._mtp_check_left = int(os.getenv("FREETOKEN_MTP801_HEADCHECK", "0") or 0)
        # round 5 bullet 6: `t_draft`, captured. `_draft_graph` is built by `engine.py`'s bring-up
        # hook (`mtp_capture_draft_graph`), not here -- capturing needs the real weights AND a
        # real CUDA device, neither of which exist yet at `__init__` time. `_draft_capture` is
        # read once, same reason as `_mtp_run`/`_shadow_tracker`: a captured decode graph and an
        # eager step must never disagree about which mode built the model. `_draftcheck_left` is
        # the captured-vs-eager diff's own countdown, same shape as `_mtp_check_left` above -- a
        # research probe, not what is loaded.
        self._draft_graph = None
        self._draft_capture = False
        self._draft_capture_fallback_fired = False
        self._draftcheck_left = int(os.getenv("FREETOKEN_MTP801_DRAFTCHECK", "0") or 0)
        # ⛔⛆ `FREETOKEN_MTP801_MASSCHECK=N` -- the GREEDY-arm agreement check bullet 7b's design
        #   promised and round 5 never wired. α reads 0.5346 by token equality and 0.7010 by
        #   `draft.py::acceptance_mass`, and it is the SECOND that drives the +33.1 % projection,
        #   so anything wrong in `acceptance_mass` moves the gate. Under the served sampler the
        #   two quantities genuinely differ and neither can check the other; under greedy both
        #   filtered distributions are one-hot and they must be the same number to the bit. That
        #   is the only place the round can tell -- and `analyse_alpha_801.py`'s own docstring
        #   claimed the check for four commits while this method never computed a mass on a
        #   greedy step at all, so the only thing that could fire was an assertion that the
        #   check's INPUT was absent.
        # ⛔ A COUNT, not a dial -- the third instance of the exception
        #   `test_load_801.py::test_the_model_reads_no_load_flag_of_its_own` documents, and it
        #   decides nothing about what is loaded or run. It is budgeted rather than a boolean
        #   because the greedy branch has to take `draft_next_token_ids_and_logits` and
        #   `filtered_probs` has to allocate a vocab-wide one-hot row -- exactly the cost this
        #   method skips the mass on a greedy arm to avoid.
        self._masscheck_left = int(os.getenv("FREETOKEN_MTP801_MASSCHECK", "0") or 0)
        # ⛔⛆ round 6 bullet 7: `FREETOKEN_MTP801_SPECCHECK=N` -- do the two TP ranks decide a
        #   verify step identically? Agreement is STRUCTURAL today (`Engine.__init__` seeds every
        #   rank with `torch.manual_seed(42)`, and the triton sampler's own `_UGEN` generator
        #   starts from torch's default seed), but it is positional: it survives only while both
        #   ranks draw the same number of values in the same order, and nothing re-anchors them.
        #   A divergence used to cost one token of discarded rank-1 text; in a verify step it
        #   desynchronises `device_len`, the ranks build different batch shapes, and the row wedges
        #   on NCCL's 60 s watchdog with the cause scrolled away (#871's shape). `spec.py`'s own
        #   header has the mechanism in full.
        # ⛔ A COUNT, not a dial -- the FOURTH of that shape ("last" is struck: bullet 9l adds a
        #   fifth below), and
        #   `test_load_801.py::test_the_model_reads_no_load_flag_of_its_own` asserts the set
        #   exactly so a fifth arrives deliberately or not at all. It decides nothing about what
        #   is loaded or run. Budgeted rather than a boolean because the check is a host sync plus
        #   a gloo collective, which is exactly the per-step cost #912 spent a round removing.
        self._speccheck_left = int(os.getenv("FREETOKEN_MTP801_SPECCHECK", "0") or 0)
        # ⭐⭐ round 6 bullet 9l: `FREETOKEN_MTP801_CHECKROW=N` -- on the first N SPECULATING steps,
        #   run a PLAIN decode forward for the same requests at the same positions and print the
        #   T=2 check row's argmax beside it. Load 6 served clean and emitted garbage from char 8,
        #   which is the FIRST verify step, and under greedy a rejection commits the check row's
        #   own argmax ⇒ the check row is wrong and no reading of the output text says which part
        #   of the T=2 forward made it so. `spec.py`'s bullet-9l header has the two-arm reasoning
        #   and the three observer effects the reference forward has to undo.
        # ⛔ A COUNT, not a dial -- the FIFTH of that shape, and "last" was wrong twice now. It
        #   decides nothing about what is loaded or run. Budgeted rather than a boolean because
        #   its cost is a SECOND FORWARD of the whole backbone: an instrument, never inside a
        #   decode figure, exactly as the host sync and the eager draft are banked.
        self._checkrow_left = int(os.getenv("FREETOKEN_MTP801_CHECKROW", "0") or 0)
        # ⭐⭐ round 6 bullet 9x: `FREETOKEN_MTP801_GDNCHECK=N` -- bullet 9w's GDN OUTPUT oracle.
        #   On the first N SPECULATING steps, the FIRST LOCAL GDN layer's fused-kernel output is
        #   compared token by token against `gdn_reference.recurrent_gated_delta_rule`, from the
        #   same entry state on the same inputs. 9t's finding 1 is why it exists: that function
        #   returns `(output, state)` and every call site in this round is `_, state = …`, so the
        #   round has checked what a verify forward STORES and never what it RETURNS -- which is
        #   the thing loads 6/10 and 9t's check row all say is wrong.
        # ⛔ A COUNT, not a dial -- the SIXTH of that shape, and the gate that asserts the set
        #   exactly asked for it to arrive deliberately. It decides nothing about what is loaded
        #   or run. Budgeted for two reasons: the reference is a python-loop recurrence over every
        #   token of every request, AND it forces the step EAGER (see
        #   :meth:`mtp_gdncheck_pending`) -- so a load that sets it is not a load whose decode
        #   number may be banked, exactly as CHECKROW's is not.
        # ⛔⛆ round 6 bullet 9aa: THE LAYER SET RIDES ON THIS SAME VALUE -- `N` or `N@a,b,c`
        #   (local GDN indices). It is NOT a seventh `os.getenv`, and that is deliberate: all six
        #   above are COUNTS, which decide nothing about what is loaded or run, and the gate that
        #   pins the set exactly asks for a seventh to arrive deliberately or not at all. A layer
        #   set is a DIAL, so giving it its own flag would break that invariant twice over --
        #   while a count and the layers it is spent on are one instrument's one setting.
        #   ⭐ `4` alone still means exactly what loads 11 and 12 ran: four steps, first local
        #   layer. Every bank written before 9aa keeps its meaning.
        _gdn = (os.getenv("FREETOKEN_MTP801_GDNCHECK", "0") or "0").strip()
        _gdn_n, _, _gdn_layers = _gdn.partition("@")
        self._gdncheck_left = int(_gdn_n or 0)
        self._gdncheck_layers = frozenset(
            int(x) for x in _gdn_layers.split(",") if x.strip()
        ) or frozenset({0})
        # ⛔⛆ PRINTED, because a layer the model never reaches is INVISIBLE otherwise. A set
        #   naming a local index this rank does not own simply never fires, and the bank would
        #   carry no report for it -- which reads identically to "that layer was clean". Naming
        #   the REQUESTED set once at bring-up lets a reader diff it against the layers that
        #   actually reported. ⭐ #866's shape: a vacuous pass that looks like a result.
        if self._gdncheck_left > 0:
            print(
                f"[#801] gdncheck armed for {self._gdncheck_left} step(s) over local GDN layer(s)"
                f" {sorted(self._gdncheck_layers)}",
                file=_ft801_sys.stderr, flush=True,
            )
        # ⭐⭐ round 6 bullet 9af: `FREETOKEN_MTP801_GDNDUMP=N` -- write the first N REPORTED GDN
        #   cells' tensors to disk, so `repro_gdn_t2_801.py --from` can replay them with no GPU and
        #   no model. 9ae acquitted the kernel (every arm exact to the bf16 quantum on random
        #   tensors), which leaves the live DATA, the live ENVIRONMENT and the INSTRUMENT -- and
        #   six GDNCHECK sweeps have measured the same number, so only a dump separates them.
        # ⛔ A COUNT, the SEVENTH of that shape and deliberately not an eighth kind: it decides
        #   nothing about what is loaded or run, and the DESTINATION is `spec.GDN_DUMP_DIR`, which
        #   the arm script mounts. A `GDNDUMP=/some/dir` would be the first DIAL among these reads
        #   -- the invariant bullet 9aa went out of its way to keep when it rode the layer SET on
        #   an existing count rather than add a flag of its own.
        self._gdndump_left = int(os.getenv("FREETOKEN_MTP801_GDNDUMP", "0") or 0)
        if self._gdndump_left > 0:
            from .spec import GDN_DUMP_DIR

            # ⛔⛆ CHECKED AT BRING-UP, NOT AT THE FIRST CELL. Without the mount the first dump
            #   raises ~4 verify steps into a 25-minute load, or -- worse, if it were made
            #   tolerant -- the load serves to the end and writes nothing, which reads exactly
            #   like "no cell qualified". #866's shape: a vacuous pass that looks like a result.
            if not os.path.isdir(GDN_DUMP_DIR):
                raise RuntimeError(
                    f"[#801] FREETOKEN_MTP801_GDNDUMP={self._gdndump_left} but {GDN_DUMP_DIR} is "
                    "not a directory. The arm script mounts it writable; a load armed without it "
                    "would spend 25 minutes and write nothing."
                )
            print(
                f"[#801] gdndump armed for {self._gdndump_left} reported cell(s) -> "
                f"{GDN_DUMP_DIR}",
                file=_ft801_sys.stderr, flush=True,
            )
        # ⭐⭐ round 6 bullet 9bt: `FREETOKEN_MTP801_BONUSROW=N` -- on the first N SPECULATING
        #   steps, run a PLAIN decode forward at the BONUS row's position and print its argmax
        #   beside the T=2 forward's. Load 27 EXONERATED row 0 (1,198/1,200 rows bit-identical to
        #   a plain decode, 0 argmax disagreements), and under greedy that settles every token row
        #   0 decides -- on a rejection the committed token IS its argmax, on an acceptance the
        #   draft was accepted BECAUSE it equals its argmax. ⇒ the only emitted token left is the
        #   bonus token from row 1, and that row has NO T=1 counterpart anywhere on the arm: on an
        #   accept `cached_len` jumps past its position, on a reject the next step decodes it with
        #   a different input token. `spec.py`'s bullet-9bt header has the reasoning and the three
        #   things the bonus forward has to advance rather than rewind.
        # ⛔ A COUNT, not a dial -- the EIGHTH of that shape, and the gate that asserts the set
        #   exactly asked for it to arrive deliberately. It decides nothing about what is loaded
        #   or run. Budgeted for CHECKROW's reason: its cost is a SECOND FORWARD of the whole
        #   backbone, so a load that sets it is not a load whose decode number may be banked.
        self._bonusrow_left = int(os.getenv("FREETOKEN_MTP801_BONUSROW", "0") or 0)
        # ⭐⭐⭐ round 6 bullet 9cb: `FREETOKEN_MTP801_HIDDENCHECK=N` -- on the first N FORWARDS,
        #   print each of `spec.FT801_HIDDENCHECK_LAYERS`'s seven block tensors as one exact
        #   integer per ROW, on BOTH arms.
        #   9bz localised the arms' first divergence to inside layer 0's block and could not open
        #   it: the linear-state pools are a COMMITTED-STATE detector and the two candidates -- the
        #   GDN mixer's output row at T=2, the MoE/MLP at two rows -- are HIDDEN states.
        # ⭐⭐ #983 WIDENED THE LAYER LIST, NOT THE DIAL: the env var is unchanged and is still a
        #   COUNT of forwards. Which layers it arms is a SOURCE constant, so a banked log's own
        #   `tensors` field is what tells a reader how many layers it holds -- and the line printed
        #   below names them, so a log can be read without its overlay to hand.
        # ⛔ A COUNT, not a dial -- the NINTH of that shape, and `test_load_801.py` asserts the set
        #   EXACTLY so that it arrived deliberately. It decides nothing about what is loaded or run.
        # ⭐⭐ **AND IT IS THE FIRST OF THE NINE THAT IS MEASUREMENT-SAFE IN THE GRAPH SENSE.**
        #   The countdown gates the PRINT, never the write: the write is armed at BUILD time so it
        #   is captured and replayed, and `can_use_cuda_graph` is never asked to decline. ⛔ The
        #   host read still SYNCS once per printed forward, so the decode figure of a HIDDENCHECK
        #   load is not the arm's -- the launcher says so, for both arms, exactly as STATECHECK's.
        self._hiddencheck_left = int(os.getenv("FREETOKEN_MTP801_HIDDENCHECK", "0") or 0)
        # ⛔⛆ PRINTED AT BRING-UP, for `gdncheck`'s reason: a dial whose buffer never got reserved
        #   emits nothing, and an empty bank reads exactly like a load that ran clean (#866).
        if self._hiddencheck_left > 0:
            print(
                f"[#801] hiddencheck armed for {self._hiddencheck_left} forward(s) over layers"
                f" {list(FT801_HIDDENCHECK_LAYERS)} x {len(FT801_HIDDENCHECK_NAMES)}"
                f" block tensors = {len(FT801_HIDDENCHECK_LAYERS) * len(FT801_HIDDENCHECK_NAMES)}"
                f" slots",
                file=_ft801_sys.stderr, flush=True,
            )
        # ⛔⛆ The mixed-batch notice is LATCHED, not counted, and it starts down. `bonus_armed`
        #   declines a batch where any request forwarded one token (#949's `mr=4`), and a load
        #   whose every step is mixed would otherwise bank NOTHING -- which reads exactly like a
        #   load that ran clean. #866's shape again.
        self._bonusrow_mixed_notice = False
        # ⛔ Set by `mtp_set_rank_group` at bring-up, from the ENGINE's own gloo `tp_cpu_group`.
        #   Never reached for: on the deployed arm (`--disable-pynccl`)
        #   `torch.distributed.group.WORLD` is an NCCL group, so a default would be right on one
        #   configuration and silently wrong on the one this round measures. Left `None`, the
        #   check DECLINES and says so rather than reporting a vacuous pass (#866's trap).
        self._speccheck_gather = None
        self._speccheck_tp_size = 1
        self._speccheck_tp_rank = None
        # round 5 bullet 7b: α as a RATE, and `t_draft` as TWO windows.
        # ⭐ `_accept_counter` folds bullet 3's per-step verdicts into one row per REQUEST
        #   (`accept.py` -- α is content-dependent, so a single pooled number would report the
        #   middle of a 40-point spread as if it were a measurement).
        # ⭐ `_draft_timing` stays the EAGER window bullet 5 built; `_draft_timing_captured` is
        #   bullet 6's, and `_draft_alternate` interleaves the two within ONE load so their ratio
        #   carries no load lottery (`weight.py::mtp_draft_alternate_enabled`).
        # ⭐ `_pending_draft_logits` is the acceptance mass's own one-step lag, held here rather
        #   than inside `ShadowTracker`: that tracker is plain Python by an asserted property and
        #   these are torch rows. Same rebuild-from-the-live-batch rule, same slot-reuse safety.
        # ⛔⛆ round 5 `/code-review`: THREE windows, not two. `_draft_timing_captured` and
        #   `_draft_timing` are both the GREEDY bs=1 shape (`capture.py::DraftGraphRunner` bakes
        #   an argmax into the graph, so a captured step is greedy by construction) and are the
        #   only pair whose ratio says anything about CAPTURE. The served arm's draft call pays
        #   two vocab-wide sorts and a `multinomial` on top and keeps the head's logits for the
        #   mass; pooling it into the eager window -- which is what bullet 7d banked -- prices
        #   capture and greedy-versus-sampled as one number, and the probe runs the served tier
        #   LAST, so the 256-sample rolling window held little else by the end of the run.
        self._accept_counter = None
        self._draft_timing_captured = None
        self._draft_timing_sampled = None
        self._draft_alternate = False
        self._draft_alternate_captured_next = False
        self._pending_draft_logits: dict = {}
        # ⭐ round 6 bullet 8: the draft ids the NEXT step verifies, uid-keyed and replaced
        #   wholesale each step exactly like `_pending_draft_logits` above. Held here rather than
        #   inside `ShadowTracker` for the same reason the logits are: that tracker is plain
        #   Python by an asserted property, and this is the verify path's own one-step lag.
        self._pending_draft_ids: dict = {}
        # round 6 bullet 8: how many verify forwards this load has run -- the `step` the
        # cross-rank check reports itself under, so two ranks' reports can be lined up.
        self._verify_steps = 0
        if mtp_load_enabled():
            from .mtp import Qwen4ExpMTPHead

            self.mtp = Qwen4ExpMTPHead(config, self.model.embed_tokens)
            self._mtp_run = mtp_run_enabled()
            self._verify_run = mtp_verify_enabled()
            if mtp_shadow_enabled():
                from .shadow import ShadowTracker
                from .timing import RollingStats

                from .accept import AcceptanceCounter

                self._shadow_tracker = ShadowTracker()
                self._accept_counter = AcceptanceCounter()
                self._draft_timing = RollingStats()
                self._draft_timing_captured = RollingStats()
                self._draft_timing_sampled = RollingStats()
                self._draft_capture = mtp_draft_capture_enabled()
                self._draft_alternate = mtp_draft_alternate_enabled()
                print(
                    "[#801] shadow tracker attached (FREETOKEN_MTP801_SHADOW=1)",
                    file=_ft801_sys.stderr,
                    flush=True,
                )
        super().__init__()

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run the vision tower. Returns ``[num_soft_tokens, hidden]`` in the text hidden space.

        ``pixel_values``: ``[num_images, num_patches, in_ch*temporal*patch**2]``;
        ``image_position_ids``: ``[num_images, num_patches, 2]`` with ``(-1, -1)`` padding.
        ⭐ There is no separate projector op here -- the tower's ``merger`` already lands in the
        text hidden size, so its output is returned unchanged.
        """
        return self.visual.forward(pixel_values, image_position_ids)

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    # ── #801 bullet 7: the three hooks the ENGINE asks a model for, so its pools can be sized
    #   for a draft head. ⛔ Duck-typed, exactly like `load_host_tables`, `prepare_for_runtime`,
    #   `make_offload_moe_cache` and `_iter_offload_moe_layers` already are: `engine/engine.py`
    #   imports nothing qwen4exp-specific and every other model answers by not having them.
    #   ⚠ All three are keyed on `self.mtp is None`, not on the environment: the loader's own
    #   predicate decided that once, in __init__, and a second read here is how they diverge.

    def mtp_pool_model_config(self, model_config: ModelConfig) -> ModelConfig:
        """The model config the engine's POOLS must be sized from; unchanged when no head is built.

        ⛔ Applied by the engine AFTER `create_model` and AFTER the expert-bank load -- see
        `mtp.py::pool_config` for why it must reach neither.

        ⛔⛆ #801 round 5 bullet 7a: the MoE half of the answer comes from the head's OWN offload
        walk (`Qwen4ExpMTPHead.num_offload_moe_layers`), not from `weight.py::mtp_head_dtype`.
        Under bullet 1's resident-bf16 dial the head has no `OffloadMoELayer` at all, so the
        pools must NOT be told to expect a 49th MoE layer -- while the attention half grows under
        both dials. Reading the built head rather than the environment is the same rule the
        comment above states for all three hooks, and here it is load-bearing: the two would
        diverge only on the box, inside `attach_offload_moe_cache`'s assertion.
        """
        if self.mtp is None:
            return model_config
        from .mtp import pool_config

        return pool_config(
            model_config,
            len(self.mtp.layers.op_list),
            num_head_moe_layers=self.mtp.num_offload_moe_layers,
        )

    def mtp_expert_source_banks(
        self, backbone_config: ModelConfig, quant_format: str
    ) -> dict | None:
        """The head's routed experts, as extra LAYERS on the engine's own bank sources.

        ``{bank name: [one [E, ...] tensor per head MoE layer]}`` -- the shape
        `load_nvfp4_expert_sources` returns, so the engine appends them to the backbone's lists and
        the offload cache sees a 49th layer with nothing special about it.

        ⛔⛆ The geometry comes from the BACKBONE's config on purpose: ``num_experts``,
        ``hidden_size`` and ``moe_intermediate_size`` are the head's too (it is one more layer of
        the same MoE), while ``num_layers`` is not, and handing it the pool config would ask the
        bank reader for 49 layers of a one-layer bank.

        ⛔ ``nvfp4`` only. Round 2's bank holds the NATIVE ModelOpt rows, which is what
        ``--nvfp4-backend triton`` (the deployed row) leaves the backbone's banks in. A marlin or
        b12x run has already repacked the backbone's rows into a tiled layout this bank is not in,
        and appending it would hand the GEMM 512 experts of garbage at exactly the right shape.
        """
        if self.mtp is None or self.mtp.num_offload_moe_layers == 0:
            # ⛔ #801 round 5 bullet 7a: a resident-bf16 head (round 5 bullet 1) already holds its
            #   512 experts as ordinary on-card tensors. Appending an NVFP4 source bank for them
            #   would add a 49th bank layer nothing ever reads, at exactly the cache-slot cost
            #   that arm exists to avoid paying twice. ⚠ `engine.py` also gates the call on the
            #   same MoE-layer delta; this is the second half of one predicate, not a duplicate --
            #   the hook is duck-typed and a future caller does not have to know the first half.
            return None
        if quant_format != "nvfp4":
            raise ValueError(
                f"llm-server #801: the draft head's expert bank is native NVFP4, but this run's "
                f"banks are {quant_format!r}. Serve the head with --nvfp4-backend triton, or "
                f"repack the bank for {quant_format!r} first."
            )
        from .weight import load_mtp_expert_source_banks, mtp_bank_path

        return load_mtp_expert_source_banks(
            mtp_bank_path(), backbone_config, num_mtp_layers=len(self.mtp.layers.op_list)
        )

    def mtp_reserve_graph_buffers(self, max_rows: int, max_tokens: int) -> None:
        """Size the tap's static output buffer BEFORE anything captures (see
        `Qwen4ExpModel.reserve_multi_stream`). No-op without a head.

        ⛔⛆ **TWO ARGUMENTS BECAUSE THERE ARE TWO CURRENCIES, AND 9cb SPENT ONE FOR THE OTHER.**
        ``max_rows`` is a **REQUEST** count -- it is the TAP's unit, because :meth:`multi_stream_rows`
        selects one row per request on prefill and a decode forward has one row per padded request
        -- and `engine.py` hands it
        ``max(max(cuda_graph_bs), max_running_req + 1)``, at most a few hundred. ``max_tokens`` is
        the scheduler's prefill TOKEN budget. ⛔ HIDDENCHECK's buffer is indexed by TOKEN ROW, so
        9cb sizing it by ``max_rows`` made a prefill chunk of 4096 rows write into a buffer ~160
        wide: `RuntimeError: The expanded size of the tensor (160) must match the existing size
        (4096)`, on BOTH ranks, at bring-up, before anything served.
        ⛔ **NEITHER ARGUMENT HAS A DEFAULT**, deliberately: this overlay and `engine.py`'s are
        versioned together, and a drift between them must be a loud `TypeError` at bring-up rather
        than a buffer that is quietly the wrong size again.

        ⛔⛆ **HIDDENCHECK'S BUFFER IS RESERVED ABOVE THE `self.mtp is None` GUARD, AND THAT IS THE
        WHOLE POINT OF A CROSS-ARM DIAL.** The guard makes everything below it dead on the CONTROL
        arm, which loads no head -- so a buffer reserved under it would arm the verify arm only,
        and 9bx already paid for that lesson once ("the launcher passes every dial to the verify
        arm only, so no load can produce a control-side ledger"). An instrumented verify arm beside
        a bare control arm is half a measurement, and this dial exists for nothing else.
        ⭐ The HOOK itself is reachable on both arms -- `engine.py` calls it by `hasattr`, and the
        method exists whether or not a head was loaded. Only its BODY was ever MTP-gated.
        """
        if self._hiddencheck_left > 0:
            # ⭐ The VERIFY width is the MODEL's knowledge, so the model applies it: a speculating
            #   decode step forwards `mtp_verify_width` rows PER request, so the bound can never
            #   be below `width * max_rows` however small the prefill budget is. `engine.py` stays
            #   model-agnostic and passes only the token budget it owns.
            self.model.reserve_hiddencheck(max(max_tokens, self.mtp_verify_width * max_rows))
        if self.mtp is None:
            return
        self.model.reserve_multi_stream(max_rows)

    def mtp_hiddencheck_take(self) -> "torch.Tensor | None":
        """Layer 0's fingerprint buffer while the print budget lasts, else ``None``. #801 r6 b9cb.

        ⭐ Duck-typed, the shape `mtp_gdncheck_pending` and `mtp_reserve_verify_buffers` already
        take: `engine.py` stays model-agnostic and a model without this name never emits a row.
        ⛔ It SPENDS a unit, so there is exactly one caller. The BUDGET is the print's, never the
        write's -- the write is unconditional once armed, because a python branch inside a captured
        graph is baked at capture time and cannot count anything.
        """
        if self._hiddencheck_left <= 0:
            return None
        buf = self.model.hiddencheck_buffer
        if buf is None:
            return None
        self._hiddencheck_left -= 1
        return buf

    # ── #801 round 6 bullet 6: the verify graph's three duck-typed hooks ────────────────────
    # ⭐ `engine/graph.py` is model-agnostic and must stay so -- the same rule `engine.py`'s own
    #   `hasattr(self.model, "mtp_shadow_step")` hooks follow. Everything the runner needs to know
    #   about a verify step arrives through these three: how WIDE a step is, how many of them to
    #   capture for, and how to reserve what a capture must not allocate.

    @property
    def mtp_verify_width(self) -> int:
        """Rows a verify step forwards per request: the committed token plus the draft.

        ⛔ Not a free parameter. `mtp.py` is a SINGLE-head MTP, so it drafts exactly one token per
        step; a second head would be a different checkpoint, not a different constant. 1 (the
        value with the dial off, and with no head at all) is today's decode step, and a runner
        that reads 1 captures nothing extra and routes nothing differently.
        """
        return 2 if (self.mtp is not None and self._verify_run) else 1

    @property
    def mtp_verify_graph_max_bs(self) -> int:
        """The largest batch size the VERIFY graph set is captured for (`weight.py`'s dial,
        default 4 = the deployed row's `-np4-`). ⛔ A cap and not the T=1 set's `max_graph_bs`
        because :meth:`mtp_reserve_verify_buffers` below costs ~1.5 MiB per (request, step) per
        rank in EACH of the 36 GDN layers."""
        from .weight import mtp_verify_graph_max_bs

        return mtp_verify_graph_max_bs() if self.mtp_verify_width > 1 else 0

    def mtp_gdncheck_pending(self) -> bool:
        """Does bullet 9w's GDN-output oracle still owe this load a report? (#801 r6 b9x.)

        ⛔⛆ **WHY `engine/graph.py` ASKS AT ALL, AND IT IS 9q's LESSON ONE BULLET LATER.**
        `GraphRunner.replay` never calls the model's python forward -- it refills the static
        buffers and calls `g.replay()` -- and `can_use_cuda_graph` admits a verify batch from the
        very first step. ⇒ an instrument in `gdn.py` is silent on exactly the steps that matter
        unless the budgeted steps run EAGER. This is what makes them.

        ⛔ A METHOD, not a property, and the round's SIXTH duck-typed name on this model. The
        runner binds these names ONCE at bring-up (`getattr(model, "mtp_restore_memos", None)`),
        so a property would be evaluated while the budget was still whole and the row would then
        run eager for the life of the load -- #701's tax on every speculating step, silently.

        ⭐ The decline is a real cost and it is named here rather than discovered as a tok/s
        number: an instrumented load measures the EAGER verify path.
        """
        return int(getattr(self, "_gdncheck_left", 0) or 0) > 0

    def mtp_reserve_verify_buffers(self, max_bs: int, width: int) -> None:
        """Reserve every GDN layer's intermediate-states buffer BEFORE the first verify capture.

        ⛔⛆ An allocation made DURING ``torch.cuda.graph(...)`` comes out of THAT graph's private
        pool -- right on the eager warm-up, wrong on every replay. Bullet 5 allocated this buffer
        per forward, per layer, which is a fresh address every step; `gdn.py::_verify_buffer`
        asserts against allocating one mid-capture, and this is what makes that assertion never
        fire. Same rule, and the same wording, as `mtp_reserve_graph_buffers` above.

        ⚠ Sized from the LIVE pool's own recurrent-state rows, so the head/backbone dtype and the
        per-rank head split are whatever the load actually built -- not restated here.

        ⭐⭐ **THE LINE IS THE MEASUREMENT BULLET 5 DEFERRED TO BULLET 9.** `gdn.py` states the
        cost -- ~1.5 MiB per (request, step) per rank, ~108 MiB at bs=1 T=2 across 36 layers --
        from the geometry, and nothing has ever read it off a load. ⛔ `bytes` is summed from the
        buffers that were ACTUALLY reserved (a mixer that declined leaves none, so the number
        cannot overstate what was allocated); `free_delta` is the driver's own free-memory move
        across the loop and is INFORMATIONAL ONLY -- torch's caching allocator can satisfy the
        whole reservation out of its existing pool and report no move at all, which is not the
        same as having cost nothing.
        """
        from freetoken.core import get_global_ctx

        pool = get_global_ctx().linear_state_pool
        if pool is None:
            return
        on_gpu = torch.cuda.is_available()
        free_before = torch.cuda.mem_get_info()[0] if on_gpu else 0
        reserved = []
        for layer in self.model.layers.op_list:
            mixer = getattr(layer, "linear_attn", None)
            reserve = getattr(mixer, "reserve_verify_states", None)
            if reserve is None:
                continue
            reserve(max_bs, width, pool.recurrent_states[pool.local_index(mixer.layer_id)])
            buffer = getattr(mixer, "_verify_states", None)
            if buffer is not None:
                reserved.append(buffer)
        if not reserved:
            return
        total = sum(b.numel() * b.element_size() for b in reserved)
        free_after = torch.cuda.mem_get_info()[0] if on_gpu else 0
        print(
            f"[#801] verify buffers: layers={len(reserved)} max_bs={max_bs} width={width} "
            f"shape={'x'.join(str(n) for n in reserved[0].shape)} "
            f"dtype={reserved[0].dtype} bytes={total} "
            f"mib={total / (1 << 20):.1f} free_delta_mib={(free_before - free_after) / (1 << 20):.1f}",
            file=_ft801_sys.stderr,
            flush=True,
        )

    def mtp_set_rank_group(self, group, tp_size: int, tp_rank: int) -> None:
        """Hand the verify step the engine's own **gloo** TP group (llm-server #801 bullet 7).

        Called once at engine bring-up, duck-typed like every other #801 hook. ⛔ The group is
        the engine's `tp_cpu_group`, which is gloo on BOTH branches of `_init_communication`
        (`--disable-pynccl` makes it an explicit `new_group(backend="gloo")`, otherwise it is a
        WORLD group itself initialised with `backend="gloo"` while pynccl carries the device
        traffic). Passing it rather than letting `spec.py` look one up is what keeps the
        cross-rank check off the NCCL path on every configuration.

        ⭐ The gather is built EAGERLY, and only when the budget is non-zero: a wiring error then
        fails at bring-up rather than 20 minutes into a load, and a row with the check off pays
        and risks nothing at all.
        """
        self._speccheck_tp_size = int(tp_size)
        self._speccheck_tp_rank = int(tp_rank)
        if self._speccheck_left <= 0 or self._speccheck_tp_size <= 1:
            return
        from .spec import gather_over

        self._speccheck_gather = gather_over(group)
        print(
            f"[#801] speccheck armed for {self._speccheck_left} step(s) over the engine's gloo "
            f"group (rank {self._speccheck_tp_rank}/{self._speccheck_tp_size})",
            file=_ft801_sys.stderr,
            flush=True,
        )

    def mtp_capture_draft_graph(
        self, attn_backend: "BaseAttnBackend", stream: torch.cuda.Stream, dummy_req: "Req"
    ) -> None:
        """`t_draft`, captured (llm-server #801 round 5 bullet 6): capture the head's own forward
        as a dedicated bs=1 CUDA graph, once, at engine bring-up. No-op without a head, with the
        capture dial off, or (defensively) if called twice.

        ⛔ Called AFTER `engine/graph.py::GraphRunner` has finished capturing and warming the
        BACKBONE's own decode graphs — `capture.py::DraftGraphRunner.capture`'s own docstring has
        the ordering rule. ⚠ `self.model.multi_stream_buffer` must already be allocated by the
        time this runs: `mtp_reserve_graph_buffers` above is called BEFORE `GraphRunner`
        construction in `engine.py`'s bring-up sequence, this hook AFTER it.
        """
        if self.mtp is None or not self._draft_capture or self._draft_graph is not None:
            return
        from .capture import DraftGraphRunner

        R = self.model.multi_stream_buffer
        assert R is not None, "#801: mtp_reserve_graph_buffers must run before this hook"
        self._draft_graph = DraftGraphRunner(R.device)
        self._draft_graph.capture(self.mtp, self.lm_head, attn_backend, stream, dummy_req, R)
        print(
            "[#801] draft graph captured (FREETOKEN_MTP801_DRAFT_CAPTURE=1)",
            file=_ft801_sys.stderr,
            flush=True,
        )

    def mtp_shadow_step(
        self,
        sampled_ids: torch.Tensor,
        batch: Batch,
        sampling_args: "BatchSamplingArgs",
        logits: torch.Tensor | None,
    ) -> dict:
        """The speculative-decoding step (llm-server #801), delegated to `spec.py`.

        ⛔ A BARE delegate, early return included. Round 5 grew this method to ~140 lines owning
        capture eligibility, the interleave phase, two countdowns, three timing windows, the
        acceptance-mass lag, the tracker and the reporting; round 5's `/code-review` called it as
        Feature Envy and parked the extraction for round 6 bullet 1, which is this. ⭐ Keeping the
        `is_decode` / no-tracker guard here too would split the step's GATING from its BODY across
        two files -- the very split the extraction removes, wearing the costume of a fix.

        ⚠ The step's mutable state (countdowns, the three `RollingStats` windows, the tracker, the
        acceptance counter, the one-shot flags) is still attributes of THIS object, and
        `spec.shadow_step` reaches for them. That is deliberate and it is NOT resolved Feature
        Envy: 60 CPU gates address that state here, and re-homing it belongs with bullet 8's
        rewrite of the step, where a box load can gate the move. `spec.py`'s own docstring says so.

        ⛔ An in-function import, so `test_shadow_wiring_801.py` can monkeypatch
        `spec_module.shadow_step` and have the patch take.
        """
        from .spec import shadow_step

        return shadow_step(self, sampled_ids, batch, sampling_args, logits)

    def mtp_publish_drafts(self, ids: dict, logits: dict | None = None) -> None:
        """Hold drafts for the next step to verify (llm-server #801 round 6 bullet 8).

        ⚠ Called by `spec.shadow_step` on every decode step; exposed as a method so the wiring
        gates can stage a step without running a real draft forward.
        """
        from .spec import publish_drafts

        publish_drafts(self, ids, logits)

    def mtp_stage_verify(self, batch: Batch, token_pool: torch.Tensor):
        """Lay this decode batch out as a verify step, or ``None`` (llm-server #801 round 6
        bullet 8). The ONE hook `scheduler/scheduler.py` reaches for.

        ⛔ A BARE delegate, the shape bullet 1 established: the gating decision lives in
        `spec.py`, in one place, rather than being split between a guard in an every-row file and
        a body in this one. ⛔ An in-function import, so the wiring gates can monkeypatch
        `spec_module.stage_verify` and have the patch take.
        """
        from .spec import stage_verify

        return stage_verify(self, batch, token_pool)

    def mtp_publish_lengths(
        self, own_cached_len: int, committed: int, max_device_len: int
    ) -> tuple:
        """The per-PUBLISHED-token *length* verdict for one commit (llm-server #801 r6 b9bh).

        ⛔ A BARE delegate, the shape every hook here takes: the arithmetic lives in `spec.py`,
        beside the commit that advances the lengths it undoes, rather than in an every-served-row
        file. ⛔ An in-function import so the wiring gates can monkeypatch `spec_module`.

        ⛔⛆ **#801 r6 b9bn: ``own_cached_len`` IS THE DRAINED FORWARD'S OWN POST-COMMIT VALUE,
        NEVER `req.cached_len` READ LIVE.** The drain runs one forward late, so the live value
        already carries the NEXT forward's commit — 36 of 36 banked rows — and passing it makes the
        verify arm finish one generated token early. The caller reads it off
        ``batch.ft801_post_commit``, which `engine/engine.py::forward_batch` writes after both
        advance paths. ⭐ The parameter is NAMED for what it is so a future caller cannot hand it
        the live number without the rename showing up in the diff.

        ⚠ Returns a plain tuple of bools rather than the `PublishPlan` — `scheduler/scheduler.py`
        reaches this by duck-typed name and must not have to import a type from `models/`.
        """
        from .spec import publish_plan

        return publish_plan(
            own_cached_len=own_cached_len,
            committed=committed,
            max_device_len=max_device_len,
        ).length_flags

    def mtp_capture_memos(self, batch: Batch) -> dict:
        """What the forward `engine/graph.py` just captured recorded in python (llm-server #801
        round 6 bullet 9s). The fourth duck-typed name that file reaches the model through.

        ⛔ A BARE delegate, the shape every hook here takes: the enumeration of WHICH memos cross
        a replay lives in `spec.py`, beside the commits that consume them, rather than in the
        engine's model-agnostic graph runner.
        """
        from .spec import capture_memos

        return capture_memos(batch)

    def mtp_restore_memos(self, batch: Batch, memos: dict) -> None:
        """Re-bind a captured forward's memos onto the live batch a replay just ran (llm-server
        #801 round 6 bullet 9s). The fifth such name, and the one that closes bullet 9q.
        """
        from .spec import restore_memos

        restore_memos(batch, memos)

    def mtp_verify_step(
        self,
        batch: Batch,
        logits: torch.Tensor,
        sampling_args: "BatchSamplingArgs",
        *,
        page_size: int,
        linear_pool=None,
    ):
        """Verify, check, commit -- or ``None`` when this batch was never staged (llm-server #801
        round 6 bullet 8). The ONE hook `engine/engine.py` reaches for, in place of the sampler.

        ⛔⛆ It carries the round's HOST SYNC (operator decision (a), 2026-09-12). `spec.py`'s
        bullet-8 header has the decision and its three grounds in full; the short version is that
        `scheduler.py::overlap_loop` prepares batch N+1 before it drains batch N, so a
        data-dependent advance cannot be deferred, and the sync is paid only on steps that
        actually speculate.
        """
        from .spec import verify_step

        self._verify_steps += 1
        return verify_step(
            self,
            batch,
            logits,
            sampling_args,
            page_size=page_size,
            linear_pool=linear_pool,
            step=self._verify_steps,
        )

    @property
    def multi_stream(self) -> torch.Tensor | None:
        """The last forward's draft input (llm-server #801): the pre-final-mixer ``R`` at each
        request's last row, or ``None`` when no head is built.

        ⚠ A VIEW of `Qwen4ExpModel`'s static buffer, not a copy -- a captured decode graph refills
        it in place, so this is what a caller reads AFTER `GraphRunner.replay`, and it is stale
        the moment the next forward runs.
        """
        return self._multi_stream

    def mtp_tap_rows(self, rows: int) -> "torch.Tensor":
        """The tap's first ``rows`` rows, read where a CAPTURED REPLAY actually leaves them.

        ⛔⛆ **#801 round 6 bullet 9h -- THE EIGHTH SILENT SITE, and load 6 faulted the GPU on
        it.** :attr:`multi_stream` is a VIEW (`Qwen4ExpModel._tap_multi_stream` returns
        ``reserve_multi_stream(n)[:n]``) and it is assigned only inside `Qwen4ExpModel.forward` --
        **which a captured replay never runs**. The last EAGER forward of a load is the prompt
        PREFILL, where :meth:`Qwen4ExpModel.multi_stream_rows` selects each request's LAST row --
        one row per REQUEST. (⛔ Named as the method, not as the accessor it calls:
        `test_load_801.py::test_the_row_rule_is_spelled_exactly_once` counts that accessor's
        spellings in this file's raw source, and prose counts.) So the view stays ONE ROW WIDE
        for the whole generation while every replayed
        verify step writes `VerifyPlan.num_tokens` rows into the buffer underneath it. The rows are
        there; the view is too short to see them. ⇒ `spec.shadow_step`'s ``multi_stream[picked]``
        gathered row 1 out of a size-1 dim and ROCr aborted the queue, and its sibling
        ``multi_stream[:rows]`` will silently return ONE row at #949's ``mr=4``.

        ⭐ **The buffer, not the view, is the source of truth**, and that is the whole fix:
        `Qwen4ExpModel.multi_stream_buffer` is the memory a replay writes into, at its bring-up
        size. ⚠ The view is kept only as a FALLBACK for the case the buffer does not exist --
        `reserve_multi_stream` has never run, which on a served row is impossible
        (`mtp_reserve_graph_buffers` precedes `GraphRunner`) but is the normal state of a unit gate
        that builds the model by hand. ⛔ When both exist the buffer WINS: a view retained across
        a buffer GROWTH is the right length and freed memory, and no `IndexError` would catch it.

        ⛔⛆ The length check is not defensive decoration. An out-of-bounds gather here is a
        **device-side** fault that this stack cannot attribute: `AMD_SERIALIZE_KERNEL=3` plus
        `HIP_LAUNCH_BLOCKING=1` reproduced it byte-for-byte and logged 537 KB with not one python
        frame in it, because ROCr's queue-error callback `abort()`s the process. Any recurrence is
        worth a loud host error instead.

        ⚠ An ACCESSOR, not a dial -- `test_load_801.py::test_the_model_reads_no_load_flag_of_its_own`
        pins this file's env-var set at exactly four names and this reads none.
        """
        source = self.model.multi_stream_buffer
        if source is None:
            source = self._multi_stream
        assert source is not None and source.shape[0] >= rows, (
            f"#801: the tap has {0 if source is None else source.shape[0]} row(s), this step "
            f"needs {rows} -- the buffer is sized by `mtp_reserve_graph_buffers` at bring-up and "
            f"a replay writes every row of the step into it"
        )
        return source[:rows]

    @property
    def draft_timing(self) -> RollingStats | None:
        """`t_draft`, eager and GREEDY (llm-server #801 round 5 bullet 5): a rolling window of how
        long `mtp_shadow_step`'s real draft call has taken, in milliseconds. `None` without a head
        or with the dial off -- same gate as `_shadow_tracker`, there is nothing to time otherwise.

        ⛔⛆ **GREEDY ONLY since round 5's `/code-review`**, and that is what makes it comparable
        with :attr:`draft_timing_captured`: the captured graph bakes an argmax in, so a captured
        step is greedy by construction, and a ratio between these two is a statement about
        CAPTURE. The served arm's own call lands in :attr:`draft_timing_sampled` instead. Round 5
        banked "capture saves 27.2 %" off a version where the served steps came here too.
        """
        return self._draft_timing

    @property
    def draft_timing_sampled(self) -> RollingStats | None:
        """`t_draft` on the SERVED arm (llm-server #801, round 5's `/code-review`): the draft call
        as a sampled request actually pays for it -- `draft.py::sample_ids`'s two vocab-wide sorts
        and a `multinomial` on top of the head's forward, plus keeping the head's logits for the
        acceptance mass.

        ⭐ This is the population closest to what a DEPLOYED verify path would pay, and it is the
        one the round's projection does not use. ⛔ Never read a capture ratio against it: it
        differs from :attr:`draft_timing_captured` by the sampler as well as by the graph.
        """
        return self._draft_timing_sampled

    @property
    def draft_timing_captured(self) -> RollingStats | None:
        """`t_draft`, CAPTURED (llm-server #801 round 5 bullets 6/7b): the sibling window to
        :attr:`draft_timing`, holding only steps that actually replayed
        `capture.py::DraftGraphRunner`'s graph.

        ⚠ ``count == 0`` with the capture dial ON is a real finding, not an empty field: it means
        no step was ever eligible (bs > 1, not greedy, or the capture never happened). ⛔ Reading
        a mean off one window and comparing it with the other's without checking both counts is
        how a fallback run reads as a captured one.
        """
        return self._draft_timing_captured

    @property
    def acceptance(self) -> "object | None":
        """`accept.py::AcceptanceCounter` -- α per request (llm-server #801 round 5 bullet 7b).
        `None` without a head or with the shadow dial off, same gate as the tracker."""
        return self._accept_counter

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        if self.mtp is None:
            return self.lm_head.forward(self.model.forward(batch.input_ids, batch))
        hidden, self._multi_stream = self.model.forward(
            batch.input_ids,
            batch,
            want_multi_stream=True,
            draft_head=self.mtp if self._mtp_run else None,
        )
        if self._mtp_run:
            if self._mtp_check_left > 0:
                # ⛔ AFTER the real forward, on the same step: the gate diffs the engine's own two
                #   implementations of each ported kernel against the tensors this step produced.
                from .headcheck import run_headcheck

                self._mtp_check_left -= run_headcheck(self, batch)
        return self.lm_head.forward(hidden)


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
