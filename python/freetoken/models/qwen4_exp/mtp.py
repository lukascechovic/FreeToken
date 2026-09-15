"""Qwen3.8-Flash-Next multi-token-prediction (draft) head — llm-server #801.

⭐ **A NEW engine file**, `models/qwen4_exp/mtp.py`. There is no `.orig` beside it because image
`llm-server/freetoken-gfx1201:2026-09-09-agree-0022` ships nothing at this path: the deployed rows
have never loaded the checkpoint's 31 `mtp.*` tensors, and ADR 0011 §1 still records that absence
as intrinsic (amending it is the round's bullet 8, not this file's job). ⛔ Like every overlay in
this directory it is BIND-MOUNTED, not patched: it is in no image and in no Dockerfile ladder.

⭐ **What this is a port OF.** `Qwen4ExpMultiTokenPredictor` in vLLM, pinned by byte at
`vllm-project/vllm@60ad959b` (`raw/r4-bullet1-reference.txt`). ⛔ The two vLLM twins — amd and
nvidia — compute the SAME head; this follows the **amd** spelling because our
:class:`~freetoken.models.qwen4_exp.hc.GatedResidual` has no `combine_and_mix` and our layer
contract combines immediately, so amd's explicit broadcast add is the one our blocks already have.
It costs one unfused kernel and no arithmetic difference. ⛔ Do not re-derive that from the twins'
42-line diff; the diff reads like a fork and is not one.

⭐ **How it is gated.** `test_mtp_801.py` diffs this module against `head_ref_801.ReferenceHead`,
a plain-torch fp32 oracle with no engine imports, on the SAME weights, in a CPU-only container
(`./check_mtp_801.sh`). The gate is EXACT — float rounding, not round 2's 9.45 % bf16 floor —
which is only possible off-GPU, and is why `layers/norm.py` needed a CPU branch (see
`overlay/norm.py`'s marker).

⚠ **Scope: the head's arithmetic** (round 4, bullets 3 and 4) — the input fusion, one decoder
layer, the final mixer, and the head's two outputs. The loader that fills these tensors is bullet
5; the tap that produces ``R`` is bullet 6. ⛔ The head is constructed and fed, never scheduled:
there is no draft/verify/commit path here and no acceptance rate.

⛔⛆ **What this module knows about the head's geometry, and what it cannot know.** The checkpoint's
`config.json` carries an ``mtp`` block — ``{"layer_types": ["full_attention"],
"num_hidden_layers": 1, "rope_theta": 10000000}`` — and a top-level ``mtp_num_hidden_layers``.
`models/qwen4_exp/config.py::parse_config` parses NONE of it, so a `ModelConfig` cannot answer how
many layers the head has, what type they are, or what rope base they use. Until it does:

* the layer count is this module's ``num_mtp_layers`` argument, defaulted to the checkpoint's own 1;
* the type is taken as full attention, which is what the block says and what the 31 head tensors
  show (a `self_attn` with a QSA indexer, no GDN keys);
* ⛔ the rope base is the BACKBONE's, because that is all `config.rotary_config` holds. On this
  checkpoint the two are both 1e7, so it is arithmetically right — and
  `test_mtp_801.py` gates exactly that equality against the real `config.json`, so a checkpoint
  that ever separates them FAILS here instead of serving a head with the wrong frequencies.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated, MoELayer, OPList
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.models.qwen3_5_moe.moe import _SharedExpert

from .hc import GatedResidual
from .model import Qwen4ExpDecoderLayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

# ⛔⛆ #801 round 5 bullet 1: deliberately NOT `from .weight import mtp_head_dtype`. This file
#   has no other dependency on `weight.py` -- that is why `check_mtp_801.sh` mounts `mtp.py`
#   alone, without `weight.py`, over the image. Importing the loader's validating dial here
#   would make THIS file's tests depend on a sibling overlay it does not otherwise need, so the
#   raw env var is read again, locally: `weight.py::mtp_head_dtype` is still the one place a
#   GARBAGE value raises (during the real load's `iter_weights` pass); a typo read here just
#   falls through to the safe (nvfp4) branch until that raise fires.
_MTP801_HEAD_DTYPE_ENV = "FREETOKEN_MTP801_HEAD_DTYPE"


def draft_config(config: ModelConfig, num_mtp_layers: int) -> ModelConfig:
    """The model config as the head's own layers see it: full attention, ``num_layers`` and up.

    ⛔⛆ WHY THIS EXISTS AT ALL. vLLM builds the draft layer with an explicit
    ``layer_type="full_attention"``; our `Qwen4ExpDecoderLayer` DERIVES the type from the config,
    and `ModelConfig.attention_group_for_layer` raises ``Expected exactly one attention group for
    layer N, got 0`` for any id past the backbone's last. So the head's layer ids have to be owned
    by a group before the layer can be built at all.

    ⚠ It also bumps ``num_index_layers``, because the head's QSA layer needs an index-key slab of
    its own. ⛔ THE ENGINE'S POOLS ARE NOT SIZED FROM THIS OBJECT — it is built here, held here,
    and never handed back. Making the KV pool and the QSA backend agree that layer ``num_layers``
    exists is bullet 7's problem and it is NOT solved by this function.
    """
    first = config.num_layers
    layer_ids = tuple(range(first, first + num_mtp_layers))
    ple_ids = set(config.qwen4_args.ple_layer_ids)
    assert not ple_ids & set(layer_ids), (
        f"the head ships no PLE tensors, but {sorted(ple_ids & set(layer_ids))} is a PLE layer"
    )

    groups, found = [], False
    for group in config.attention_groups:
        if isinstance(group, FullAttentionGroupConfig):
            group = replace(
                group,
                layer_ids=group.layer_ids + layer_ids,
                num_index_layers=group.num_index_layers + num_mtp_layers,
            )
            found = True
        groups.append(group)
    assert found, "no full-attention group to put the head's layer in"
    return replace(config, attention_groups=tuple(groups))


def pool_config(
    config: ModelConfig, num_mtp_layers: int, *, num_head_moe_layers: int
) -> ModelConfig:
    """:func:`draft_config` plus the layer count the ENGINE'S POOLS are sized from (#801 bullet 7).

    ⛔⛆ WHY IT IS A SECOND FUNCTION. `draft_config` exists so the head's LAYER can be BUILT: it
    grafts the head's ids onto the full-attention group and bumps ``num_index_layers``, and bullet
    4 deliberately handed the result back to nobody. That is not enough to SERVE the head, because
    three engine pools size themselves off the model config and none of them reads the group alone:

    * `kvcache/__init__.py::create_kv_pool` passes ``num_layers=model_config.num_layers`` to
      `MHAKVCache`, whose layer map is ``[-1] * num_layers`` and which raises ``KV layer id 48
      outside [0, 48)`` for the head's own id. The K/V slab itself is sized by the GROUP
      (``layer_ids``), so `draft_config` already pays for it -- ``num_layers`` is the bound check.
    * `attention/qsa_sparse.py::QSASparseAttnBackend` builds ``_idx_slot`` from the same group, and
      `QSAKVCache._alloc_index_tiers` sizes the compressed slab and the pending ring from
      ``num_index_layers`` -- both `draft_config`'s.
    * `moe/offload_cache.py` is sized from ``num_moe_layers``, which is DERIVED
      (``num_layers - first_k_dense_replace``), so bumping ``num_layers`` is also what gives the
      head's `OffloadMoELayer` -- already found by `iter_offload_moe_layers`, since the head hangs
      off the model as an ordinary `BaseOP` -- a slab and a bank row to live in.

    ⛔ The head's MoE layer id is ``num_moe_layers`` exactly (``first_k_dense_replace == 0`` here,
    so the decoder id and the MoE id coincide), which is the bank layer round 2's NVFP4 bank was
    built at. A checkpoint with leading dense layers would separate the two and a gate fails.

    ⛔⛆ **``num_head_moe_layers`` IS NOT ``num_mtp_layers``, AND THAT IS THE WHOLE POINT** (#801
    round 5 bullet 7a). It is how many of the head's layers are OFFLOAD MoE layers -- i.e. how
    many `attach_offload_moe_cache`'s walk will actually find. Under the deployed nvfp4 dial the
    head's ``.mlp`` is an `OffloadMoELayer` and the two numbers are equal. Under round 5 bullet
    1's research-only bf16 dial it is `_ResidentBf16HeadMoE`, the walk finds one FEWER layer than
    the pools were told to expect, and `engine.py`'s ``assert len(layers) ==
    config.model_config.num_moe_layers`` fires -- twenty minutes into a load, with the whole
    checkpoint already on the cards. ⚠ Keyword-only and with NO default on purpose: the silent
    wrong answer is exactly the nvfp4 shape, so a caller that forgets it must fail at the call,
    not on the box.

    ⛔⛆ **Why the lever is ``first_k_dense_replace`` and not ``num_moe_layers``.** The latter is a
    derived ``@property`` (``num_layers - first_k_dense_replace``) on a frozen dataclass and
    cannot be set. So the count is moved by the only field that moves it. ⚠ That field's own
    MEANING -- "the first K decoder layers are dense" -- is NOT what is true here; what is true is
    that the head's layer, the LAST one, is not an offload MoE layer. The COUNT is what every
    consumer reads (`OffloadMoeCache(num_layers=...)`, the ``total_experts`` budget,
    `_resolve_cpu_layers`' index range, and the walk assertion), and the backbone's own MoE index
    space ``[0, 48)`` is unchanged by the bump -- `test_load_801.py::
    TestThePoolConfigIsDtypeAware::test_the_backbone_moe_index_space_is_untouched` pins exactly
    that, because it is the part a reader would reasonably doubt. ⛔ This config is applied AFTER
    `create_model`, so nothing ever builds a decoder layer from the bumped field.

    ⭐ The attention half is dtype-BLIND: ``num_layers`` and ``num_index_layers`` grow under both
    dials, because the head's ``self_attn`` is the same QSA layer either way and still needs its
    KV bound check and its index slab.

    ⛔⛆ THIS CONFIG MUST NEVER REACH `create_model` OR THE EXPERT-BANK LOADER. ``num_layers`` is
    what `models/qwen4_exp/model.py` builds ``range(config.num_layers)`` decoder layers from, and
    what `models/loader.py` validates the checkpoint's expert layers against
    (``expected_layers = set(range(config.num_layers))``). The engine applies this AFTER the model
    is built and keeps the backbone's config for `load_expert_banks`; the head's experts come from
    its own bank, not from the checkpoint index.
    """
    return replace(
        draft_config(config, num_mtp_layers),
        num_layers=config.num_layers + num_mtp_layers,
        first_k_dense_replace=(
            config.first_k_dense_replace + (num_mtp_layers - num_head_moe_layers)
        ),
    )


class _ResidentBf16HeadMoE(BaseOP):
    """The head's routed experts as ordinary resident bf16 tensors (llm-server #801 round 5
    bullet 1) -- the α_bf16 fidelity control against the deployed α_nvfp4 arm.

    ⛔ RESEARCH-ONLY. Never what a deployed row would use: it bypasses the shared
    offload-bank/cache path entirely (`FREETOKEN_MTP801_HEAD_DTYPE=bf16`, off by default).
    Attribute names (``gate``, ``shared_expert``, ``shared_expert_gate``, ``experts``) match
    `models/qwen4_exp/moe.py::Qwen4ExpMoE` / its `Qwen3_5MoE` base exactly, so the checkpoint's
    own ``mtp.layers.0.mlp.*`` keys -- returned UNCHANGED by `weight.py::_rename` under the
    bf16 dial -- fill this module with no renaming step of their own.

    ⛔⛆ **Why this can't just reuse `Qwen4ExpMoE.forward`.** For ``weight_format == "bf16"``,
    `layers/moe.py::MoELayer.forward` reads a GLOBAL ``ctx.moe_backend`` singleton, built ONCE
    at engine startup from the BACKBONE's ``config.moe_backend`` (`engine/engine.py:389`). On
    our served offload row that singleton is ``OffloadMoeBackend``, whose own ``.forward()``
    hard-raises -- offload MoE is handled by ``OffloadMoELayer``, never by calling the backend
    directly (`moe_pkg/offload.py`). A resident bf16 layer reached through the ordinary
    ``Qwen4ExpMoE.forward()`` path would hit that raise regardless of what config built the
    LAYER, because the per-layer config only chooses the layer CLASS
    (`layers/moe.py::make_moe_layer`'s ``is_offload_moe_backend`` check) -- it does not change
    which global object an already-built resident layer's ``.forward()`` reaches for. So this
    module drives ``MoELayer.routed_forward()`` instead: it computes its own top-k locally
    (``fused_topk``, the exact call `MoELayer.forward`'s non-bf16 branch already makes) and
    calls ``experts.routed_forward(...)`` -> ``_resident_gemm`` -> ``fused_experts_impl``
    directly, never touching ``ctx.moe_backend``.

    ⛔⛆ **The shared expert's quant can't be read off the backbone's config either.** The
    checkpoint's ``hf_quant_config.json`` excludes every ``mtp.*`` tensor from quantisation
    (round 2's own finding), so the head's shared expert is plain bf16 -- but `_SharedExpert`
    branches on ``config.expert_quant``/``config.dense_quant``, which on this (NVFP4) row would
    otherwise build an NVFP4 shared expert for a bf16 checkpoint tensor. Both are forced off on
    a LOCAL config copy, for this construction only; the backbone's own config is untouched.

    ⛔⛆ **What this class did not close at bullet 1 -- CORRECTED at 7a, read to the end.** The
    paragraph below is bullet 1's own record of the pool-sizing gap it left open, kept because it
    is what the gap looked like before it was closed; ⭐ **bullet 7a (`a64e34c`) closed it, in
    this file**, and the correction is spelled out after it. ⛔ Do not read the "Unresolved here"
    sentence as live. `mtp.py::pool_config` bumps ``num_layers`` by ``num_mtp_layers`` so the KV
    pool's bound check and the QSA index slab both see the head's attention layer -- needed
    under EITHER dtype, since the head's ``self_attn`` is unchanged here. But
    ``moe/offload_cache.py``'s ``num_moe_layers`` is DERIVED from that same bumped
    ``num_layers`` (`num_layers - first_k_dense_replace`), and `iter_offload_moe_layers`
    physically walks the model for `OffloadMoELayer` instances and asserts the count matches.
    Under the bf16 dial the head's ``.mlp`` is THIS class, not an `OffloadMoELayer` -- so that
    walk finds only the backbone's layers and the assertion reads ``len(layers) == 48`` against
    an expectation of ``49``.

    ⭐⭐ **CORRECTED (round 5 bullet 7a).** The fix landed exactly where this paragraph predicted
    it would have to -- `pool_config`, above, now takes a keyword-only, no-default
    ``num_head_moe_layers`` and bumps ``first_k_dense_replace`` by
    ``num_mtp_layers - num_head_moe_layers``, so the derived MoE count matches what
    `iter_offload_moe_layers` physically walks under BOTH dials: 49 under nvfp4, 48 under this
    class. ``num_head_moe_layers`` has no default on purpose -- the silent wrong answer is the
    old nvfp4 shape, so a caller that forgets it fails at the call rather than on the box, and
    `Qwen4ExpMTPHead.num_offload_moe_layers` answers with the SAME walk the engine asserts on.
    ⇒ this class no longer leaves a pool-sizing gap for the bf16 arm.

    ⛔ What still defers the bf16 arm is the `⚠ TP=1 only` note below, NOT pool sizing --
    `README.md`'s *"What round 5 deliberately did NOT do"* carries all four reasons. ⛔⛆ Leaving
    the stale sentence here would have sent the next reader to fix a bug that no longer exists,
    on the file that already fixed it.

    ⚠ TP=1 only, matching round 5's first target (`pinned`). `MoELayer`'s own
    ``intermediate_size_per_partition = div_even(intermediate_size, tp_size)`` would shard the
    resident experts across ranks the way the NVFP4 bank does NOT (that bank is unsharded);
    untested at TP=2.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        dense_config = replace(config, expert_quant="none", dense_quant="none")
        self.shared_expert = _SharedExpert(
            dense_config, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation="silu",
            apply_router_weight_on_input=False,
            weight_format="bf16",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.moe_shared_gate import (
            shared_gate_mul_add,
            shared_gate_sigmoid,
        )
        from freetoken.moe.fused import fused_topk

        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        routed = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return shared_gate_mul_add(routed, shared, gate)


class Qwen4ExpMTPHead(BaseOP):
    """The draft head's ``residual_linear_shared`` input fusion.

    Checkpoint keys (at the ``mtp.`` prefix, four of the head's 31)::

        mtp.pre_fc_norm_embedding.weight   [hidden]
        mtp.fc_embedding.weight            [hidden, hidden]
        mtp.pre_fc_norm_hidden.weight      [hc_count*hidden]
        mtp.fc_hidden.weight               [hidden, hidden]      ⭐ ONE matrix, not four

    ⚠ **The head owns no embedding.** The checkpoint sets ``mtp_use_dedicated_embeddings: false``
    and ships no `mtp.embed_tokens.*`: the head reads the backbone's
    ``model.language_model.embed_tokens``. It is therefore held under a LEADING UNDERSCORE, which
    is what keeps :meth:`BaseOP.state_dict` from walking into it — the same device `model.py` uses
    for its `_ple` tuple. If it ever entered the state dict, the loader would demand a checkpoint
    key that does not exist.
    """

    def __init__(
        self, config: ModelConfig, embed_tokens: BaseOP, *, num_mtp_layers: int = 1
    ) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size

        # ⛔⛆ BOTH norms are FLAT — one fp32 statistic over the whole last dim — NOT
        #   `GroupedPlusOneRMSNorm`. The issue body says grouped and is wrong: both vLLM twins use
        #   `GemmaRMSNorm(hidden_size * hc_count)` on the FLATTENED multi-stream, and the
        #   checkpoint's [hc_count*hidden] weight is one element per feature EITHER WAY, so no
        #   shape check catches the difference. A grouped norm here produces a plausible, wrong
        #   head. Pinned by `test_mtp_801.py::TestFuseInput::test_pre_fc_norm_hidden_is_not_grouped`.
        #   ⚠ The hyper-connection norms ARE the grouped one (per stream) — that is exactly why
        #   the two are easy to confuse.
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.fc_embedding = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=False)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(args.ple_state_width, config.rms_norm_eps)
        # ⭐ ONE [hidden, hidden] matrix applied to every hyper-connection branch; the checkpoint
        #   ships exactly one `mtp.fc_hidden.weight`. `F.linear` broadcasts over the stream axis.
        self.fc_hidden = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=False)

        # ⭐ The head's own decoder stack, at the checkpoint's own names (`mtp.layers.0.*`). It is
        #   an `OPList` of ONE for this checkpoint; see `draft_config` for why the layer ids have
        #   to be grafted onto the full-attention group before this can be built at all.
        self._num_mtp_layers = num_mtp_layers
        self._draft_config = draft_config(config, num_mtp_layers)
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(self._draft_config, config.num_layers + i)
                for i in range(num_mtp_layers)
            ]
        )
        # ⭐ #801 round 5 bullet 1: swap the head's OWN mlp for a resident bf16 module under
        #   the research-only fidelity dial. Built from the BACKBONE's `config`, not
        #   `self._draft_config` -- same reason `mtp_expert_source_banks` reads the backbone's
        #   config: the MoE geometry (num_experts, hidden_size, moe_intermediate_size) is the
        #   backbone's, `self._draft_config` only differs in `attention_groups`, which this
        #   class does not touch. ⛔ Leaves the head's `self_attn`/QSA layer, built above,
        #   completely alone -- only `.mlp` is replaced.
        if os.getenv(_MTP801_HEAD_DTYPE_ENV, "nvfp4").strip().lower() == "bf16":
            self.layers.op_list[0].mlp = _ResidentBf16HeadMoE(config)
        # ⭐ `use_combine=False`: the top mixer owns an UNMERGED `input_mix_weight_down`, returns
        #   no inject logits and never combines. It collapses the multi-stream for the LM head.
        #   ⚠ Built from `config`, not the draft config — it is not a layer and owns no layer id.
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

        # shared, not owned: leading underscore keeps it out of the state dict (see the docstring)
        self._embed_tokens = embed_tokens

    @property
    def num_offload_moe_layers(self) -> int:
        """How many of the head's layers the engine's offload walk will find (#801 round 5
        bullet 7a): ``num_mtp_layers`` under the deployed nvfp4 dial, ``0`` under round 5 bullet
        1's resident-bf16 one.

        ⭐⭐ **It is the SAME walk `engine.py` asserts on**, run over the head alone. That is the
        whole value of this property: `mtp.py::pool_config` is then told the count by the object
        the assertion will actually inspect, so the two agree STRUCTURALLY. Deriving it from
        `weight.py::mtp_head_dtype` instead would re-read the environment a second time, after
        construction already decided -- which is precisely how a config and a model diverge, and
        the round's own rule (`model.py`'s engine-hook comment) says not to.
        """
        from freetoken.moe.offload_cache import iter_offload_moe_layers

        return len(list(iter_offload_moe_layers(self)))

    def fuse_input(self, input_ids: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """``residual_linear_shared``: the backbone's multi-stream ``R`` and the next token, fused.

        ``input_ids [T]``, ``R [T, hc_count*hidden]`` -> ``R0 [T, hc_count*hidden]``::

            e  = fc_embedding(pre_fc_norm_embedding(embed(input_ids)))          # [T, hidden]
            h  = fc_hidden(pre_fc_norm_hidden(R).unflatten(-1, (hc, hidden)))   # ⛔ FLAT norm
            R0 = (e.unsqueeze(-2) + h).flatten(-2)

        ⭐ The embedding is added to EVERY stream with unit weight (the amd spelling), before the
        layer's ordinary `mix` — not injected through a hyper-connection combine.

        ⚠ ``R`` is the PRE-final-mixer multi-stream, i.e. what `Qwen4ExpModel.forward` feeds its
        `hyper_connection_mixer` rather than what it returns. Bullet 6 taps it; until then the
        caller supplies it.
        """
        e = self._embed_tokens.forward(input_ids)
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(e))

        h = self.pre_fc_norm_hidden.forward(R).unflatten(-1, (self.hc_count, self.hidden_size))
        h = self.fc_hidden.forward(h)

        return (e.unsqueeze(-2) + h).flatten(-2)

    def forward(
        self, input_ids: torch.Tensor, R: torch.Tensor, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(sample_hidden [T, hidden], multi_hidden [T, hc_count*hidden])``.

        ⭐ **Both outputs fall out of our frozen layer contract for free.** `Qwen4ExpDecoderLayer`
        combines immediately, so ``R' = layer(R0)`` IS the pre-final-mixer multi-stream — what a
        k>1 draft step would consume — and ``mixer.mix(R')[0]`` is the collapsed [T, hidden] the LM
        head reads. vLLM gets the same two from one fused `combine_and_mix`; we get them from one
        unfused kernel, no extra compute, and ⛔ NO change to the layer contract.

        ⚠ We are k=1, so ``multi_hidden`` is kept and unused. Keeping it costs nothing.

        ⛔⛆ k>1 chains the SAME layer across draft STEPS (vLLM: ``spec_step_idx %
        num_mtp_layers``); it does NOT run the layers in sequence inside one forward. A
        ``for layer in self.layers.op_list`` loop here would look right and be a different head
        the moment a second layer existed.
        """
        hidden = self.layers.op_list[0].forward(self.fuse_input(input_ids, R), batch)
        return self.hyper_connection_mixer.mix(hidden)[0], hidden


__all__ = ["Qwen4ExpMTPHead", "draft_config", "pool_config"]
