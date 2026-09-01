"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``).
``model.visual.*`` -- the 333-tensor, 897,862,112 B BF16 vision tower -- is gated on
``config.is_multimodal`` (i.e. ``FREETOKEN_LOAD_VISION=1``) and dropped when it is off.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files, shard_tensor
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.models.config import LinearGatedDeltaGroupConfig, ModelConfig

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# The vision patch embedding is stored as a Conv3d kernel and consumed as a Linear weight.
_PATCH_EMBED_WEIGHT = "visual.patch_embed.proj.weight"

# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


# ⭐ TP (llm-server #777): the dense tensors whose module is tensor-parallel, sharded HERE, on the
# checkpoint's OWN per-projection keys. Everything not listed here or in the GDN tables below is
# deliberately replicated: the HC mixers, the PLE projections, the QSA indexer, every norm, the
# routers and the vision tower are all ``LinearReplicated`` and every rank needs them whole.
#
# ⛔⛆ These five are FUSION PARTS and must be sharded BEFORE ``_try_fuse`` concatenates them. A
# flat row chunk of the fused tensor has the right shape and plausible values and is a different
# tensor: ``qkv_proj`` is ``[2*qo | kv | kv]``, so rank 0's top half of it is part of q and none
# of k or v. ``LinearColParallelMerged`` divides each declared output size on its own, so the
# per-part shard is the layout it allocates.
_SHARD_BEFORE_FUSE = (
    ".self_attn.q_proj.weight",  # head-major, [q | gate] interleaved per head: a chunk is head-aligned
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
)
# Not fused, so the order is irrelevant: the row-parallel projections (sharded on their INPUT
# axis, the one their column-parallel producer already split) and the vocab-parallel pair.
_SHARD_UNFUSED = (
    ".self_attn.o_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
)
_TP_SHARDED = _SHARD_BEFORE_FUSE + _SHARD_UNFUSED
# The two whose width follows ``num_kv_heads`` rather than the rank count: below one head per rank
# ``shard_tensor`` replicates instead of splitting. See the assert in ``_shard_for_rank``.
_KV_PROJ = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")

# ⭐ TP bullet 6 (llm-server #777): the vocab pair is the one sharded tensor whose rank-local
# buffer is NOT the rank-local slice. ``VocabParallelEmbedding`` allocates ``div_ceil(V, tp)``
# rows on EVERY rank and clamps the live range with ``vocab_range``, because ``ParallelLMHead``
# all-gathers the logits (a collective needs the same shape on every rank) and then truncates
# the gathered row back to ``num_embeddings``. ``shard_tensor`` hands the LAST rank the short
# real slice, so on a vocab that does not divide, the loader's tensor is narrower than the buffer.
_VOCAB_PARALLEL = ("model.embed_tokens.weight", "lm_head.weight")


def _pad_vocab_rows(local: torch.Tensor, full_rows: int, *, world_size: int) -> torch.Tensor:
    """``local`` grown to the module's ``div_ceil(V, tp)`` partition with ZERO rows.

    The pad is never read -- the embedding masks by ``vocab_range`` and the LM head slices the
    gathered logits back to ``num_embeddings`` -- but the buffer it fills is ``torch.empty``, so
    zeros are what keeps a mis-wired read deterministic rather than garbage.
    """
    per_rank = div_ceil(full_rows, world_size)
    pad = per_rank - local.shape[0]
    assert pad >= 0, f"vocab shard is {local.shape[0]} rows, wider than the {per_rank}-row buffer"
    if pad == 0:
        return local
    return torch.cat(
        [local, torch.zeros(pad, *local.shape[1:], dtype=local.dtype, device=local.device)], dim=0
    )


# ⭐ TP bullet 5 (llm-server #777): the GDN, whose tensors are COMPOSITE one level deeper.
#
# ⛔⛆ ``in_proj_qkv.weight`` and ``conv1d.weight`` are both laid out on ``conv_dim``, which is a
# CONCATENATION ``2 * key_dim + value_dim`` -- on the deployed 16 k / 48 v x 128 geometry that is
# ``[2048 | 2048 | 6144]``. A flat row chunk of either gives rank 0 all of q, all of k and a third
# of v: the right shape, plausible bf16, a different tensor. ``shard_tensor`` cannot help -- none
# of its substrings match a ``linear_attn`` key -- so the sub-block split is written here.
#
# ⚠ The whole GDN moves in ONE bullet on purpose. ``in_proj`` is a four-part fusion
# (``qkv | z | b | a``); sharding ``z``/``b``/``a`` while leaving ``qkv`` whole would emit a fused
# tensor that tiles nothing the module declares.
_GDN_INFIX = ".linear_attn."
# dim 0, and the axis is ``[key | key | value]``: each sub-block follows its OWN head count.
_GDN_CONV_COMPOSITE = ("in_proj_qkv.weight", "conv1d.weight")
# dim 0, one block (``z``: head_v_dim; ``b``/``a``/``A_log``/``dt_bias``: exactly one) per v head.
_GDN_VALUE_ROWS = ("in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias")
# dim 1: out_proj consumes the column-sharded value_dim its producer emitted (LinearRowParallel).
_GDN_VALUE_COLUMNS = ("out_proj.weight",)
# The gated norm is ``head_v_dim`` wide -- a per-head width, not a per-head COUNT. Replicated.
_GDN_REPLICATED = ("norm.weight",)


def _head_slice(
    tensor: torch.Tensor, dim: int, num_heads: int, *, rank: int, world_size: int
) -> torch.Tensor:
    """This rank's heads out of ``dim``, on the same rule ``div_even(..., allow_replicate=True)``
    encodes in the modules: an even split, or -- when there are fewer heads than ranks -- the one
    head this rank shares with its neighbours (``shard_tensor``'s kv branch, generalised)."""
    width = tensor.shape[dim]
    assert width % num_heads == 0, f"{width} does not divide into {num_heads} heads"
    per_head = width // num_heads
    if world_size > num_heads:
        assert world_size % num_heads == 0, (
            f"{world_size} ranks must be divisible by {num_heads} heads for replication"
        )
        lo = (rank * num_heads // world_size) * per_head
        take = per_head
    else:
        take = div_even(num_heads, world_size) * per_head
        lo = rank * take
    return tensor.narrow(dim, lo, take).clone()


def _shard_gdn(
    leaf: str,
    tensor: torch.Tensor,
    group: LinearGatedDeltaGroupConfig,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """This rank's slice of one GDN tensor, named by its leaf below ``.linear_attn.``.

    ⛔⛆ An unrecognised leaf RAISES rather than falling through as replicated: a GDN tensor that
    nobody classified would be loaded whole into a rank-local buffer, which is the silent failure
    this bullet exists to remove.
    """
    if leaf in _GDN_REPLICATED:
        return tensor
    if leaf in _GDN_CONV_COMPOSITE:
        key_dim = group.num_key_heads * group.key_head_dim
        value_dim = group.num_value_heads * group.value_head_dim
        assert tensor.shape[0] == 2 * key_dim + value_dim, (
            f"GDN {leaf} is {tuple(tensor.shape)}, expected conv_dim "
            f"{2 * key_dim + value_dim} = 2*{key_dim} + {value_dim} rows"
        )
        q, k, v = torch.split(tensor, [key_dim, key_dim, value_dim], dim=0)
        return torch.cat(
            [
                _head_slice(q, 0, group.num_key_heads, rank=rank, world_size=world_size),
                _head_slice(k, 0, group.num_key_heads, rank=rank, world_size=world_size),
                _head_slice(v, 0, group.num_value_heads, rank=rank, world_size=world_size),
            ],
            dim=0,
        )
    if leaf in _GDN_VALUE_ROWS:
        return _head_slice(tensor, 0, group.num_value_heads, rank=rank, world_size=world_size)
    if leaf in _GDN_VALUE_COLUMNS:
        return _head_slice(tensor, 1, group.num_value_heads, rank=rank, world_size=world_size)
    raise NotImplementedError(
        f"GDN tensor {leaf!r} is not classified for tensor parallelism; add it to one of "
        f"_GDN_CONV_COMPOSITE / _GDN_VALUE_ROWS / _GDN_VALUE_COLUMNS / _GDN_REPLICATED"
    )


def _shard_for_rank(
    name: str, tensor: torch.Tensor, *, config: ModelConfig
) -> torch.Tensor:
    """This rank's slice of ``name``, or the tensor whole when its module is replicated.

    Attention and the dense projections delegate the slicing to
    :func:`freetoken.models.loader.shard_tensor` so the loader and every other model divide on the
    same rule (including its ``num_kv_heads < world_size`` replication branch, which is the one
    ``div_even(..., allow_replicate=True)`` mirrors in the modules). The GDN cannot: its tensors
    are composite on an axis ``shard_tensor`` does not recognise, so it goes through
    :func:`_shard_gdn`.
    """
    tp = get_tp_info()
    if tp.size == 1:
        return tensor
    leaf = name.split(_GDN_INFIX, 1)[1] if _GDN_INFIX in name else None
    if leaf is not None:
        return _shard_gdn(
            leaf, tensor, config.linear_attention_group(), rank=tp.rank, world_size=tp.size
        )
    if not name.endswith(_TP_SHARDED):
        return tensor
    local = shard_tensor(
        name, tensor, rank=tp.rank, world_size=tp.size, num_kv_heads=config.num_kv_heads
    )
    # shard_tensor returns the tensor UNCHANGED when its own substring tables do not recognise a
    # key. That silent pass-through would load the full weight into a rank-local buffer, so a key
    # this loader declares tensor-parallel must actually come back smaller.
    #
    # ⛔⛆ EXCEPT on the replication branch, where whole IS the rank-local tensor. With a single kv
    # head, ``shard_tensor`` hands every rank that one head -- the entire projection -- and
    # ``div_even(1, tp, allow_replicate=True)`` returns 1, so the module declares the full kv
    # width to match. Asserting "smaller" there would reject the geometry the modules support.
    # (The deployed checkpoint has 2 kv heads, so this branch is unreachable on it; the toy
    # configs reach it, and so would any future 1-kv-head checkpoint.)
    if not (name.endswith(_KV_PROJ) and config.num_kv_heads < tp.size):
        assert local.shape != tensor.shape, (
            f"{name!r} is declared tensor-parallel but shard_tensor returned it whole "
            f"({tuple(tensor.shape)}) at rank {tp.rank}/{tp.size}"
        )
    if name.endswith(_VOCAB_PARALLEL):
        local = _pad_vocab_rows(local, tensor.shape[0], world_size=tp.size)
    return local


_VISION_PREFIXES = ("model.visual.", "visual.")


def _rename(raw_name: str, *, include_vision: bool) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if raw_name.startswith(_VISION_PREFIXES):
        # ``model.visual.blocks.0.attn.qkv.weight`` -> ``visual.blocks.0.attn.qkv.weight``.
        return (
            "visual." + raw_name.split("visual.", 1)[1] if include_vision else None
        )
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    # ⭐ TP (llm-server #777, correcting #725): this loader is NOT rank-independent. #725 left the
    # mixers replicated and banked "every dense weight stays replicated" -- that premise is what
    # failed: the modules shard their own head counts, so the tensors backing them must arrive
    # rank-local. ``_TP_SHARDED`` names exactly which, and ``_shard_for_rank`` runs BEFORE
    # ``_try_fuse`` so the fusion parts are cut on their own axes rather than out of the concat.
    if not include_non_moe:
        return

    from .config import parse_config

    config = parse_config(cached_load_hf_config(model_path))
    include_vision = config.is_multimodal

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name, include_vision=include_vision)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                if name == _PATCH_EMBED_WEIGHT:
                    # The tower's patch_embed is a Conv3d whose stride equals its kernel, which
                    # is exactly a Linear over the flattened patch. Fold the reshape into the
                    # load so the forward never pays it: [H, C, T, P, P] -> [H, C*T*P*P].
                    tensor = tensor.reshape(tensor.shape[0], -1)
                tensor = _shard_for_rank(name, tensor, config=config)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def _ple_row_shard(folder: str, bank_rows: int) -> tuple[int, int]:
    """This rank's ``[lo, hi)`` row range of the flat n-gram table (llm-server #725).

    The table is sharded on the HASH HEAD axis: rank ``r`` owns heads
    ``[r * H/tp, (r + 1) * H/tp)``, and because ``derive_ngram_hash_constants`` lays each head's
    prime-sized vocab out back to back, those heads are one contiguous row range. The split point
    is read from the checkpoint's OWN ``ngram_heads_offsets`` rather than recomputed from the
    prime derivation, so a checkpoint whose constants ever diverge from the oracle shards
    correctly instead of silently loading the wrong rows -- ``NGramEmbedding.local_row_base``
    rebases lookups against the same tensor. The last rank absorbs the padding rows the bank is
    rounded up to.
    """
    info = get_tp_info()
    if info.size == 1:
        return 0, bank_rows
    offsets = None
    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    for name, shard in weight_map.items():
        if name.endswith(".ple.ple_embedding.ngram_heads_offsets"):
            with safetensors.safe_open(os.path.join(folder, shard), framework="pt",
                                       device="cpu") as f:
                offsets = f.get_tensor(name).tolist()
            break
    if offsets is None:
        raise ValueError("PLE table cannot be TP-sharded: no ngram_heads_offsets in the checkpoint")
    per_rank = div_even(len(offsets), info.size)
    lo = int(offsets[per_rank * info.rank])
    hi = bank_rows if info.rank == info.size - 1 else int(offsets[per_rank * (info.rank + 1)])
    return lo, hi


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    shard_bytes = rows * cols
    row_lo, row_hi = _ple_row_shard(folder, expected * rows)
    bank = HostBank((row_hi - row_lo, cols), torch.float8_e4m3fn)
    bar = byte_bar((row_hi - row_lo) * cols, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            # This shard covers global rows [lo, hi); keep only its overlap with our own range.
            # Exactly one shard straddles each boundary, and ``read_range_into`` bounces any
            # sub-block-aligned window, so a partial read needs no special case here.
            lo, hi = shard * rows, (shard + 1) * rows
            take_lo, take_hi = max(lo, row_lo), min(hi, row_hi)
            if take_lo >= take_hi:
                continue
            take = (take_hi - take_lo) * cols
            read_range_into(buf, path, file_offset=offset + (take_lo - lo) * cols, nbytes=take,
                            dest_offset=(take_lo - row_lo) * cols, workers=workers, chunk=chunk)
            bar.update(take)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "PleTable",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
]
