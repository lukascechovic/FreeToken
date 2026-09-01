"""qwen4_exp weight loading against a synthetic checkpoint shaped like the RadixArk NVFP4 one.

The tensors are tiny but the key names, dtypes and the fusion geometry that matters
(hc_lowrank=320 + hc_count=4 -> a 12-row zero pad) are the real ones.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.aot_models import SUPPORTED_MODELS, expert_bank_row_bytes
from freetoken.models.qwen4_exp.weight import (
    _FUSIONS,
    _GDN_CONV_COMPOSITE,
    _GDN_INFIX,
    _GDN_REPLICATED,
    _GDN_VALUE_COLUMNS,
    _GDN_VALUE_ROWS,
    _SHARD_BEFORE_FUSE,
    _SHARD_UNFUSED,
    _ZERO_CENTERED_NORM_SUFFIXES,
    iter_weights,
    load_ple_table,
)
from freetoken.moe.host_banks import HostBank, read_range_into

from .common import EXPERT_BUFFERS, as_rank, hf_config, model_buffer_shapes

H = 32  # hidden_size
HC = 4  # hc_count
LR = 320  # hc_lowrank; kept real so the merged HC pad is the real (-(320+4)) % 16 = 12
HCH = HC * H  # hyper-connection stream width
KH, VH, HD = 2, 6, 8  # GDN key / value heads, head dim
QH, KVH, AHD = 4, 2, 64  # QSA q / kv heads, head dim
IHD = 16  # indexer head dim
# ⚠ AHD and IHD are NOT free (bullet 6). ``RotaryEmbedding`` asserts ``head_size in
# [64, 128, 256, 512]``, and the transformers config validator refuses a ``rotary_dim``
# (= partial_rotary_factor 0.25 x AHD) wider than ``indexer_head_dim``. Both used to be smaller,
# and the whole model then could not be BUILT from this config -- which is what bullet 6 walks
# the loader against.
E, I = 3, 6  # routed experts, moe_intermediate_size
NGRAM_DIM, NGRAM_ROWS, NGRAM_SHARDS = 4, 7, 4
VOCAB = 11  # deliberately ODD: div_ceil leaves the last rank short, and the shards must still join


def _config_json() -> dict:
    """The synthetic checkpoint's own ``config.json``, matching the tensors above.

    ⛔⛆ Not decoration. ``iter_weights`` reads the config for the vision gate (our ladder patch
    `0004-qwen4exp-vision`) and, since #777, for the head counts it shards on. The fixture used to
    write no ``config.json`` at all, so the whole module died in the fixture -- ten collection
    errors that a failure COUNT matched against upstream perfectly. Geometry drift between this
    and ``_raw_checkpoint`` is pinned by ``test_the_config_matches_the_synthetic_tensors``.
    """
    cfg = hf_config(
        num_layers=2, head_dim=AHD, num_q=QH, num_kv=KVH,
        index_head_dim=IHD, index_heads=4, hidden=H,
        vocab_size=VOCAB,
        layer_types=["linear_attention", "full_attention"],
        num_experts=E, num_experts_per_tok=2,
        moe_intermediate_size=I, shared_expert_intermediate_size=I,
        linear_num_key_heads=KH, linear_num_value_heads=VH,
        linear_key_head_dim=HD, linear_value_head_dim=HD, linear_conv_kernel_dim=4,
        hc_count=HC, hc_lowrank=LR,
        ple_layer_ids=[1], ple_embed_dim=H,  # one-indexed: layer index 0 carries the PLE
    )
    return {**vars(cfg), "text_config": vars(cfg.text_config)}


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _raw_checkpoint() -> dict[str, torch.Tensor]:
    """Layer 0 = GDN + PLE, layer 1 = QSA; plus the mtp / visual / routed-expert noise."""
    lm = "model.language_model"
    raw: dict[str, torch.Tensor] = {
        f"{lm}.embed_tokens.weight": _bf16(VOCAB, H),
        "lm_head.weight": _bf16(VOCAB, H),
    }
    raw.update(_hc_weights(f"{lm}.hyper_connection_mixer", inject=False))
    for layer in (0, 1):
        raw.update(_hc_weights(f"{lm}.layers.{layer}.attn_hyper_connection", inject=True))
        raw.update(_hc_weights(f"{lm}.layers.{layer}.mlp_hyper_connection", inject=True))
        raw.update({
            f"{lm}.layers.{layer}.mlp.gate.weight": _bf16(E, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.gate_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.up_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, I),
            f"{lm}.layers.{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        })
        for expert in range(E):
            base = f"{lm}.layers.{layer}.mlp.experts.{expert}"
            for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                raw[f"{base}.{proj}.weight"] = torch.randint(
                    0, 256, (out, inn // 2), dtype=torch.uint8
                )
                raw[f"{base}.{proj}.weight_scale"] = torch.ones(
                    out, inn // 16 or 1, dtype=torch.float8_e4m3fn
                )
                raw[f"{base}.{proj}.weight_scale_2"] = torch.tensor(0.5)
                raw[f"{base}.{proj}.input_scale"] = torch.tensor(0.25)
    gdn = f"{lm}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H),
        f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H),
        f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4),
        f"{gdn}.A_log": _bf16(VH),
        f"{gdn}.dt_bias": _bf16(VH),
        f"{gdn}.norm.weight": _bf16(HD),
        f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    ple = f"{lm}.layers.0.ple"
    raw.update({
        f"{ple}.key_proj.weight": _bf16(HCH, H),
        f"{ple}.value_proj.weight": _bf16(H, H),
        f"{ple}.norm_key.weight": _bf16(HCH),
        f"{ple}.norm_query.weight": _bf16(HCH),
        f"{ple}.norm_conv.weight": _bf16(HCH),
        f"{ple}.conv1d.weight": _bf16(HCH, 1, 4),
        f"{ple}.ple_embedding.layer_multipliers": torch.randint(1, 1 << 40, (3,)),
        f"{ple}.ple_embedding.ngram_heads_offsets": torch.arange(4),
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes": torch.full((4,), 5),
    })
    attn = f"{lm}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD),
        f"{attn}.k_norm.weight": _bf16(AHD),
        f"{attn}.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        f"{attn}.indexer.q_layernorm.weight": _bf16(IHD),
        f"{attn}.indexer.k_layernorm.weight": _bf16(IHD),
    })
    raw.update({
        "mtp.hyper_connection_mixer.hc_norm.weight": _bf16(HCH),
        "mtp.layers.0.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        "mtp.layers.0.mlp.experts.gate_up_proj": _bf16(E, 2 * I, H),
        "mtp.layers.0.mlp.experts.down_proj": _bf16(E, H, I),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * H, H),
        "model.visual.merger.norm.weight": _bf16(H),
    })
    return raw


def _ngram_table() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = {
        f"{prefix}.shard_{i}.weight": (
            torch.arange(i * NGRAM_ROWS * NGRAM_DIM, (i + 1) * NGRAM_ROWS * NGRAM_DIM)
            .remainder(200).to(torch.uint8).view(NGRAM_ROWS, NGRAM_DIM).view(torch.float8_e4m3fn)
        )
        for i in range(NGRAM_SHARDS)
    }
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    shards[f"{prefix}.weight_scale"] = scale
    return shards, scale


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    folder = tmp_path_factory.mktemp("qwen4_exp_ckpt")
    (folder / "config.json").write_text(json.dumps(_config_json()), encoding="utf-8")
    raw = _raw_checkpoint()
    table, _scale = _ngram_table()
    # Spread the dense tensors over two shards so the fusion buffer has to survive a file
    # boundary, and put the n-gram table in its own shards like the real checkpoint does.
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    shard_names = sorted(table)
    save_file({n: table[n] for n in shard_names[:2]}, str(folder / "model-plefp8-00000.safetensors"))
    save_file({n: table[n] for n in shard_names[2:]}, str(folder / "model-plefp8-00001.safetensors"))
    return str(folder), {**raw, **table}


@pytest.fixture(scope="module")
def loaded(checkpoint) -> dict[str, torch.Tensor]:
    folder, _raw = checkpoint
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    }


def _expected_names() -> set[str]:
    names = {"model.embed_tokens.weight", "lm_head.weight"}
    names |= {f"model.hyper_connection_mixer.{leaf}" for leaf in
              ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")}
    for layer in (0, 1):
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            names |= {f"model.layers.{layer}.{hc}.{leaf}" for leaf in (
                "hc_norm.weight", "input_mix_weight_down_block_inject.weight",
                "input_mix_weight_up.weight")}
        names |= {f"model.layers.{layer}.mlp.{leaf}" for leaf in (
            "gate.weight", "shared_expert.gate_up_proj.weight",
            "shared_expert.down_proj.weight", "shared_expert_gate.weight")}
    names |= {f"model.layers.0.linear_attn.{leaf}" for leaf in (
        "in_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")}
    names |= {f"model.layers.0.ple.{leaf}" for leaf in (
        "key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight",
        "norm_conv.weight", "conv1d.weight", "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_heads_vocab_sizes")}
    names |= {f"model.layers.1.self_attn.{leaf}" for leaf in (
        "qkv_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
        "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight",
        "indexer.k_layernorm.weight")}
    return names


def test_key_map_is_exactly_the_model_state_dict(loaded):
    assert set(loaded) == _expected_names()


def test_mtp_visual_experts_and_table_never_loaded(loaded):
    for name in loaded:
        assert not name.startswith(("mtp.", "model.visual."))
        assert ".mlp.experts." not in name
        assert "ngram_embedding" not in name
        assert not name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))


def test_hc_merge_is_down_then_inject_then_zero_pad(loaded, checkpoint):
    _folder, raw = checkpoint
    key = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    merged = loaded[key]
    assert merged.shape == (LR + HC + 12, HCH)  # pad = (-(320 + 4)) % 16
    down = raw["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"]
    inject = raw["model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"]
    assert torch.equal(merged[:LR], down)
    assert torch.equal(merged[LR:LR + HC], inject)
    assert torch.equal(merged[LR + HC:], torch.zeros(12, HCH, dtype=merged.dtype))


def test_top_level_mixer_keeps_the_unmerged_down(loaded, checkpoint):
    _folder, raw = checkpoint
    got = loaded["model.hyper_connection_mixer.input_mix_weight_down.weight"]
    assert got.shape == (LR, HCH)
    assert torch.equal(
        got, raw["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"]
    )
    assert torch.equal(
        loaded["model.hyper_connection_mixer.input_mix_weight_up.weight"],
        raw["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"],
    )


def test_qkv_fusion_slices_back_to_q_k_v(loaded, checkpoint):
    _folder, raw = checkpoint
    attn = "model.language_model.layers.1.self_attn"
    parts = [raw[f"{attn}.{p}_proj.weight"] for p in ("q", "k", "v")]
    fused = loaded["model.layers.1.self_attn.qkv_proj.weight"]
    assert fused.shape == (2 * QH * AHD + 2 * KVH * AHD, H)  # q carries the output gate
    for part, back in zip(parts, torch.split(fused, [p.shape[0] for p in parts], dim=0)):
        assert torch.equal(part, back)


def test_gdn_in_proj_slices_round_trip(loaded, checkpoint):
    _folder, raw = checkpoint
    gdn = "model.language_model.layers.0.linear_attn"
    parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
    fused = loaded["model.layers.0.linear_attn.in_proj.weight"]
    assert fused.shape == (sum(p.shape[0] for p in parts), H)
    splits = torch.split(fused, [p.shape[0] for p in parts], dim=0)
    for part, back in zip(parts, splits):
        assert torch.equal(part, back)


def test_shared_expert_gate_up_merge(loaded, checkpoint):
    _folder, raw = checkpoint
    base = "model.language_model.layers.1.mlp.shared_expert"
    merged = loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(merged[:I], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(merged[I:], raw[f"{base}.up_proj.weight"])


ZERO_CENTERED = (
    "model.layers.0.attn_hyper_connection.hc_norm.weight",
    "model.layers.0.mlp_hyper_connection.hc_norm.weight",
    "model.hyper_connection_mixer.hc_norm.weight",
    "model.layers.0.ple.norm_key.weight",
    "model.layers.0.ple.norm_query.weight",
    "model.layers.0.ple.norm_conv.weight",
    "model.layers.1.self_attn.q_norm.weight",
    "model.layers.1.self_attn.k_norm.weight",
    "model.layers.1.self_attn.indexer.q_layernorm.weight",
    "model.layers.1.self_attn.indexer.k_layernorm.weight",
)


def test_zero_centered_norms_are_loaded_raw(loaded, checkpoint):
    """(1+w) is applied at runtime in fp32, so the loader must not fold it into the bf16 weight."""
    _folder, raw = checkpoint
    for name in ZERO_CENTERED:
        raw_name = name.replace("model.", "model.language_model.", 1)
        assert torch.equal(loaded[name], raw[raw_name]), name


def test_the_zero_centered_suffix_list_covers_every_such_norm():
    assert {n for n in ZERO_CENTERED if n.endswith(_ZERO_CENTERED_NORM_SUFFIXES)} == set(ZERO_CENTERED)
    assert not "model.layers.0.linear_attn.norm.weight".endswith(_ZERO_CENTERED_NORM_SUFFIXES)


def test_gdn_gated_norm_passes_through(loaded, checkpoint):
    _folder, raw = checkpoint
    assert torch.equal(
        loaded["model.layers.0.linear_attn.norm.weight"],
        raw["model.language_model.layers.0.linear_attn.norm.weight"],
    )


def test_hash_constants_stay_int64(loaded):
    for leaf in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        assert loaded[f"model.layers.0.ple.ple_embedding.{leaf}"].dtype is torch.int64


def test_load_ple_table_concatenates_shards_in_index_order(checkpoint):
    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_ple_table(folder, args, pin=False)
    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert table.tensor.dtype is torch.float8_e4m3fn
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows.view(torch.uint8),
                           raw[f"{prefix}.shard_{shard}.weight"].view(torch.uint8))
    assert table.weight_scale.dtype is torch.bfloat16
    assert float(table.weight_scale) == 0.125


def test_load_ple_table_rejects_a_shard_count_mismatch(checkpoint):
    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS + 1, ngram_head_dim=NGRAM_DIM)
    with pytest.raises(ValueError, match="shards 0"):
        load_ple_table(folder, args, pin=False)


# ======================================================================================
# read_range_into: the O_DIRECT byte-range read the PLE table load is built on
# ======================================================================================


@pytest.fixture(scope="module")
def blob(tmp_path_factory) -> tuple[str, bytes]:
    data = random.Random(7).randbytes(5_000_003)
    path = tmp_path_factory.mktemp("blob") / "data.bin"
    path.write_bytes(data)
    return str(path), data


@pytest.mark.parametrize("file_offset, nbytes, dest_offset", [
    (1, 4095, 0),                 # sub-block, unaligned source
    (2239, 1_000_000, 0),         # the real checkpoint's header-end phase
    (4095, 4097, 1),              # straddles two block boundaries
    (4_999_000, 1003, 123_456),   # runs to EOF
])
def test_read_range_into_matches_the_file(blob, file_offset, nbytes, dest_offset):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    got = read_range_into(view, path, file_offset=file_offset, nbytes=nbytes,
                          dest_offset=dest_offset, chunk=1 << 20)
    assert got == nbytes
    assert bytes(view[dest_offset:dest_offset + nbytes]) == data[file_offset:file_offset + nbytes]


def test_read_range_into_is_chunk_and_thread_safe(blob):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    read_range_into(view, path, file_offset=2239, nbytes=4_000_000, dest_offset=1024,
                    workers=8, chunk=64 << 10)
    assert bytes(view[1024:1024 + 4_000_000]) == data[2239:2239 + 4_000_000]


def test_read_range_into_rejects_a_short_destination(blob):
    path, _data = blob
    bank = HostBank((1024,), torch.uint8)
    with pytest.raises(ValueError, match="destination holds"):
        read_range_into(bank.memoryview(), path, file_offset=0, nbytes=1 << 20)


# ======================================================================================
# AOT shape table
# ======================================================================================


def test_aot_entry_carries_the_checkpoint_geometry():
    entry = next(m for m in SUPPORTED_MODELS
                 if m.architecture == "Qwen4ExpForConditionalGeneration")
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (2560, 640, 10)
    assert entry.kv_groups == ((2, 256),)
    rows = expert_bank_row_bytes("nvfp4", entry.hidden_size, entry.moe_intermediate_size)
    assert set(rows) == {"gate_up_packed", "gate_up_scale", "gate_up_global",
                         "down_packed", "down_scale", "down_global"}
    for name, nbytes in rows.items():
        assert nbytes % 16 == 0, name  # fused multi-bank copy only engages on 16B multiples


def test_every_registry_architecture_is_claimed_by_an_aot_entry():
    from freetoken.models.register import _MODEL_REGISTRY

    claimed = {m.architecture for m in SUPPORTED_MODELS}
    claimed |= {a for m in SUPPORTED_MODELS for a in m.arch_aliases}
    assert "Qwen4ExpForConditionalGeneration" in claimed
    assert set(_MODEL_REGISTRY) - claimed == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_fusion_pad_rides_the_tensor_device():
    """safetensors loads straight to cuda; a cpu-allocated pad row would break torch.cat."""
    from freetoken.models.qwen4_exp.weight import _try_fuse

    buf = {}
    down = torch.randn(320, 64, device="cuda", dtype=torch.bfloat16)
    inject = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    assert _try_fuse("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", down, buf) == ()
    key, fused = _try_fuse("model.layers.0.attn_hyper_connection.block_inject_weight.weight", inject, buf)
    assert fused.device.type == "cuda" and fused.shape[0] == 336
    assert torch.equal(fused[324:], torch.zeros(12, 64, device="cuda", dtype=torch.bfloat16))


# ======================================================================================
# TP=2: the loader shards BEFORE it fuses (llm-server #777, bullet 4)
# ======================================================================================
#
# ⛔⛆ The failure this section exists to catch is SILENT. `qkv_proj` is `[2*qo | kv | kv]` and
# `gate_up_proj` is `[gate | up]`; a flat row chunk of either fused tensor has exactly the shape
# the rank-local buffer wants and is full of plausible bf16 values. Only a comparison against the
# per-projection shard separates them, so every positive assertion below is paired with a
# rejection of the flat chunk.
#
# The GDN is composite one level deeper again and has its own section at the bottom of this file
# (bullet 5): `in_proj_qkv` and `conv1d.weight` are laid out on `conv_dim = 2*key_dim + value_dim`,
# so even a per-projection chunk of them is the wrong tensor.

ATTN = "model.layers.1.self_attn"
RAW_ATTN = "model.language_model.layers.1.self_attn"
SHARED = "model.layers.1.mlp.shared_expert"
RAW_SHARED = "model.language_model.layers.1.mlp.shared_expert"


def _load_at(folder: str, rank: int, size: int) -> dict[str, torch.Tensor]:
    with as_rank(rank, size):
        return {
            name: tensor.clone()
            for name, tensor in iter_weights(
                folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
            )
        }


@pytest.fixture(scope="module")
def tp2(checkpoint) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Both ranks of a TP=2 load of the same checkpoint."""
    folder, _raw = checkpoint
    return _load_at(folder, 0, 2), _load_at(folder, 1, 2)


def test_the_config_matches_the_synthetic_tensors(checkpoint):
    """The config.json the fixture writes is read by the loader, so drift from the tensors is a bug."""
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.utils import cached_load_hf_config

    folder, _raw = checkpoint
    with as_rank(0, 1):
        config = parse_config(cached_load_hf_config(folder))
    assert (config.num_qo_heads, config.num_kv_heads, config.head_dim) == (QH, KVH, AHD)
    assert (config.hidden_size, config.vocab_size) == (H, VOCAB)
    assert config.shared_expert_intermediate_size == I
    assert not config.is_multimodal  # the visual.* tensors must stay dropped


def test_tp1_load_is_unchanged(checkpoint, loaded):
    """⛔⛆ Negative control: at TP=1 `_shard_for_rank` is a no-op, byte for byte."""
    folder, _raw = checkpoint
    at_tp1 = _load_at(folder, 0, 1)
    assert set(at_tp1) == set(loaded)
    for name, tensor in at_tp1.items():
        assert torch.equal(tensor, loaded[name]), name


@pytest.mark.parametrize("rank", (0, 1))
def test_qkv_is_sharded_per_projection_then_fused(tp2, checkpoint, rank):
    _folder, raw = checkpoint
    fused = tp2[rank][f"{ATTN}.qkv_proj.weight"]
    parts = [
        raw[f"{RAW_ATTN}.{p}_proj.weight"].chunk(2, dim=0)[rank] for p in ("q", "k", "v")
    ]
    # q carries the output gate and its rows are head-major, so a chunk is head-aligned.
    assert fused.shape == (QH * AHD + KVH * AHD, H)
    assert torch.equal(fused, torch.cat(parts, dim=0))


@pytest.mark.parametrize("rank", (0, 1))
def test_a_flat_chunk_of_the_fused_qkv_is_rejected(tp2, loaded, rank):
    """⛔⛆ The wrong answer has the RIGHT shape: rank 0's flat half is part of q and no k or v."""
    flat = loaded[f"{ATTN}.qkv_proj.weight"].chunk(2, dim=0)[rank]
    got = tp2[rank][f"{ATTN}.qkv_proj.weight"]
    assert flat.shape == got.shape  # this is why the comparison has to be on values
    assert not torch.equal(flat, got)


@pytest.mark.parametrize("rank", (0, 1))
def test_shared_expert_gate_up_is_sharded_per_projection(tp2, checkpoint, loaded, rank):
    _folder, raw = checkpoint
    got = tp2[rank][f"{SHARED}.gate_up_proj.weight"]
    gate, up = (raw[f"{RAW_SHARED}.{p}_proj.weight"].chunk(2, dim=0)[rank] for p in ("gate", "up"))
    assert torch.equal(got, torch.cat([gate, up], dim=0))
    flat = loaded[f"{SHARED}.gate_up_proj.weight"].chunk(2, dim=0)[rank]
    assert flat.shape == got.shape and not torch.equal(flat, got)  # ⛔⛆ same shape, wrong tensor


@pytest.mark.parametrize("rank", (0, 1))
def test_row_parallel_projections_shard_on_the_input_axis(tp2, checkpoint, rank):
    """`o_proj` and `down_proj` consume what their column-parallel producer emitted: dim 1."""
    _folder, raw = checkpoint
    for key, raw_key in ((f"{ATTN}.o_proj.weight", f"{RAW_ATTN}.o_proj.weight"),
                         (f"{SHARED}.down_proj.weight", f"{RAW_SHARED}.down_proj.weight")):
        got = tp2[rank][key]
        assert torch.equal(got, raw[raw_key].chunk(2, dim=1)[rank]), key
        assert got.shape[0] == raw[raw_key].shape[0], key  # the OUTPUT axis stays whole


def test_the_vocab_pair_shards_and_joins_back(tp2, loaded):
    """VOCAB is odd, so the real rows are unequal -- but the BUFFER is not.

    ⛔⛆ Corrected at bullet 6. ``VocabParallelEmbedding`` allocates ``div_ceil(11, 2) = 6`` rows
    on BOTH ranks (``ParallelLMHead`` all-gathers the logits, and a collective needs one shape),
    so the loader has to pad the last rank's five real rows rather than hand back a short tensor.
    """
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        rank0, rank1 = tp2[0][key], tp2[1][key]
        assert (rank0.shape[0], rank1.shape[0]) == (6, 6), key  # div_ceil(11, 2) on BOTH ranks
        joined = torch.cat([rank0, rank1], dim=0)
        assert torch.equal(joined[:VOCAB], loaded[key]), key
        assert not joined[VOCAB:].any(), key  # the pad is ZERO, not the buffer's torch.empty


def test_replicated_tensors_are_identical_on_both_ranks(tp2, loaded):
    """Everything not in `_TP_SHARDED` is a replicated module and every rank needs it whole."""
    rank0, rank1 = tp2
    assert set(rank0) == set(rank1) == set(loaded)
    # The parts vanish into their fusion, so the exclusion has to cover the FUSED key too --
    # derived from `_FUSIONS` rather than listed, or a new fusion silently escapes this test.
    sharded_fused = tuple(
        fused for fused, (parts, _pad) in _FUSIONS.items()
        if any(key.endswith(part) for part in parts for key in _SHARD_BEFORE_FUSE)
    )
    assert len(sharded_fused) == 2  # qkv_proj and the shared expert's gate_up_proj
    skip = _SHARD_BEFORE_FUSE + _SHARD_UNFUSED + sharded_fused
    for name in loaded:
        if name.endswith(skip):
            continue
        # The GDN is sharded by its own tables; only the head-dim-wide gated norm stays whole.
        if _GDN_INFIX in name and not name.endswith(_GDN_REPLICATED):
            continue
        assert torch.equal(rank0[name], loaded[name]), name
        assert torch.equal(rank1[name], loaded[name]), name


def test_the_shard_tables_agree_with_the_fusion_table():
    """A rename that moves a key between the two tables would silently shard after the concat."""
    fusion_parts = {part for parts, _pad in _FUSIONS.values() for part in parts}
    for key in _SHARD_BEFORE_FUSE:
        assert any(key.endswith(part) for part in fusion_parts), key
    for key in _SHARD_UNFUSED:
        assert not any(key.endswith(part) for part in fusion_parts), key


# ======================================================================================
# TP=2: the GDN's composite tensors, sharded per SUB-BLOCK (llm-server #777, bullet 5)
# ======================================================================================
#
# ⛔⛆ `in_proj_qkv.weight` and `conv1d.weight` are both laid out on
# `conv_dim = 2*key_dim + value_dim` -- here `[16 | 16 | 48]`, on the deployed geometry
# `[2048 | 2048 | 6144]`. A flat row chunk of either gives rank 0 all of q, all of k and a third
# of v: the right shape, plausible bf16, a different tensor. `shard_tensor` cannot catch it -- none
# of its substrings matches a `linear_attn` key, so it would return the tensor whole.
#
# The whole GDN moves together because `in_proj` is a four-part fusion (`qkv | z | b | a`):
# sharding `z`/`b`/`a` while leaving `qkv` whole emits a fused tensor that tiles nothing.

GDN = "model.layers.0.linear_attn"
RAW_GDN = "model.language_model.layers.0.linear_attn"
KEY_DIM, VALUE_DIM = KH * HD, VH * HD
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 80 = [16 | 16 | 48]


def _sub_block_shard(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """The reference answer: split `[key | key | value]` first, THEN halve each sub-block."""
    q, k, v = torch.split(tensor, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=0)
    return torch.cat([part.chunk(2, dim=0)[rank] for part in (q, k, v)], dim=0)


@pytest.mark.parametrize("rank", (0, 1))
def test_the_conv_weight_is_sharded_per_sub_block(tp2, checkpoint, rank):
    _folder, raw = checkpoint
    got = tp2[rank][f"{GDN}.conv1d.weight"]
    assert got.shape == (CONV_DIM // 2, 1, 4)
    assert torch.equal(got, _sub_block_shard(raw[f"{RAW_GDN}.conv1d.weight"], rank))


@pytest.mark.parametrize("rank", (0, 1))
def test_a_flat_chunk_of_the_conv_axis_is_rejected(tp2, loaded, rank):
    """⛔⛆ The wrong answer has the RIGHT shape: a flat half is all of q, all of k, a third of v."""
    for leaf in ("conv1d.weight", "in_proj.weight"):
        flat = loaded[f"{GDN}.{leaf}"].chunk(2, dim=0)[rank]
        got = tp2[rank][f"{GDN}.{leaf}"]
        assert flat.shape == got.shape, leaf  # which is why the comparison must be on values
        assert not torch.equal(flat, got), leaf


@pytest.mark.parametrize("rank", (0, 1))
def test_in_proj_fuses_four_rank_local_parts_in_order(tp2, checkpoint, rank):
    """`in_proj` is `qkv | z | b | a`, and each part is cut on its OWN axis before the concat."""
    _folder, raw = checkpoint
    fused = tp2[rank][f"{GDN}.in_proj.weight"]
    expected = torch.cat(
        [_sub_block_shard(raw[f"{RAW_GDN}.in_proj_qkv.weight"], rank)]
        + [raw[f"{RAW_GDN}.in_proj_{p}.weight"].chunk(2, dim=0)[rank] for p in ("z", "b", "a")],
        dim=0,
    )
    assert fused.shape == (CONV_DIM // 2 + VALUE_DIM // 2 + VH // 2 + VH // 2, H)
    assert torch.equal(fused, expected)


@pytest.mark.parametrize("rank", (0, 1))
def test_the_per_v_head_vectors_shard_with_the_v_heads(tp2, checkpoint, rank):
    """`A_log` and `dt_bias` are ONE ENTRY per v head and gate `b`/`a`, which are one row each."""
    _folder, raw = checkpoint
    for leaf in ("A_log", "dt_bias"):
        got = tp2[rank][f"{GDN}.{leaf}"]
        assert got.shape == (VH // 2,), leaf
        assert torch.equal(got, raw[f"{RAW_GDN}.{leaf}"].chunk(2, dim=0)[rank]), leaf


@pytest.mark.parametrize("rank", (0, 1))
def test_out_proj_shards_on_the_value_axis_it_consumes(tp2, checkpoint, rank):
    """Row-parallel: the v heads its producer column-split are its INPUT axis, dim 1."""
    _folder, raw = checkpoint
    full = raw[f"{RAW_GDN}.out_proj.weight"]
    got = tp2[rank][f"{GDN}.out_proj.weight"]
    assert torch.equal(got, full.chunk(2, dim=1)[rank])
    assert got.shape == (H, VALUE_DIM // 2)  # the OUTPUT axis stays whole


def test_the_gated_norm_stays_replicated(tp2, loaded):
    """`norm.weight` is `head_v_dim` wide -- a per-head WIDTH, not a per-head count."""
    key = f"{GDN}.norm.weight"
    assert loaded[key].shape == (HD,)
    assert torch.equal(tp2[0][key], loaded[key]) and torch.equal(tp2[1][key], loaded[key])


@pytest.mark.parametrize("rank", (0, 1))
def test_every_gdn_tensor_fits_the_module_buffer_it_loads_into(tp2, rank):
    """⛔⛆ The end of the chain: the loader and the module derive these widths INDEPENDENTLY
    (bullet 3 sized the module, bullet 5 the tensors), and a disagreement is a silent wrong
    weight, not a crash, on any axis a flat chunk happens to fit."""
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet

    with as_rank(rank, 2):
        gdn = Qwen4ExpGatedDeltaNet(
            hidden_size=H, num_k_heads=KH, num_v_heads=VH, head_k_dim=HD, head_v_dim=HD,
            conv_kernel_size=4, rms_norm_eps=1e-6, layer_id=0, output_gate="sigmoid",
        )
    got = tp2[rank]
    assert got[f"{GDN}.in_proj.weight"].shape == gdn.in_proj.weight.shape
    assert gdn._in_proj_split == [gdn.conv_dim, gdn.value_dim, gdn.num_v_heads, gdn.num_v_heads]
    assert got[f"{GDN}.conv1d.weight"].shape == gdn.conv1d.weight.shape
    assert got[f"{GDN}.A_log"].shape == gdn.A_log.shape
    assert got[f"{GDN}.dt_bias"].shape == gdn.dt_bias.shape
    assert got[f"{GDN}.out_proj.weight"].shape == gdn.out_proj.weight.shape
    assert got[f"{GDN}.norm.weight"].shape == gdn.norm.weight.shape


def test_an_unclassified_gdn_tensor_raises_instead_of_replicating(checkpoint):
    """⛔⛆ A GDN tensor nobody classified would otherwise load WHOLE into a rank-local buffer."""
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.weight import _shard_for_rank
    from freetoken.utils import cached_load_hf_config

    folder, _raw = checkpoint
    with as_rank(0, 1):
        config = parse_config(cached_load_hf_config(folder))
    with as_rank(0, 2), pytest.raises(NotImplementedError, match="not classified"):
        _shard_for_rank(f"{GDN}.some_new_gate.weight", _bf16(VALUE_DIM, H), config=config)


def test_the_vision_tower_is_never_sharded(checkpoint):
    """⭐ Bullet 4: the tower stays WHOLE at TP>1 -- it is replicated, not tensor-parallel.

    The synthetic checkpoint is text-only, so no ``visual.*`` key reaches the round-trip fixtures
    and the replication walk above never sees one. This asks `_shard_for_rank` directly, on the
    real tower's key shapes: ⛔⛆ the danger is a SUFFIX collision, because the tower's own
    projections end in the same `.qkv.` / `.proj.` words the language model shards on.
    """
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.weight import _shard_for_rank
    from freetoken.utils import cached_load_hf_config

    folder, _raw = checkpoint
    with as_rank(0, 1):
        config = parse_config(cached_load_hf_config(folder))
    tower = (
        "visual.blocks.0.attn.qkv.weight",
        "visual.blocks.0.attn.proj.weight",
        "visual.blocks.0.mlp.gate_proj.weight",
        "visual.blocks.0.mlp.down_proj.weight",
        "visual.patch_embed.proj.weight",
        "visual.merger.mlp.0.weight",
    )
    for name in tower:
        tensor = _bf16(H, H)
        for rank in (0, 1):
            with as_rank(rank, 2):
                assert torch.equal(_shard_for_rank(name, tensor, config=config), tensor), name


def test_a_single_kv_head_is_replicated_rather_than_rejected(checkpoint):
    """⛔⛆ Regression: the "must come back smaller" assert has ONE legitimate exception.

    With fewer kv heads than ranks `shard_tensor` replicates instead of splitting, and at exactly
    one kv head the replicated slice IS the whole tensor -- which is what
    `div_even(1, tp, allow_replicate=True)` makes the module declare. Asserting "smaller"
    unconditionally rejected that geometry. The deployed checkpoint has 2 kv heads, so this is
    the toy config's job to cover.
    """
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.weight import _shard_for_rank

    with as_rank(0, 1):
        config = parse_config(hf_config(num_kv=1))
    assert config.num_kv_heads == 1
    kv = _bf16(config.num_kv_heads * config.head_dim, H)
    for name in (".self_attn.k_proj.weight", ".self_attn.v_proj.weight"):
        for rank in (0, 1):
            with as_rank(rank, 2):
                # every rank gets the same single head, and that head is the whole projection
                assert torch.equal(_shard_for_rank(f"model.layers.3{name}", kv, config=config), kv)
    # ⛔⛆ The negative control: q_proj is NOT kv-replicated, so it must still come back narrower.
    q = _bf16(config.num_qo_heads * config.head_dim, H)
    with as_rank(0, 2):
        got = _shard_for_rank("model.layers.3.self_attn.q_proj.weight", q, config=config)
    assert got.shape[0] == q.shape[0] // 2


def test_the_gdn_tables_cover_every_gdn_tensor_the_loader_emits(loaded):
    """A new GDN key must be classified deliberately; this is what makes the raise above reachable
    only for genuinely new tensors rather than for something already shipping."""
    classified = _GDN_CONV_COMPOSITE + _GDN_VALUE_ROWS + _GDN_VALUE_COLUMNS + _GDN_REPLICATED
    emitted = {n.split(_GDN_INFIX, 1)[1] for n in loaded if _GDN_INFIX in n}
    # `in_proj_{qkv,z,b,a}` vanish into their fusion, so what the loader EMITS is `in_proj.weight`.
    assert emitted == {"in_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight",
                       "out_proj.weight"}
    assert emitted - {"in_proj.weight"} <= set(classified)
    assert set(_FUSIONS[".linear_attn.in_proj.weight"][0]) == {
        f".linear_attn.{leaf}" for leaf in
        ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")
    }
    for part in _FUSIONS[".linear_attn.in_proj.weight"][0]:
        assert part.split(_GDN_INFIX, 1)[1] in classified, part


# ======================================================================================
# TP=2: every loader key matches its model BUFFER (llm-server #777, bullet 6)
# ======================================================================================
#
# ⭐ The closing cross-check of the whole loader. Bullets 2-5 each pinned one module's widths
# against the tensors that fill it; this walks the WHOLE model, built on the meta device under
# `as_rank`, against everything `iter_weights` emits. #725 banked 290 shape mismatches at
# `--tp-size 2`; this is the instrument that has to read zero.
#
# ⛔⛆ Names alone are not the test. Every one of #725's 290 mismatches had the RIGHT NAME -- the
# loader emitted the model's own state-dict key and a tensor of the wrong width -- so the walk
# has to compare SHAPES, at every rank. `model_buffer_shapes` (common.py) is that walk, shared
# with `test_weight_ckpt.py`, which runs the same comparison against the REAL checkpoint.


@pytest.fixture(scope="module")
def buffers(checkpoint) -> dict[tuple[int, int], dict[str, tuple[int, ...]]]:
    folder, _raw = checkpoint
    return {(r, s): model_buffer_shapes(folder, r, s) for r, s in ((0, 1), (0, 2), (1, 2))}


@pytest.mark.parametrize("rank, size", [(0, 1), (0, 2), (1, 2)])
def test_the_loader_emits_exactly_the_model_state_dict(buffers, tp2, loaded, rank, size):
    """The emitted NAMES are the model's own keys, minus the offloaded routed experts."""
    emitted = set(loaded if size == 1 else tp2[rank])
    declared = {k for k in buffers[(rank, size)] if not k.endswith(EXPERT_BUFFERS)}
    assert emitted == declared


@pytest.mark.parametrize("rank, size", [(0, 1), (0, 2), (1, 2)])
def test_every_loader_tensor_fits_the_buffer_it_loads_into(buffers, tp2, loaded, rank, size):
    """⛔⛆ #725's 290 mismatches, at zero. `(0, 1)` is the negative control: TP=1 never moved."""
    emitted = loaded if size == 1 else tp2[rank]
    declared = buffers[(rank, size)]
    mismatched = {
        name: (tuple(tensor.shape), declared[name])
        for name, tensor in emitted.items()
        if tuple(tensor.shape) != declared[name]
    }
    assert mismatched == {}


def test_tp2_actually_narrows_the_model(buffers):
    """⛔⛆ Guard on the instrument: a walk that compares a TP=1 model to a TP=1 loader would pass
    while proving nothing. At least the mixers, the vocab pair and the MoE must have moved."""
    tp1, tp2_r0 = buffers[(0, 1)], buffers[(0, 2)]
    assert set(tp1) == set(tp2_r0)
    narrowed = {k for k in tp1 if tp2_r0[k] != tp1[k]}
    assert f"{GDN}.in_proj.weight" in narrowed
    assert f"{GDN}.conv1d.weight" in narrowed
    assert f"{GDN}.out_proj.weight" in narrowed
    assert f"{ATTN}.qkv_proj.weight" in narrowed
    assert f"{ATTN}.o_proj.weight" in narrowed
    assert {"model.embed_tokens.weight", "lm_head.weight"} <= narrowed
    assert f"{SHARED}.gate_up_proj.weight" in narrowed


def test_the_two_ranks_declare_the_same_buffer_shapes(buffers):
    """Every collective in this model is shape-symmetric, so the two ranks must agree -- and on
    an ODD vocab they only do because the module pads its partition, which is what forced the
    loader's own pad."""
    rank0, rank1 = buffers[(0, 2)], buffers[(1, 2)]
    assert rank0 == rank1
