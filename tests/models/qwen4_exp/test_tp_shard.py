"""TP=2 shard equivalence for qwen4_exp (llm-server #725, on #724's design).

Two axes carry the tensor-parallel split and nothing else does: the routed experts' intermediate
dimension and the PLE's hash heads. Both are supposed to be algebraically exact -- the sharded
model must produce bit-for-bit what TP=1 produces, not merely something close -- so these tests
assert reconstruction, not tolerance.

⛔ The subtle one is the PLE gather. ``DistributedCommunicator.all_gather`` concatenates on dim 0,
but the PLE embedding needs its ranks' head slices side by side on the FEATURE axis, so
``NGramEmbedding.forward`` transposes the gathered buffer. A wrong transpose there still returns
the right shape and plausible values; only a comparison against the TP=1 reference catches it.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.nvfp4_banks import _intermediate_shard
from freetoken.models.qwen4_exp.ple import NGramEmbedding, GpuResidentTable, PLEMetadata

from .common import EOS, VOCAB, as_rank, hash_constants, parsed_config


# ----------------------------------------------------------------------------------
# The routed experts: intermediate-dim shard
# ----------------------------------------------------------------------------------


def test_intermediate_shard_partitions_I_exactly():
    """Every column of I is owned by exactly one rank, in order, with no gap or overlap."""
    I, tp = 640, 2
    windows = []
    for rank in range(tp):
        with as_rank(rank, tp):
            per_rank, lo = _intermediate_shard(I)
        windows.append((lo, lo + per_rank))
    assert windows == [(0, 320), (320, 640)]


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_intermediate_shard_reassembles_every_nvfp4_bank_axis(tp):
    """The six NVFP4 banks carry I on three different axes; each must tile back to the original.

    ``gate_up_*`` hold I as an unpacked row axis, ``down_packed`` as the 2-values-per-byte packed
    axis (I // 2) and ``down_scale`` as the 16-values-per-block scale axis (I // 16). A shard that
    forgets to divide by the packing factor reads the wrong bytes and still fits the buffer.
    """
    E, H, I = 2, 32, 640
    gate_up = torch.arange(E * I * (H // 2), dtype=torch.uint8).reshape(E, I, H // 2)
    down_packed = torch.arange(E * H * (I // 2), dtype=torch.uint8).reshape(E, H, I // 2)
    down_scale = torch.arange(E * H * (I // 16), dtype=torch.uint8).reshape(E, H, I // 16)

    parts = {"gate_up": [], "down_packed": [], "down_scale": []}
    for rank in range(tp):
        with as_rank(rank, tp):
            Ip, lo = _intermediate_shard(I)
        parts["gate_up"].append(gate_up[:, lo:lo + Ip])
        parts["down_packed"].append(down_packed[:, :, lo // 2:(lo + Ip) // 2])
        parts["down_scale"].append(down_scale[:, :, lo // 16:(lo + Ip) // 16])

    assert torch.equal(torch.cat(parts["gate_up"], dim=1), gate_up)
    assert torch.equal(torch.cat(parts["down_packed"], dim=2), down_packed)
    assert torch.equal(torch.cat(parts["down_scale"], dim=2), down_scale)


def test_intermediate_shard_rejects_a_split_that_breaks_the_nvfp4_block():
    """320 is a multiple of 16 so TP=2 is legal; a split that is not must fail loudly at load."""
    with as_rank(0, 8):
        with pytest.raises(ValueError, match="NVFP4 block size 16"):
            _intermediate_shard(80)  # 80 / 8 = 10, not a multiple of 16


# ----------------------------------------------------------------------------------
# The PLE: hash-head shard
# ----------------------------------------------------------------------------------


def _embedding(args, table_rows: torch.Tensor, row_lo: int = 0, row_hi: int | None = None):
    emb = NGramEmbedding(args)
    mult, sizes, offsets = hash_constants(args)
    emb.layer_multipliers = mult
    emb.ngram_heads_vocab_sizes = sizes
    emb.ngram_heads_offsets = offsets
    hi = table_rows.shape[0] if row_hi is None else row_hi
    emb.attach_table(GpuResidentTable(table_rows[row_lo:hi], dtype=torch.float32))
    return emb


def _decode_meta(batch: int, args) -> PLEMetadata:
    torch.manual_seed(11)
    return PLEMetadata(
        input_ids=torch.randint(0, VOCAB, (batch,), dtype=torch.int64),
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int64),
        seq_lens=[1] * batch,
        ngram_context=torch.randint(0, VOCAB, (batch, args.ngram_size - 1), dtype=torch.int64),
        state_slots=torch.arange(batch, dtype=torch.int64),
        fresh_slots=None,
        is_decode=True,
    )


def test_ple_head_shard_reconstructs_the_tp1_embedding_exactly():
    args = parsed_config().qwen4_args
    assert args.num_ngram_heads % 2 == 0
    _, sizes, offsets = hash_constants(args)
    total = int(offsets[-1] + sizes[-1])

    torch.manual_seed(3)
    table = torch.randn(total, args.ngram_head_dim, dtype=torch.float32)
    meta = _decode_meta(4, args)

    with as_rank(0, 1):
        reference = _embedding(args, table).forward(meta)
    assert reference.shape == (4, args.ple_embed_dim)

    # Each rank looks up only its own heads, against a table holding only its own rows.
    per_rank = args.num_ngram_heads // 2
    locals_ = []
    for rank in range(2):
        with as_rank(rank, 2):
            lo = int(offsets[per_rank * rank])
            hi = total if rank == 1 else int(offsets[per_rank * (rank + 1)])
            emb = _embedding(args, table, lo, hi)
            ids = emb.row_ids(meta)
            assert ids.shape == (4, per_rank)
            assert int(ids.min()) >= 0 and int(ids.max()) < hi - lo
            locals_.append(emb.table.lookup(ids))

    # Drive rank 1's real forward with an all_gather that returns what the group would produce,
    # so the dim-0 -> feature-axis transpose under test is the shipped one.
    gathered = torch.cat(locals_, dim=0)
    with as_rank(1, 2):
        emb = _embedding(args, table, int(offsets[per_rank]), total)
        emb._comm = _FakeComm(gathered)
        got = emb.forward(meta)

    assert got.shape == reference.shape
    assert torch.equal(got, reference)


def test_ple_head_shard_is_not_reproduced_by_a_naive_dim0_gather():
    """Guards the transpose: concatenating the gather on dim 0 has the right size and is wrong."""
    args = parsed_config().qwen4_args
    _, sizes, offsets = hash_constants(args)
    total = int(offsets[-1] + sizes[-1])
    torch.manual_seed(5)
    table = torch.randn(total, args.ngram_head_dim, dtype=torch.float32)
    meta = _decode_meta(4, args)

    with as_rank(0, 1):
        reference = _embedding(args, table).forward(meta)

    per_rank = args.num_ngram_heads // 2
    locals_ = []
    for rank in range(2):
        with as_rank(rank, 2):
            lo = int(offsets[per_rank * rank])
            hi = total if rank == 1 else int(offsets[per_rank * (rank + 1)])
            emb = _embedding(args, table, lo, hi)
            locals_.append(emb.table.lookup(emb.row_ids(meta)))

    naive = torch.cat(locals_, dim=0).reshape(reference.shape)
    assert not torch.equal(naive, reference)


class _FakeComm:
    """Stands in for the process group: ``all_gather`` returns the group's dim-0 concatenation."""

    def __init__(self, gathered: torch.Tensor) -> None:
        self._gathered = gathered

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self._gathered

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the PLE shard must not all-reduce; it all-gathers")


# ----------------------------------------------------------------------------------
# The QSA attention layer: rank-local head counts + a row-parallel ``o_proj``
# ----------------------------------------------------------------------------------
#
# ``qkv_proj`` is a ``LinearColParallelMerged``: it shards each declared output size by tp_size,
# so at TP=2 the GEMM it runs is HALF as wide. Everything the layer derives from the head counts
# -- ``num_q``, ``num_kv``, ``qo_attn_dim``, ``kv_attn_dim``, ``_qkv_split`` -- must shrink with
# it, and ``o_proj`` must all-reduce because each rank then holds a PARTIAL sum. ⛔⛆ Getting the
# split wrong is silent: ``[12288, 512, 512]`` applied to a 6656-wide result raises, but any
# split that merely sums to 6656 returns the right shape and plausible values.

DEPLOYED = dict(num_q=24, num_kv=2, head_dim=256, hidden=512)


def _attention(config, layer_id: int = 3):
    from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention

    return Qwen4ExpAttention(config, layer_id=layer_id)


def test_attention_tp1_geometry_is_unchanged():
    """Negative control: at TP=1 every width is still the full, pre-sharding one."""
    config = parsed_config(**DEPLOYED)
    with as_rank(0, 1):
        attn = _attention(config)

    assert (attn.num_q, attn.num_kv) == (24, 2)
    assert (attn.qo_attn_dim, attn.kv_attn_dim) == (6144, 512)
    assert attn._qkv_split == [12288, 512, 512]
    assert attn.qkv_proj.local_output_size == 13312
    assert attn.o_proj.local_input_size == 6144


@pytest.mark.parametrize("rank", [0, 1])
def test_attention_tp2_declares_rank_local_shapes(rank):
    """At TP=2 each rank owns 12 q heads and 1 kv head -- one kv head per rank, no margin."""
    config = parsed_config(**DEPLOYED)
    with as_rank(rank, 2):
        attn = _attention(config)

    assert (attn.num_q, attn.num_kv) == (12, 1)
    assert (attn.qo_attn_dim, attn.kv_attn_dim) == (3072, 256)
    assert attn._qkv_split == [6144, 256, 256]


@pytest.mark.parametrize("tp", [1, 2])
def test_attention_split_tiles_the_local_gemm_width(tp):
    """The split the layer applies must tile the width ``qkv_proj`` actually produces."""
    config = parsed_config(**DEPLOYED)
    for rank in range(tp):
        with as_rank(rank, tp):
            attn = _attention(config)
        assert sum(attn._qkv_split) == attn.qkv_proj.local_output_size
        assert attn.qkv_proj.weight.shape[0] == sum(attn._qkv_split)
        # ``o_proj`` consumes exactly the gated attention output this rank computes.
        assert attn.o_proj.local_input_size == attn.qo_attn_dim
        assert attn.o_proj.weight.shape[1] == attn.qo_attn_dim
        assert attn.qkv_proj.full_output_size == 13312
        assert attn.o_proj.full_input_size == 6144


@pytest.mark.parametrize("tp, expected_calls", [(1, 0), (2, 1)])
def test_attention_o_proj_all_reduces_only_when_sharded(tp, expected_calls, monkeypatch):
    """⛔ Each rank's ``o_proj`` output is a PARTIAL sum at TP>1, so it must all-reduce -- and
    must NOT at TP=1, where an unnecessary collective would cost 1.6 ms/token for nothing."""
    from freetoken.distributed.impl import DistributedCommunicator

    calls = []
    monkeypatch.setattr(
        DistributedCommunicator,
        "all_reduce",
        lambda self, x: (calls.append(x.shape), x)[1],
    )

    config = parsed_config(**DEPLOYED)
    with as_rank(0, tp):
        attn = _attention(config)
    attn.o_proj.weight = torch.zeros_like(attn.o_proj.weight)
    attn.o_proj.forward(torch.zeros(2, attn.qo_attn_dim))

    assert len(calls) == expected_calls


# ----------------------------------------------------------------------------------
# The GDN (GatedDeltaNet) layer: rank-local head counts + a row-parallel ``out_proj``
# ----------------------------------------------------------------------------------
#
# Same shape of bug as the attention layer, one level worse: ``conv_dim`` is not a head count
# but a CONCATENATION, ``2 * key_dim + value_dim``, and the conv weight, the conv state slot and
# ``_in_proj_split`` are all sized from it. ``b`` and ``a`` are one column per v head, and
# ``dt_bias``/``A_log`` one entry per v head, so they shard with the v heads and nothing else.
#
# ⛔⛆ The module's local ``conv_dim`` must be exactly what ``LinearStatePool`` allocates a conv
# state for; the two derive it independently, and a disagreement is a silent state-shape bug.

# The deployed GDN geometry (hidden scaled down -- it does not enter the sharding math).
GDN = dict(
    hidden_size=256, num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128,
    conv_kernel_size=4, rms_norm_eps=1e-6, layer_id=0, output_gate="sigmoid",
)


def _gdn(**overrides):
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet

    return Qwen4ExpGatedDeltaNet(**{**GDN, **overrides})


def test_gdn_tp1_geometry_is_unchanged():
    """Negative control: at TP=1 every width is still the full, pre-sharding one."""
    with as_rank(0, 1):
        gdn = _gdn()

    assert (gdn.num_k_heads, gdn.num_v_heads) == (16, 48)
    assert (gdn.key_dim, gdn.value_dim, gdn.conv_dim) == (2048, 6144, 10240)
    assert gdn._in_proj_split == [10240, 6144, 48, 48]
    assert gdn.conv1d.weight.shape == (10240, 1, 4)
    assert gdn.dt_bias.shape == gdn.A_log.shape == (48,)
    assert gdn.out_proj.local_input_size == 6144


@pytest.mark.parametrize("rank", [0, 1])
def test_gdn_tp2_declares_rank_local_shapes(rank):
    """At TP=2 each rank owns 8 k heads and 24 v heads, and conv_dim halves WITH them."""
    with as_rank(rank, 2):
        gdn = _gdn()

    assert (gdn.num_k_heads, gdn.num_v_heads) == (8, 24)
    assert (gdn.key_dim, gdn.value_dim, gdn.conv_dim) == (1024, 3072, 5120)
    assert gdn._in_proj_split == [5120, 3072, 24, 24]
    assert gdn.conv1d.weight.shape == (5120, 1, 4)
    assert gdn.dt_bias.shape == gdn.A_log.shape == (24,)
    assert gdn.out_proj.local_input_size == 3072


@pytest.mark.parametrize("tp", [1, 2])
def test_gdn_split_tiles_the_local_gemm_width(tp):
    """The split the layer applies must tile the width ``in_proj`` actually produces, and every
    tensor sized off a head count must agree with it."""
    for rank in range(tp):
        with as_rank(rank, tp):
            gdn = _gdn()
        assert sum(gdn._in_proj_split) == gdn.in_proj.local_output_size
        assert gdn.in_proj.weight.shape[0] == sum(gdn._in_proj_split)
        assert gdn.in_proj.full_output_size == 10240 + 6144 + 48 + 48
        assert gdn.conv_dim == 2 * gdn.key_dim + gdn.value_dim
        assert gdn.conv1d.weight.shape[0] == gdn.conv_dim
        assert gdn.out_proj.local_input_size == gdn.value_dim
        assert gdn.out_proj.weight.shape[1] == gdn.value_dim
        assert gdn.out_proj.full_input_size == 6144


@pytest.mark.parametrize("tp", [1, 2])
def test_gdn_conv_dim_matches_the_state_pool_allocation(tp):
    """⛔⛆ The module and ``LinearStatePool`` derive the local conv_dim independently. If they
    disagree the conv state slot is the wrong width -- right shape nowhere, wrong answer or a
    stride error deep inside the fused kernel."""
    from freetoken.kvcache.linear_state_pool import _linear_local_dims
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="gdn", layer_ids=(0,), num_key_heads=GDN["num_k_heads"],
        num_value_heads=GDN["num_v_heads"], key_head_dim=GDN["head_k_dim"],
        value_head_dim=GDN["head_v_dim"], conv_kernel_dim=GDN["conv_kernel_size"],
        output_gate=GDN["output_gate"],
    )
    _, pool_conv_dim, pool_v_heads = _linear_local_dims(group, tp)

    with as_rank(0, tp):
        gdn = _gdn()

    assert gdn.conv_dim == pool_conv_dim
    assert gdn.num_v_heads == pool_v_heads


@pytest.mark.parametrize("tp, expected_calls", [(1, 0), (2, 1)])
def test_gdn_out_proj_all_reduces_only_when_sharded(tp, expected_calls, monkeypatch):
    """Each rank's core output covers its own v heads only, so ``out_proj`` holds a PARTIAL sum
    at TP>1 and must all-reduce -- and must not at TP=1."""
    from freetoken.distributed.impl import DistributedCommunicator

    calls = []
    monkeypatch.setattr(
        DistributedCommunicator, "all_reduce", lambda self, x: (calls.append(x.shape), x)[1]
    )

    with as_rank(0, tp):
        gdn = _gdn()
    gdn.out_proj.weight = torch.zeros_like(gdn.out_proj.weight)
    gdn.out_proj.forward(torch.zeros(2, gdn.value_dim))

    assert len(calls) == expected_calls


# ----------------------------------------------------------------------------------
# ⛔⛆ The quantized dense paths are REFUSED at TP>1, not sharded
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expert_quant, attn_quant",
    [("fp8_block", "none"), ("none", "fp8_pertensor"), ("none", "nvfp4")],
)
def test_row_parallel_quant_refuses_every_quantized_mode_at_tp2(expert_quant, attn_quant):
    """None of the three quantized linears is TP-aware on its input axis and none all-reduces:
    block-FP8 carries a [out/128, in/128] scale grid, per-tensor FP8 one scale fitted to the FULL
    row, NVFP4 packs two values per byte along that same axis. Splitting ``in_f`` on any of them
    returns a layer of the right shape that computes the wrong number, so it must raise."""
    from freetoken.models.quant_linear import make_row_parallel_quant

    with as_rank(0, 2), pytest.raises(NotImplementedError):
        make_row_parallel_quant(expert_quant, attn_quant, 6144, 256)


def test_row_parallel_quant_is_row_parallel_for_the_deployed_bf16_case():
    """On THIS checkpoint the GDN sits in the modelopt ignore list, so it takes the bf16 path --
    the only one that is actually sharded."""
    from freetoken.layers import LinearRowParallel
    from freetoken.models.quant_linear import make_row_parallel_quant

    for tp, expect_local in ((1, 6144), (2, 3072)):
        with as_rank(0, tp):
            layer = make_row_parallel_quant("nvfp4", "none", 6144, 256)
        assert isinstance(layer, LinearRowParallel)
        assert layer.local_input_size == expect_local


def test_gdn_fp8_in_proj_is_refused_at_tp2():
    """The fp8 input projections are not TP-aware either; the GDN refuses rather than build one."""
    with as_rank(0, 2), pytest.raises(NotImplementedError):
        _gdn(expert_quant="fp8_block")
