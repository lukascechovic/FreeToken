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

import contextlib

import pytest
import torch

from freetoken.models.nvfp4_banks import _intermediate_shard
from freetoken.models.qwen4_exp.ple import NGramEmbedding, GpuResidentTable, PLEMetadata

from .common import EOS, VOCAB, hash_constants, parsed_config


@contextlib.contextmanager
def as_rank(rank: int, size: int):
    """Run the block as one rank of a ``size``-way TP group (set_tp_info is write-once)."""
    from freetoken.distributed import info

    saved = info._TP_INFO
    info._TP_INFO = info.DistributedInfo(rank=rank, size=size)
    try:
        yield
    finally:
        info._TP_INFO = saved


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
