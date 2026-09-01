"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
def test_decode_graph_replay_matches_eager():
    config = parsed_config()
    fixture = Fixture(config, num_pages=256)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))


# ======================================================================================
# The TP=2 gate (llm-server #777): can the sparse backend run on a per-rank head subset?
# ======================================================================================
#
# ⛔⛆ This is the gate the whole two-card ticket hangs on, and it is a REAL question, not a
# formality. ``Qwen4ExpIndexer.index_qk_proj`` is ``LinearReplicated``, so at TP=2 the indexer
# still emits index state for ALL heads while a column-sharded ``qkv_proj`` hands the backend
# only this rank's heads. If ``qsa_forward`` carries any assumption about the full head count --
# in the pool's page geometry, in the block scoring, or in the attend kernel's ``rep = HQ // KVH``
# tiling -- then sharding the mixers is wasted work and the ticket becomes a recorded no.
#
# The geometry here is the DEPLOYED one, not a toy: 24 q heads / 2 kv heads / head_dim 256, which
# leaves exactly ONE kv head per rank at TP=2 with no margin. ``rep`` is 12 on both arms.
# Lengths run past ``index_budget + index_ratio - 1`` so the sparse selection path is genuinely
# exercised rather than collapsing to dense.

_FULL_Q, _FULL_KV, _TP = 24, 2, 2
_LOCAL_Q, _LOCAL_KV = _FULL_Q // _TP, _FULL_KV // _TP
_GATE_HEAD_DIM = 256
_GATE_LEN = 2600  # > index_budget (2048) + index_ratio - 1, so selection is really sparse


def _gate_config(num_q: int, num_kv: int):
    return parsed_config(num_q=num_q, num_kv=num_kv, head_dim=_GATE_HEAD_DIM)


def _gate_tensors(device, dtype, steps: int, seed: int = 4242):
    """One full-head q/k/v set plus the replicated indexer state, shared by both arms."""
    from freetoken.models.qwen4_exp.attention import QSAIndexerInputs

    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen) * 0.5

    rows = _GATE_LEN + steps
    q = randn(rows, _FULL_Q, _GATE_HEAD_DIM)
    k = randn(rows, _FULL_KV, _GATE_HEAD_DIM)
    v = randn(rows, _FULL_KV, _GATE_HEAD_DIM)
    # The indexer is replicated, so every rank sees exactly these -- all 4 index heads.
    index = QSAIndexerInputs(
        q=randn(rows, 4, 128).contiguous(),
        k=randn(rows, 128).contiguous(),
        q_norm_weight=randn(128).float(),
        k_norm_weight=randn(128).float(),
        eps=1e-6,
    )
    return q, k, v, index


def _index_rows(index, rows: slice):
    """This forward's slice of the replicated indexer state (it is per-token, not per-head)."""
    from dataclasses import replace

    return replace(index, q=index.q[rows].contiguous(), k=index.k[rows].contiguous())


def _slice_heads(q, k, v, rank: int):
    """The heads rank ``rank`` owns under a head-aligned column shard of ``qkv_proj``."""
    qs = slice(rank * _LOCAL_Q, (rank + 1) * _LOCAL_Q)
    kvs = slice(rank * _LOCAL_KV, (rank + 1) * _LOCAL_KV)
    return q[:, qs], k[:, kvs], v[:, kvs]


def _gate_run(num_q, num_kv, q, k, v, index, steps: int):
    """Prefill ``_GATE_LEN`` tokens then ``steps`` decode steps; return every phase's output."""
    config = _gate_config(num_q, num_kv)
    fixture = Fixture(config, num_pages=128, dtype=torch.bfloat16)
    req = fixture.req(0, 0, _GATE_LEN)
    outputs = []
    for step in range(steps + 1):
        if step:
            fixture.step(req)
        batch = fixture.batch([req], "prefill" if step == 0 else "decode")
        rows = slice(req.cached_len, req.device_len)
        outputs.append(
            fixture.backend.qsa_forward(
                q[rows].contiguous(),
                k[rows].reshape(-1, num_kv * _GATE_HEAD_DIM).contiguous(),
                v[rows].reshape(-1, num_kv * _GATE_HEAD_DIM).contiguous(),
                _index_rows(index, rows),
                QSA_LAYER,
                batch,
            ).clone()
        )
    return outputs


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
def test_qsa_backend_on_a_rank_local_head_subset_matches_the_full_head_slice(rank):
    """⛔⛆ #777's gate: the backend on a head SUBSET == the full-head run, restricted to it.

    Both arms are fed identical tokens, an identical page layout and the identical (replicated)
    indexer state; the only difference is that the second arm sees ``_LOCAL_Q``/``_LOCAL_KV``
    heads instead of the full count, exactly as a column-sharded ``qkv_proj`` would hand them
    over. Attention is independent per head, so the subset output must equal the corresponding
    slice of the full output -- anything else means the backend is carrying a full-head
    assumption and the two-card design cannot use it.
    """
    device = torch.device("cuda")
    steps = 3
    q, k, v, index = _gate_tensors(device, torch.bfloat16, steps)

    full = _gate_run(_FULL_Q, _FULL_KV, q, k, v, index, steps)
    local = _gate_run(_LOCAL_Q, _LOCAL_KV, *_slice_heads(q, k, v, rank), index, steps)

    assert len(full) == len(local) == steps + 1
    heads = slice(rank * _LOCAL_Q, (rank + 1) * _LOCAL_Q)
    for step, (got, reference) in enumerate(zip(local, full)):
        assert got.shape == reference[:, heads].shape, (
            f"step {step}: subset arm returned {tuple(got.shape)}, "
            f"expected {tuple(reference[:, heads].shape)}"
        )
        torch.testing.assert_close(
            got.float(),
            reference[:, heads].float(),
            rtol=2e-2,
            atol=2e-2,
            msg=lambda m, step=step: f"{'prefill' if step == 0 else f'decode step {step}'}: {m}",
        )


@requires_cuda
def test_the_head_subset_gate_can_actually_fail():
    """Negative control for the gate above: comparing the WRONG rank's heads must NOT match.

    ⛔⛆ Without this the gate is an instrument that prints a pass regardless of sign. Rank 1's
    subset run is compared against rank 0's slice of the full-head output; attention is
    per-head, so those are different heads over the same tokens and must disagree.
    """
    device = torch.device("cuda")
    q, k, v, index = _gate_tensors(device, torch.bfloat16, steps=0)

    full = _gate_run(_FULL_Q, _FULL_KV, q, k, v, index, steps=0)[0]
    rank1 = _gate_run(_LOCAL_Q, _LOCAL_KV, *_slice_heads(q, k, v, 1), index, steps=0)[0]

    wrong = full[:, 0:_LOCAL_Q]  # rank 0's heads, against rank 1's run
    assert not torch.allclose(rank1.float(), wrong.float(), rtol=2e-2, atol=2e-2), (
        "the head-subset comparison passes even against the wrong heads -- it proves nothing"
    )


@requires_cuda
def test_the_head_subset_gate_runs_a_genuinely_sparse_selection(monkeypatch):
    """The gate must not be answered by a length that collapses QSA to dense attention.

    QSA is exactly dense while a request sees at most ``index_budget + index_ratio - 1`` tokens
    (every complete block is selected). ``_GATE_LEN`` is chosen past that, and this pins it: at
    least one row must select strictly fewer tokens than its causal prefix.
    """
    device = torch.device("cuda")
    q, k, v, index = _gate_tensors(device, torch.bfloat16, steps=0)

    config = _gate_config(_FULL_Q, _FULL_KV)
    budget = config.qwen4_args.index_budget
    ratio = config.qwen4_args.index_ratio
    assert _GATE_LEN > budget + ratio - 1, (
        f"_GATE_LEN {_GATE_LEN} is inside the dense regime (budget {budget}, ratio {ratio})"
    )

    fixture = Fixture(config, num_pages=128, dtype=torch.bfloat16)
    seen = selection_spy(monkeypatch, fixture.backend)
    req = fixture.req(0, 0, _GATE_LEN)
    batch = fixture.batch([req], "prefill")
    fixture.backend.qsa_forward(
        q[: _GATE_LEN].contiguous(),
        k[: _GATE_LEN].reshape(_GATE_LEN, -1).contiguous(),
        v[: _GATE_LEN].reshape(_GATE_LEN, -1).contiguous(),
        _index_rows(index, slice(0, _GATE_LEN)),
        QSA_LAYER,
        batch,
    )
    indices = seen["indices"]
    selected = (indices >= 0).sum(dim=1)
    prefix = batch.positions.to(selected.dtype).to(selected.device) + 1
    assert bool((selected < prefix).any()), (
        "every row selected its whole causal prefix -- the gate ran on dense attention"
    )
