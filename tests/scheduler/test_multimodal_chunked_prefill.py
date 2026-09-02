"""Chunked multimodal prefill: an image prompt no longer has to fit one prefill pass.

The model scatters a batch's ``mm_embeds`` POSITIONALLY over the ``image_token_id`` slots of
the batch's ``input_ids``, and a prefill batch carries only the current chunk's ids -- so the
scheduler must hand every chunk exactly the soft-token rows its own placeholders consume, in
prompt order, and nothing else. These tests pin that on the CPU harness, with rows numbered so
a wrong slice is visible, and they pin the two things chunking must NOT change: a multimodal
request stays out of the shared prefix cache (image placeholders share an id across pictures,
so a prefix hit would serve the wrong picture's KV), and a text prompt in the same batch is
untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder, slice_mm_embeds
from freetoken.scheduler.utils import PendingReq

IMG = 151655  # any id will do; the tests never touch a tokenizer
TXT = 7
CHUNK = 8
HIDDEN = 4
WIDTH = 128
MAX_RUNNING = 4


def _setup_context():
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        return get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=1)
        set_global_ctx(ctx)
        return ctx


def _linear_pool(num_slots=16):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _build_managers(cache_type: str, num_pages: int = 256):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    pool = _linear_pool() if cache_type == "hybrid_radix" else None
    cm = CacheManager(num_pages=num_pages, page_size=1, page_table=pt, type=cache_type,
                      linear_state_pool=pool)
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm, image_token_id=IMG)
    return cm, tm, dm, pm


def _prompt(layout: str) -> torch.Tensor:
    """'t' is a text token, 'I' an image placeholder."""
    return torch.tensor([IMG if c == "I" else TXT for c in layout], dtype=torch.int32)


def _rows(n: int) -> torch.Tensor:
    """Row k holds the value k in every column, so a slice names the rows it carries."""
    return torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, HIDDEN).contiguous()


def _row_ids(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t[:, 0]] if t.numel() else []


# One image whose 8-slot run STRADDLES the chunk-1/chunk-2 boundary, a chunk with no image at
# all, and a second image in the final chunk: 40 tokens, 5 chunks of 8.
#            chunk 0  | chunk 1  | chunk 2  | chunk 3  | chunk 4
LAYOUT = "tttttttt" + "ttttIIII" + "IIIItttt" + "tttttttt" + "tIIIIttt"
EXPECTED_ROWS = [[], [0, 1, 2, 3], [4, 5, 6, 7], [], [8, 9, 10, 11]]
N_ROWS = 12


def _drive(cm, tm, pm, pending: PendingReq, budget: int = CHUNK):
    """Schedule chunk after chunk the way the scheduler does (allocate, forward, next), and
    return each batch's reqs in order. The final chunk is a plain Req; it is cached once."""
    pm.pending_list = [pending]
    seen = []
    while pm.runnable:
        batch = pm.schedule_next_batch(budget)
        assert batch is not None, "a multimodal prompt over the budget must chunk, not stall"
        cm.allocate_paged(batch.reqs)
        for r in batch.reqs:
            r.chunk_ids = r.input_ids[r.cached_len : r.device_len].clone()  # what the forward saw
            r.complete_one()
        seen.append(batch.reqs)
    return seen


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_each_chunk_is_handed_exactly_its_own_rows_in_order(cache_type):
    cm, tm, _dm, pm = _build_managers(cache_type)
    fresh_slots = cm.linear_state_pool.num_free_slots if cache_type == "hybrid_radix" else None
    pending = PendingReq(uid=1, input_ids=_prompt(LAYOUT),
                         sampling_params=SamplingParams(max_tokens=4), mm_embeds=_rows(N_ROWS))
    batches = _drive(cm, tm, pm, pending)
    assert len(batches) == 5 and all(len(b) == 1 for b in batches)
    reqs = [b[0] for b in batches]

    # every chunk carries a tensor (never None: that is the cache manager's "has an image")
    assert all(r.mm_embeds is not None for r in reqs)
    assert [_row_ids(r.mm_embeds) for r in reqs] == EXPECTED_ROWS
    # ... and the slots in each chunk's ids match the rows it was handed, one to one
    for r in reqs:
        assert int((r.chunk_ids == IMG).sum()) == r.mm_embeds.shape[0]
    assert torch.equal(torch.cat([r.chunk_ids for r in reqs]), _prompt(LAYOUT))
    assert all(isinstance(r, ChunkedReq) for r in reqs[:-1])
    assert type(reqs[-1]) is Req

    # The wrong-image guard survives chunking: the finished prompt is NOT inserted into the
    # shared prefix cache, and its pages come back when the request is released.
    final = reqs[-1]
    cm.cache_req(final, finished=False)
    assert cm.prefix_cache.size_info.evictable_size == 0
    cm.cache_req(final, finished=True)
    tm.free(final.table_idx)
    cm.check_integrity()
    assert len(cm.free_slots) == cm.num_pages
    if cache_type == "hybrid_radix":  # live + ping-pong slots all returned (slot 0 is padding)
        assert cm.linear_state_pool.num_free_slots == fresh_slots


def test_an_unchunked_image_prompt_gets_every_row_as_before():
    cm, tm, _dm, pm = _build_managers("radix")
    pending = PendingReq(uid=2, input_ids=_prompt("ttIIIIt"),
                         sampling_params=SamplingParams(max_tokens=2), mm_embeds=_rows(4))
    (reqs,) = _drive(cm, tm, pm, pending, budget=64)
    assert type(reqs[0]) is Req and _row_ids(reqs[0].mm_embeds) == [0, 1, 2, 3]


def test_a_text_prompt_sharing_the_batch_is_untouched_and_rows_follow_request_order():
    from freetoken.scheduler.scheduler import Scheduler

    cm, _tm, _dm, pm = _build_managers("radix")
    pm.pending_list = [
        PendingReq(10, _prompt("tttt"), SamplingParams(max_tokens=2)),
        PendingReq(11, _prompt("tIIt"), SamplingParams(max_tokens=2), mm_embeds=_rows(2) + 100),
        PendingReq(12, _prompt("Itt"), SamplingParams(max_tokens=2), mm_embeds=_rows(1) + 200),
    ]
    batch = pm.schedule_next_batch(64)
    assert [r.uid for r in batch.reqs] == [10, 11, 12]
    assert batch.reqs[0].mm_embeds is None
    Scheduler._gather_multimodal(None, batch)  # only reads the batch
    assert _row_ids(batch.mm_embeds) == [100, 101, 200]
    # the concatenation lines up with the batch's ids: one row per slot, in order
    ids = torch.cat([r.input_ids[r.cached_len:] for r in batch.reqs])
    assert int((ids == IMG).sum()) == batch.mm_embeds.shape[0]


def test_a_cached_prefix_skips_the_rows_its_placeholders_already_consumed():
    """A prefix hit on a later turn (the image-aware cache, patch 0009) resumes after the
    prefix's placeholders: the offset counts slots in [0, cached_len), the length in the chunk."""
    ids = _prompt("tIIt" + "tIIIt" + "It")
    rows = _rows(6)
    mid, last = dict(is_last_chunk=False), dict(is_last_chunk=True)
    assert _row_ids(slice_mm_embeds(ids, rows, IMG, 4, 5, **mid)) == [2, 3, 4]
    assert _row_ids(slice_mm_embeds(ids, rows, IMG, 9, 2, **last)) == [5]
    assert _row_ids(slice_mm_embeds(ids, rows, IMG, 0, 11, **last)) == list(range(6))
    assert slice_mm_embeds(ids, rows, IMG, 0, 1, **mid).shape == (0, HIDDEN)
    with pytest.raises(AssertionError, match="exceed vision features"):
        slice_mm_embeds(ids, _rows(5), IMG, 0, 11, **last)


def test_the_last_chunk_refuses_surplus_rows_that_no_slot_consumed():
    """The offline API attaches the tensor unchecked; a middle chunk cannot know the prompt's
    total, the last one can -- and must, or a surplus row is dropped with every assert green."""
    ids = _prompt("tIIt" + "tIIIt" + "It")
    surplus = _rows(7)
    assert _row_ids(slice_mm_embeds(ids, surplus, IMG, 0, 4, is_last_chunk=False)) == [0, 1]
    with pytest.raises(AssertionError, match="exceed image-token slots"):
        slice_mm_embeds(ids, surplus, IMG, 0, 11, is_last_chunk=True)
    with pytest.raises(AssertionError, match="exceed image-token slots"):
        slice_mm_embeds(ids, surplus, IMG, 9, 2, is_last_chunk=True)


def test_the_adder_no_longer_needs_a_full_pass_budget():
    """The deferral (and its NotImplementedError) is gone: a multimodal prompt admits on whatever
    budget is left this pass, exactly like a text prompt."""
    assert "full_token_budget" not in {f.name for f in PrefillAdder.__dataclass_fields__.values()}
    cm, tm, _dm, pm = _build_managers("radix")
    pending = PendingReq(uid=3, input_ids=_prompt(LAYOUT),
                         sampling_params=SamplingParams(max_tokens=4), mm_embeds=_rows(N_ROWS))
    batches = _drive(cm, tm, pm, pending, budget=3)  # smaller than any image run
    assert torch.equal(torch.cat([b[0].mm_embeds for b in batches]), _rows(N_ROWS))


def test_a_multimodal_prompt_on_a_model_without_a_placeholder_id_is_an_engine_bug():
    """The tokenizer worker refuses an image for such a model, so the adder never sees one; if
    it does, it must not hand the tensor over whole (nothing would scatter, the answer would
    come from a blank) -- it fails loudly instead."""
    cm, tm, _dm, pm = _build_managers("radix")
    pm.image_token_id = None
    pm.pending_list = [PendingReq(uid=4, input_ids=_prompt("ttttIIttttII"),
                                  sampling_params=SamplingParams(max_tokens=2), mm_embeds=_rows(4))]
    with pytest.raises(AssertionError, match="declares no image_token_id"):
        pm.schedule_next_batch(5)


@pytest.mark.parametrize("model_module, model_class", [
    ("freetoken.models.qwen4_exp.model", "Qwen4ExpModel"),
    ("freetoken.models.gemma4.model", "Gemma4Model"),
])
def test_the_scatter_holds_per_chunk_including_an_empty_one(model_module, model_class):
    """The model's merge, run on the CPU with the rows a chunk was handed: the per-chunk
    invariant (slots in THIS chunk == rows handed) replaces "must not be split"."""
    import importlib

    model_cls = getattr(importlib.import_module(model_module), model_class)
    ctx = _setup_context()
    fake = SimpleNamespace(_image_token_id=IMG)
    ids = _prompt(LAYOUT)
    rows = _rows(N_ROWS)
    for start, expected in zip(range(0, len(LAYOUT), CHUNK), EXPECTED_ROWS):
        chunk_ids = ids[start : start + CHUNK]
        ctx._batch = SimpleNamespace(mm_embeds=slice_mm_embeds(
            ids, rows, IMG, start, CHUNK, is_last_chunk=start + CHUNK >= len(LAYOUT)))
        x = torch.full((CHUNK, HIDDEN), -1.0)
        out = model_cls._merge_multimodal(fake, chunk_ids, x)
        assert _row_ids(out[chunk_ids == IMG]) == expected
        assert torch.all(out[chunk_ids != IMG] == -1.0)
    # the guard still fires when a chunk is handed the wrong rows
    ctx._batch = SimpleNamespace(mm_embeds=rows)
    with pytest.raises(AssertionError, match="exactly its own soft-token rows"):
        model_cls._merge_multimodal(fake, ids[:CHUNK], torch.zeros(CHUNK, HIDDEN))
    ctx._batch = None
