"""Image-aware prefix-cache keys (llm-server #791): a session with a picture in it keeps its
prefix, and a different picture can never hit the first picture's KV.

CPU harness, both served cache types (plain radix and the hybrid GDN radix), driven the way the
scheduler drives them (schedule, allocate, forward, commit)."""
from __future__ import annotations

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.mm_key import (
    MARKERS_PER_IMAGE, extend_key, image_cache_key_ids, image_digest, placeholder_runs,
)
from freetoken.scheduler.utils import PendingReq

IMG = 151655
TXT = 7
CHUNK = 8
HIDDEN = 4
WIDTH = 128
MAX_RUNNING = 4
PATCH = 6


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
    return torch.tensor([IMG if c == "I" else TXT for c in layout], dtype=torch.int32)


def _rows(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, HIDDEN).contiguous()


def _images(seeds, valid_counts, pad_to=None):
    """pixel_values [N, P, D] float32 (zero right-padded) + image_position_ids [N, P, 2] with
    (-1, -1) padding -- the wire contract of UserMsg. Deterministic per seed."""
    n = len(seeds)
    p = pad_to or max(valid_counts)
    pixels = torch.zeros(n, p, PATCH, dtype=torch.float32)
    pos = torch.full((n, p, 2), -1, dtype=torch.int64)
    for i, (seed, count) in enumerate(zip(seeds, valid_counts)):
        g = torch.Generator().manual_seed(seed)
        pixels[i, :count] = torch.rand(count, PATCH, generator=g)
        pos[i, :count, 0] = torch.arange(count)
        pos[i, :count, 1] = 0
    return pixels, pos


def _pending(uid, layout, seeds, key=True):
    prompt = _prompt(layout)
    runs = placeholder_runs(prompt, IMG)
    pixels, pos = _images(seeds, [length for _, length in runs])
    n_rows = int((prompt == IMG).sum())
    cache_key = image_cache_key_ids(prompt, IMG, pixels, pos) if key else None
    return PendingReq(uid=uid, input_ids=prompt, sampling_params=SamplingParams(max_tokens=4),
                      mm_embeds=_rows(n_rows), cache_key_ids=cache_key), runs


def _drive(cm, tm, pm, pending: PendingReq, budget: int = CHUNK):
    pm.pending_list = [pending]
    seen = []
    while pm.runnable:
        batch = pm.schedule_next_batch(budget)
        assert batch is not None
        cm.allocate_paged(batch.reqs)
        for r in batch.reqs:
            r.chunk_ids = r.input_ids[r.cached_len : r.device_len].clone()
            r.complete_one()
        seen.append(batch.reqs)
    return seen


def _finish(cm, req: Req, gen_tokens: int = 3):
    """Decode a few tokens (input_ids grows past the prompt) and commit like the scheduler."""
    for t in range(gen_tokens):
        req.append_host(torch.tensor([1000 + t], dtype=torch.int32))
        cm.allocate_paged([req])
        req.complete_one()
    cm.cache_req(req, finished=True)


#                chunk 0     chunk 1     chunk 2     chunk 3
LAYOUT = "tttttttt" + "ttttIIII" + "IIIItttt" + "ttIIIttt"   # image A: 8 slots, image B: 3 slots


# ------------------------------------------------------------------ the key stream itself

def test_key_marks_the_first_placeholders_of_each_image_and_nothing_else():
    prompt = _prompt(LAYOUT)
    runs = placeholder_runs(prompt, IMG)
    assert runs == [(12, 8), (26, 3)]
    pixels, pos = _images([1, 2], [8, 3])
    key = image_cache_key_ids(prompt, IMG, pixels, pos)
    assert key is not None and key.dtype == prompt.dtype and len(key) == len(prompt)
    marked = {12, 13, 26, 27}
    for i in range(len(prompt)):
        if i in marked:
            assert int(key[i]) < 0, i
        else:
            assert int(key[i]) == int(prompt[i]), i
    # two pictures -> two different markers; the real ids are untouched
    assert (int(key[12]), int(key[13])) != (int(key[26]), int(key[27]))
    assert torch.equal(prompt, _prompt(LAYOUT))


def test_same_pixels_same_key_different_pixels_different_key():
    prompt = _prompt("tttIIIItt")
    a1, p1 = _images([1], [4])
    a2, p2 = _images([1], [4])
    b, pb = _images([2], [4])
    ka = image_cache_key_ids(prompt, IMG, a1, p1)
    assert torch.equal(ka, image_cache_key_ids(prompt, IMG, a2, p2))
    kb = image_cache_key_ids(prompt, IMG, b, pb)
    assert not torch.equal(ka, kb)
    assert int(ka[3]) != int(kb[3])  # they diverge AT the first placeholder


def test_right_padding_from_a_neighbour_image_does_not_change_the_key():
    """The padded tray (still the offline shape, and every shape before #890) pads every image
    to the widest in the request; the same picture next to a bigger one must key the same as it
    does alone. ``test_mm_encode_890.py`` carries the packed half of this."""
    pixels_alone, pos_alone = _images([1], [4])
    pixels_padded, pos_padded = _images([1, 2], [4, 9])
    assert image_digest(pixels_alone[0], pos_alone[0]) == image_digest(pixels_padded[0], pos_padded[0])


def test_one_token_run_carries_one_marker():
    prompt = _prompt("ttItt")
    pixels, pos = _images([3], [1])
    key = image_cache_key_ids(prompt, IMG, pixels, pos)
    assert int(key[2]) < 0 and int(key[3]) == TXT
    assert MARKERS_PER_IMAGE == 2


def test_bypass_when_the_key_cannot_be_derived():
    prompt = _prompt("tttIIIItt")
    pixels, pos = _images([1, 2], [4, 4])       # two images, one run
    assert image_cache_key_ids(prompt, IMG, pixels, pos) is None
    one, one_pos = _images([1], [4])
    assert image_cache_key_ids(prompt, None, one, one_pos) is None      # no placeholder id
    assert image_cache_key_ids(prompt, IMG, None, None) is None          # offline: no pixels
    assert image_cache_key_ids(_prompt("ttttt"), IMG, one, one_pos) is None  # no run at all


def test_extend_key_appends_the_real_ids_that_decode_added():
    key = torch.tensor([7, -5, -9, 7], dtype=torch.int32)
    grown = torch.tensor([7, IMG, IMG, 7, 100, 101], dtype=torch.int32)
    out = extend_key(key, grown)
    assert out.tolist() == [7, -5, -9, 7, 100, 101]
    assert torch.equal(extend_key(key, grown[:3]), key[:3])


# ------------------------------------------------------------------ on the cache managers

@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_same_picture_same_prefix_hits_and_different_picture_misses_at_its_first_placeholder(cache_type):
    cm, tm, _dm, pm = _build_managers(cache_type)
    first, runs = _pending(1, LAYOUT, seeds=[1, 2])
    batches = _drive(cm, tm, pm, first)
    reqs = [b[0] for b in batches]
    # the key rides every chunk at the chunk's own length; the token pool saw REAL ids
    for r in reqs:
        assert r.cache_key_ids is not None and len(r.cache_key_ids) == len(r.input_ids)
        assert int((r.chunk_ids == IMG).sum()) == r.mm_embeds.shape[0]
    assert int((torch.cat([r.chunk_ids for r in reqs]) == IMG).sum()) == 11
    final = reqs[-1]
    assert type(final) is Req
    if cache_type == "hybrid_radix":
        # A hybrid match resumes only from a donated GDN snapshot boundary (the forward writes
        # one at every tracked x64 boundary during prefill; this harness runs no forward), so
        # stand one in after image B's run -- the way test_abort_inflight_prefill does.
        final.mamba_last_track_seqlen = 30
    _finish(cm, final)

    run_a_start, run_a_len = runs[0]
    # (1) same prefix, same two pictures -> a prefix hit that covers BOTH images
    again, _ = _pending(2, LAYOUT, seeds=[1, 2])
    hit = cm.match_req(again)
    assert hit.cuda_handle.cached_len >= runs[1][0] + runs[1][1], (cache_type, hit.cuda_handle.cached_len)
    # (2) same prefix, a DIFFERENT first picture -> the match stops before image A's KV
    other, _ = _pending(3, LAYOUT, seeds=[9, 2])
    miss = cm.match_req(other)
    assert miss.cuda_handle.cached_len <= run_a_start, (cache_type, miss.cuda_handle.cached_len)
    # (3) same first picture, a different SECOND one -> reuse through image A, stop before B
    half, _ = _pending(4, LAYOUT, seeds=[1, 8])
    m = cm.match_req(half)
    assert run_a_start + run_a_len <= m.cuda_handle.cached_len <= runs[1][0] or cache_type == "hybrid_radix", (
        cache_type, m.cuda_handle.cached_len)
    assert m.cuda_handle.cached_len <= runs[1][0]
    # (4) a TEXT request that happens to carry literal placeholder ids never reaches an image's KV
    text = PendingReq(uid=5, input_ids=_prompt(LAYOUT), sampling_params=SamplingParams(max_tokens=4))
    t = cm.match_req(text)
    assert t.cuda_handle.cached_len <= run_a_start


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_an_image_request_without_a_key_keeps_the_old_bypass(cache_type):
    """The offline path attaches embeddings with nothing to hash: no match, no insert."""
    cm, tm, _dm, pm = _build_managers(cache_type)
    pending, _ = _pending(1, LAYOUT, seeds=[1, 2], key=False)
    assert pending.cache_key_ids is None
    assert cm.match_req(pending).cuda_handle.cached_len == 0
    final = [b[0] for b in _drive(cm, tm, pm, pending)][-1]
    _finish(cm, final)
    evictable = (cm.prefix_cache.full_evictable_size if cache_type == "hybrid_radix"
                 else cm.prefix_cache.size_info.evictable_size)
    assert evictable == 0
    again, _ = _pending(2, LAYOUT, seeds=[1, 2])
    assert cm.match_req(again).cuda_handle.cached_len == 0
