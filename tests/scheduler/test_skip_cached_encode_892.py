"""Patch 0019 (llm-server #892): do not re-run the vision tower on an image the prefix cache
already holds.

llama.cpp skips the ENCODE for a picture it has already served from KV; FreeToken re-runs the
whole tower on every image in every turn, because ``_attach_mm_embeds`` fires at admission
before ``add_one_req`` and long before the prefix match (``cache.py`` ``match_req``, called from
``prefill.py``'s ``_try_allocate_one``) -- ``scheduler.py:656`` / ``:688`` on the 0021 tree, i.e.
BEFORE this patch moves them. #843 priced the gap at 0.27 s
per image per turn, linear in N.

⛔ "Is this image's KV already held?" is NOT image-keyed -- an image's KV is reusable only as part
of a MATCHING PREFIX -- so the check IS the prefix match, moved to admission and locked there.
These tests walk the request in the order it travels:

1. a match's ``cached_len`` becomes a per-image verdict (``mm_key.cached_leading_images``),
2. the tower is handed only the images that verdict did not cover,
3. a prefill chunk still receives exactly its own rows once the leading rows were never encoded,
4. the match is taken ONCE, at admission, locked, and carried,
5. a reservation that never becomes a request gives its lock back,
6. end to end: a re-sent picture is not re-encoded.
"""
from __future__ import annotations

import torch

import pytest

from freetoken.multimodal import _pack_batch
from freetoken.scheduler.mm_encode import encode_one_at_a_time
from freetoken.scheduler.mm_key import cached_leading_images
from freetoken.scheduler.prefill import slice_mm_embeds

IMG, TXT = 101, 7
D = 8          # stands in for the served 1536 = 3 channels * 2 temporal * 16**2
MERGE = 2      # so one soft token is 4 patches, as on the row


def _ids(*runs: int | str) -> torch.Tensor:
    """``_ids(3, "img4", 2)`` -> 3 text ids, a 4-placeholder run, 2 text ids."""
    out: list[int] = []
    for run in runs:
        if isinstance(run, str):
            out.extend([IMG] * int(run.removeprefix("img")))
        else:
            out.extend([TXT] * run)
    return torch.tensor(out, dtype=torch.int32)


# --------------------------------------------------------------------------- #
# 1. a prefix match becomes a per-image verdict
# --------------------------------------------------------------------------- #
def test_a_miss_leaves_every_image_to_encode():
    ids = _ids(2, "img4", 3, "img4", 2)
    assert cached_leading_images(ids, IMG, 0) == (0, 0)


def test_a_leading_image_wholly_inside_the_matched_prefix_is_already_held():
    ids = _ids(2, "img4", 3, "img4", 2)
    # the match covers the first image's whole run and the text after it
    assert cached_leading_images(ids, IMG, 9) == (1, 4)


def test_the_verdict_lands_exactly_on_the_run_s_last_placeholder():
    ids = _ids(2, "img4", 3, "img4", 2)
    assert cached_leading_images(ids, IMG, 6) == (1, 4)   # [0,6) is 2 text + the 4 placeholders
    assert cached_leading_images(ids, IMG, 5) == (0, 0)   # one placeholder short


def test_an_image_straddling_the_match_boundary_is_never_skipped():
    """⛔ The rows of a half-cached image are still consumed by the chunk that holds its tail, so
    a straddling run must be encoded whole -- and it stops the verdict for everything after it."""
    ids = _ids(2, "img4", 3, "img4", 2)
    assert cached_leading_images(ids, IMG, 11) == (1, 4)  # inside the second run


def test_two_leading_images_are_both_held_and_their_rows_add_up():
    ids = _ids(1, "img4", 1, "img6", 1, "img2", 1)
    assert cached_leading_images(ids, IMG, 13) == (2, 10)


def test_every_image_inside_the_prefix_leaves_nothing_to_encode():
    ids = _ids(1, "img4", 1, "img6", 4)
    assert cached_leading_images(ids, IMG, len(ids)) == (2, 10)


def test_a_text_prompt_holds_no_images():
    assert cached_leading_images(_ids(12), IMG, 6) == (0, 0)


def test_a_model_with_no_placeholder_id_skips_nothing():
    assert cached_leading_images(_ids(2, "img4", 2), None, 8) == (0, 0)


def _packed(patch_counts, seed=0):
    """A packed ``[sum(P), D]`` run plus the grid it came from, the shape 0018 ships."""
    grid = torch.tensor([[1, c // MERGE, MERGE] for c in patch_counts], dtype=torch.int64)
    g = torch.Generator().manual_seed(seed)
    return _pack_batch(torch.rand(sum(patch_counts), D, generator=g), grid, MERGE)


def _tray(pixels, positions, counts):
    """The right-padded ``[N, P_max, *]`` batch the offline ``LLM.generate`` API still hands over."""
    p_max = max(counts)
    tray = pixels.new_zeros((len(counts), p_max, pixels.shape[-1]))
    tray_pos = positions.new_full((len(counts), p_max, positions.shape[-1]), -1)
    offset = 0
    for index, count in enumerate(counts):
        tray[index, :count] = pixels[offset : offset + count]
        tray_pos[index, :count] = positions[offset : offset + count]
        offset += count
    return tray, tray_pos


class _RecordingTower:
    """Stands in for ``model.encode_images``: records every batch it is handed."""

    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, pixels, positions):
        self.calls.append(tuple(pixels.shape))
        assert pixels.shape[:2] == positions.shape[:2]
        soft = pixels.shape[1] // (MERGE * MERGE)
        return pixels.reshape(-1, D)[:soft].clone()


# --------------------------------------------------------------------------- #
# 2. the tower runs only on the images the cache does not hold
# --------------------------------------------------------------------------- #
def test_a_leading_image_the_cache_holds_never_reaches_the_tower():
    """⭐ THE POINT OF THE PATCH. #843 priced the encode at 0.27 s per image per turn, and
    #883/#887 measured that a CACHED image is decoded and encoded exactly like a cold one."""
    counts = [16, 4, 8]
    pixels, positions, out_counts = _packed(counts)
    tower = _RecordingTower()
    encode_one_at_a_time(tower, pixels, positions, "cpu", out_counts, skip_images=1)
    assert tower.calls == [(1, 4, D), (1, 8, D)]


def test_the_rows_returned_are_exactly_the_encoded_images_in_prompt_order():
    """⛔ The rows are consumed POSITIONALLY downstream, so their order and count are the
    contract: what comes back must be the tail of the full-request result, never a re-based
    re-ordering of it."""
    counts = [16, 4, 8]
    pixels, positions, out_counts = _packed(counts, seed=5)
    whole = encode_one_at_a_time(_RecordingTower(), pixels, positions, "cpu", out_counts)
    tail = encode_one_at_a_time(
        _RecordingTower(), pixels, positions, "cpu", out_counts, skip_images=1
    )
    assert torch.equal(tail, whole[16 // (MERGE * MERGE) :])


def test_skipping_every_image_but_the_last_still_returns_one_uncatted_result():
    counts = [16, 4, 8]
    pixels, positions, out_counts = _packed(counts)
    tower = _RecordingTower()
    got = encode_one_at_a_time(tower, pixels, positions, "cpu", out_counts, skip_images=2)
    assert tower.calls == [(1, 8, D)]
    assert got.shape[0] == 8 // (MERGE * MERGE)


def test_skipping_every_image_is_refused_rather_than_answered_with_a_guessed_shape():
    """⛔ There is no honest empty result here: the hidden size is the TOWER's, and it never ran.
    The caller drops ``mm_embeds`` entirely for this case (no placeholder falls in the extend
    region when every image is inside the matched prefix) rather than inventing a ``[0, H]``."""
    counts = [16, 4]
    pixels, positions, out_counts = _packed(counts)
    with pytest.raises(ValueError, match="every image"):
        encode_one_at_a_time(
            _RecordingTower(), pixels, positions, "cpu", out_counts, skip_images=2
        )


def test_a_skip_past_the_end_is_refused_too():
    counts = [16, 4]
    pixels, positions, out_counts = _packed(counts)
    with pytest.raises(ValueError, match="every image"):
        encode_one_at_a_time(
            _RecordingTower(), pixels, positions, "cpu", out_counts, skip_images=3
        )


def test_the_deployed_call_is_unchanged_when_nothing_is_skipped():
    """⚠ Every text row and every cold image turn takes this path; 0019 must not move it."""
    counts = [16, 4, 8]
    pixels, positions, out_counts = _packed(counts, seed=7)
    a, b = _RecordingTower(), _RecordingTower()
    without = encode_one_at_a_time(a, pixels, positions, "cpu", out_counts)
    with_zero = encode_one_at_a_time(b, pixels, positions, "cpu", out_counts, skip_images=0)
    assert a.calls == b.calls == [(1, 16, D), (1, 4, D), (1, 8, D)]
    assert torch.equal(without, with_zero)


def test_the_padded_offline_tray_skips_the_same_images():
    counts = [16, 4, 8]
    pixels, positions, out_counts = _packed(counts, seed=9)
    tray, tray_pos = _tray(pixels, positions, counts)
    tower = _RecordingTower()
    got = encode_one_at_a_time(tower, tray, tray_pos, "cpu", skip_images=1)
    assert tower.calls == [(1, 4, D), (1, 8, D)]
    packed = encode_one_at_a_time(
        _RecordingTower(), pixels, positions, "cpu", out_counts, skip_images=1
    )
    assert torch.equal(got, packed)


def test_a_packed_run_whose_counts_do_not_add_up_is_still_refused_with_a_skip():
    """⛔ The batch still CARRIES every image's patches -- 0019 skips the encode, not the
    shipping -- so the whole-batch validation must stay whole-batch."""
    counts = [16, 4]
    pixels, positions, _ = _packed(counts)
    with pytest.raises(ValueError, match="patch counts total"):
        encode_one_at_a_time(
            _RecordingTower(), pixels, positions, "cpu", [16, 8], skip_images=1
        )


# --------------------------------------------------------------------------- #
# 3. a chunk still gets its own rows when the leading rows were never encoded
# --------------------------------------------------------------------------- #
# 1 text, image A (2 slots), 1 text, image B (2), 1 text, image C (2), 1 text
_THREE = _ids(1, "img2", 1, "img2", 1, "img2", 1)


def _rows(n, base=0):
    """``n`` distinguishable soft-token rows, so a slice can be checked by CONTENT."""
    return torch.arange(base, base + n, dtype=torch.float32).unsqueeze(1)


def test_an_unchunked_turn_gets_every_row_that_was_actually_encoded():
    """Image A is inside the matched prefix, so only B and C were encoded: four rows, and the
    chunk that holds B and C must be handed all four."""
    assert cached_leading_images(_THREE, IMG, 4) == (1, 2)
    got = slice_mm_embeds(
        _THREE, _rows(4), IMG, cached_len=4, chunk_size=6, is_last_chunk=True, skipped_rows=2
    )
    assert torch.equal(got, _rows(4))


def test_two_chunks_partition_the_encoded_rows_exactly_once_each():
    """⛔ The rows are consumed positionally, so a rebase that is off by one image serves a
    picture's soft tokens into another picture's placeholders -- silently."""
    embeds = _rows(4)
    first = slice_mm_embeds(
        _THREE, embeds, IMG, cached_len=4, chunk_size=3, is_last_chunk=False, skipped_rows=2
    )
    second = slice_mm_embeds(
        _THREE, embeds, IMG, cached_len=7, chunk_size=3, is_last_chunk=True, skipped_rows=2
    )
    assert torch.equal(first, embeds[:2])    # image B
    assert torch.equal(second, embeds[2:])   # image C
    assert torch.equal(torch.cat([first, second]), embeds)


def test_a_chunk_with_no_placeholder_of_its_own_returns_an_empty_view_not_none():
    """⚠ The cache manager reads ``mm_embeds is not None`` as "this request carries an image",
    and that must hold for every chunk of the prompt -- the rebase must not change it."""
    got = slice_mm_embeds(
        _THREE, _rows(4), IMG, cached_len=9, chunk_size=1, is_last_chunk=True, skipped_rows=2
    )
    assert got is not None and got.shape[0] == 0


def test_the_last_chunk_still_refuses_rows_no_slot_consumed():
    """The books close on what was ENCODED, not on what the prompt contains."""
    with pytest.raises(AssertionError, match="exceed image-token slots"):
        slice_mm_embeds(
            _THREE, _rows(6), IMG, cached_len=4, chunk_size=6, is_last_chunk=True, skipped_rows=2
        )


def test_a_chunk_reaching_below_the_skipped_rows_is_an_engine_bug():
    """⛔ Every chunk of the request starts at or after the match the skip was decided from, so
    this is unreachable -- and it is asserted rather than argued, because the alternative is a
    negative index quietly wrapping to the end of the tensor."""
    with pytest.raises(AssertionError, match="below the images that were skipped"):
        slice_mm_embeds(
            _THREE, _rows(4), IMG, cached_len=0, chunk_size=10, is_last_chunk=True, skipped_rows=2
        )


def test_nothing_skipped_is_the_deployed_behaviour():
    """⚠ Every text row, every cold turn and every pre-0019 arm takes this path."""
    embeds = _rows(6)
    without = slice_mm_embeds(
        _THREE, embeds, IMG, cached_len=4, chunk_size=6, is_last_chunk=True
    )
    with_zero = slice_mm_embeds(
        _THREE, embeds, IMG, cached_len=4, chunk_size=6, is_last_chunk=True, skipped_rows=0
    )
    assert torch.equal(without, with_zero) and torch.equal(without, embeds[2:])


# --------------------------------------------------------------------------- #
# 4. the match is taken ONCE, at admission, locked, and carried
# --------------------------------------------------------------------------- #
# A CPU rig driven the way the scheduler drives it, after ``test_image_prefix_key.py``.
HIDDEN = 4
PATCH = 6
WIDTH = 128
MAX_RUNNING = 4
IMG_ID = 151655                                    # the served placeholder id

#         turn 1: 24 tokens, image A = 8 slots at [12, 20)
_T1 = "tttttttt" + "ttttIIII" + "IIIItttt"
#         turn 2: turn 1 again, then new text and image B = 3 slots at [28, 31)
_T2 = _T1 + "tttt" + "III" + "tt"


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


def _rig(cache_type: str, num_pages: int = 256):
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
    pm = PrefillManager(cm, tm, DecodeManager(page_size=1), image_token_id=IMG_ID)
    return cm, tm, pm


def _prompt(layout: str) -> torch.Tensor:
    return torch.tensor([IMG_ID if c == "I" else TXT for c in layout], dtype=torch.int32)


def _soft(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, HIDDEN).contiguous()


def _msg(uid: int, layout: str, seeds, key: bool = True):
    """A ``UserMsg`` as the tokenizer worker sends it, with the scheduler's key already derived
    (that derivation moves ABOVE the tower run in bullet 6; here it is simply done for us)."""
    from freetoken.core import SamplingParams
    from freetoken.message.backend import UserMsg
    from freetoken.scheduler.mm_key import image_cache_key_ids, placeholder_runs

    prompt = _prompt(layout)
    runs = placeholder_runs(prompt, IMG_ID)
    counts = [length for _, length in runs]
    p_max = max(counts)
    pixels = torch.zeros(len(counts), p_max, PATCH, dtype=torch.float32)
    pos = torch.full((len(counts), p_max, 2), -1, dtype=torch.int64)
    for i, (seed, count) in enumerate(zip(seeds, counts)):
        g = torch.Generator().manual_seed(seed)
        pixels[i, :count] = torch.rand(count, PATCH, generator=g)
        pos[i, :count, 0] = torch.arange(count)
        pos[i, :count, 1] = 0
    return UserMsg(
        uid=uid, input_ids=prompt, sampling_params=SamplingParams(max_tokens=4),
        pixel_values=pixels, image_position_ids=pos,
        cache_key_ids=image_cache_key_ids(prompt, IMG_ID, pixels, pos) if key else None,
    )


def _evictable(cm, cache_type: str) -> int:
    return (cm.prefix_cache.full_evictable_size if cache_type == "hybrid_radix"
            else cm.prefix_cache.size_info.evictable_size)


def _serve(cm, tm, pm, msg, mm_embeds, cache_type, reservation=None, budget: int = 64):
    """Admit, prefill every chunk, decode three tokens, commit -- what the scheduler does."""
    from freetoken.scheduler.utils import PendingReq

    msg.mm_embeds = mm_embeds
    pm.add_one_req(msg, reservation) if reservation is not None else pm.add_one_req(msg)
    pending = pm.pending_list[-1]
    assert isinstance(pending, PendingReq)
    last, admitted_at = _drain(cm, pm, budget)
    return pending, last, admitted_at


def _drain(cm, pm, budget: int = 64):
    """Prefill every pending chunk and step it once -- the scheduler's loop, minus the forward."""
    last, admitted_at = None, None
    while pm.runnable:
        batch = pm.schedule_next_batch(budget)
        assert batch is not None
        if admitted_at is None:
            # read it BEFORE the forward's complete_one() advances it
            admitted_at = batch.reqs[-1].cached_len
        cm.allocate_paged(batch.reqs)
        for r in batch.reqs:
            r.complete_one()
        last = batch.reqs[-1]
    return last, admitted_at


def _finish(cm, req, cache_type, boundary=None, gen_tokens: int = 3):
    if cache_type == "hybrid_radix":
        # This harness runs no forward, so stand in the GDN boundary a real prefill would have
        # donated -- the way test_image_prefix_key.py and test_abort_inflight_prefill.py do.
        req.mamba_last_track_seqlen = boundary
    for t in range(gen_tokens):
        req.append_host(torch.tensor([1000 + t], dtype=torch.int32))
        cm.allocate_paged([req])
        req.complete_one()
    cm.cache_req(req, finished=True)


def _turn_one(cm, tm, pm, cache_type):
    """Serve and commit turn 1, so turn 2 has a prefix to hit."""
    _, req, _ = _serve(cm, tm, pm, _msg(1, _T1, seeds=[1]), _soft(8), cache_type)
    _finish(cm, req, cache_type, boundary=len(_T1))


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_the_match_is_taken_once_at_admission_and_never_again_at_prefill(cache_type):
    """⛔ Carrying it is a CORRECTNESS requirement, not an optimisation: a second match could
    return a LONGER prefix, and ``slice_mm_embeds`` would then skip rows that WERE encoded --
    the last-chunk assert, i.e. a crash."""
    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)

    msg = _msg(2, _T2, seeds=[1, 5])
    matches: list[int] = []
    real = cm.match_req
    cm.match_req = lambda r: (matches.append(r.uid), real(r))[1]

    reservation = pm.reserve_prefix(msg)
    assert reservation is not None
    assert reservation.cached_len == len(_T1)
    assert (reservation.skip_images, reservation.skipped_rows) == (1, 8)

    pending, req, admitted_at = _serve(cm, tm, pm, msg, _soft(3), cache_type, reservation)
    assert matches == [2], "the prefill pass re-matched the prefix the reservation already held"
    assert req.cache_handle is reservation.handle
    assert admitted_at == len(_T1) and req.mm_skipped_rows == 8
    assert pending.reservation is None, "the admitted request owns the lock now"


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_the_reservation_locks_the_prefix_it_matched(cache_type):
    """⛔ The lock is not optional and cannot be deferred (#891). Nothing bounds the
    admission -> prefill window, and a match that shrinks in it fires the ``slice_mm_embeds``
    assert. Locking is what turns the admission match from a PREDICTION into a fact."""
    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)

    before = _evictable(cm, cache_type)
    reservation = pm.reserve_prefix(_msg(2, _T2, seeds=[1, 5]))
    assert reservation is not None
    assert _evictable(cm, cache_type) == before - reservation.cached_len


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_pass_that_cannot_admit_keeps_the_reservation_and_its_lock(cache_type):
    """A refused pass leaves the request pending and it retries; the reservation must survive
    that, or the retry would sit on an unlocked handle it still intends to trust."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    msg = _msg(2, _T2, seeds=[1, 5])
    reservation = pm.reserve_prefix(msg)
    locked = _evictable(cm, cache_type)

    msg.mm_embeds = _soft(3)
    pending = PendingReq(msg.uid, msg.input_ids, msg.sampling_params, mm_embeds=msg.mm_embeds,
                         cache_key_ids=msg.cache_key_ids, reservation=reservation)
    starved = PrefillAdder(token_budget=64, reserved_size=10**6, cache_manager=cm,
                           table_manager=tm, image_token_id=IMG_ID)
    assert starved.try_add_one(pending) is None
    assert pending.reservation is reservation
    assert _evictable(cm, cache_type) == locked

    # the retry admits on the very same handle, without re-matching
    pm.pending_list = [pending]
    batch = pm.schedule_next_batch(64)
    assert batch is not None and batch.reqs[0].cache_handle is reservation.handle


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_cold_prompt_reserves_nothing_and_locks_nothing(cache_type):
    """⚠ There is no gain to hold a lock for, so none is taken -- decision 2 of the round:
    only a request that will actually skip an encode reserves."""
    cm, _tm, pm = _rig(cache_type)
    before = _evictable(cm, cache_type)
    assert pm.reserve_prefix(_msg(1, _T1, seeds=[1])) is None
    assert _evictable(cm, cache_type) == before


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_request_with_no_derivable_key_never_reserves(cache_type):
    """The offline path attaches embeddings with nothing to hash: it keeps the #791 bypass."""
    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    assert pm.reserve_prefix(_msg(2, _T2, seeds=[1, 5], key=False)) is None


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_text_request_still_matches_at_prefill_exactly_as_before(cache_type):
    """⚠ Decision 2 again, from the other side: the path every text row takes must not move."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, tm, pm = _rig(cache_type)
    ids = torch.full((24,), TXT, dtype=torch.int32)
    first = PendingReq(1, ids, SamplingParams(max_tokens=4))
    pm.pending_list = [first]
    batch = pm.schedule_next_batch(64)
    cm.allocate_paged(batch.reqs)
    batch.reqs[0].complete_one()
    _finish(cm, batch.reqs[0], cache_type, boundary=24)

    again = PendingReq(2, torch.cat([ids, torch.full((6,), TXT, dtype=torch.int32)]),
                       SamplingParams(max_tokens=4))
    assert again.reservation is None
    pm.pending_list = [again]
    batch = pm.schedule_next_batch(64)
    assert batch.reqs[0].cached_len == 24 and batch.reqs[0].mm_skipped_rows == 0


# --------------------------------------------------------------------------- #
# 5. a reservation that never becomes a request gives its lock back
# --------------------------------------------------------------------------- #
# ⛔ A reservation holds a TREE LOCK from admission until the request owns it. Every path on
# which the request dies before admission must hand that lock back, and must hand it back
# EXACTLY ONCE -- a leak and a double release look identical from the outside (the leak pins
# pages nothing will ever read; the double release drops a lock a live request still needs).
# Ownership, stated once and tested here: the reservation holds the lock while the request is
# pending, ``try_add_one`` clears ``pending_req.reservation`` at admission, and from then on the
# ``Req`` owns it on the ordinary lifecycle (``cache_req`` / ``_free_req_resources``).
def _count_unlocks(cm):
    """Record every ``CacheManager.unlock`` -- the one call a leak or a double release moves."""
    seen: list = []
    real = cm.unlock
    cm.unlock = lambda handle: (seen.append(handle), real(handle))[1]
    return seen


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_an_abort_before_admission_gives_the_reservation_s_lock_back(cache_type):
    """The request reserved, then its client went away before any pass admitted it. Nothing
    downstream will ever free that handle -- ``abort_req`` returns None here, because there is
    no ``Req`` yet -- so the release has to happen at the pop."""
    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    free_before = _evictable(cm, cache_type)

    msg = _msg(2, _T2, seeds=[1, 5])
    reservation = pm.reserve_prefix(msg)
    assert reservation is not None
    msg.mm_embeds = _soft(3)
    pm.add_one_req(msg, reservation)
    pending = pm.pending_list[-1]
    assert _evictable(cm, cache_type) == free_before - reservation.cached_len

    unlocks = _count_unlocks(cm)
    assert pm.abort_req(2) is None, "nothing was admitted, so there is no Req to free"
    assert unlocks == [reservation.handle]
    assert _evictable(cm, cache_type) == free_before
    assert pm.pending_list == []
    assert pending.reservation is None, "a popped request must not release the lock again"


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_an_admitted_chunked_request_keeps_its_lock_when_aborted(cache_type):
    """⛔ The one that would be a DOUBLE release. A chunked request stays in ``pending_list``
    after its first chunk is admitted, so ``abort_req`` finds it -- but the ``Req`` owns the
    handle by then and the scheduler frees it (``_free_req_resources`` -> ``cache_req``).
    ``try_add_one`` clearing ``pending_req.reservation`` at admission is what keeps this safe."""
    from freetoken.scheduler.prefill import ChunkedReq

    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    msg = _msg(2, _T2, seeds=[1, 5])
    reservation = pm.reserve_prefix(msg)
    locked = _evictable(cm, cache_type)

    msg.mm_embeds = _soft(3)
    pm.add_one_req(msg, reservation)
    batch = pm.schedule_next_batch(4)          # a budget too small for the 9-token extend
    chunk = batch.reqs[0]
    assert isinstance(chunk, ChunkedReq)
    cm.allocate_paged(batch.reqs)
    chunk.complete_one()
    assert pm.pending_list[0].reservation is None

    unlocks = _count_unlocks(cm)
    assert pm.abort_req(2) is chunk, "the admitted chunk is what the scheduler frees"
    assert unlocks == [], "the Req owns the lock now; abort_req must not release it"
    assert _evictable(cm, cache_type) == locked

    cm.cache_req(chunk, finished=True)         # what _free_req_resources does
    assert unlocks == [reservation.handle], "released exactly once, by the request's own free"


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_text_request_aborts_without_touching_a_lock(cache_type):
    """⚠ Decision 2 of the round again: a request that never reserved keeps today's abort."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, tm, pm = _rig(cache_type)
    ids = torch.full((24,), TXT, dtype=torch.int32)
    pm.pending_list = [PendingReq(7, ids, SamplingParams(max_tokens=4))]
    unlocks = _count_unlocks(cm)
    assert pm.abort_req(7) is None
    assert unlocks == []
    assert pm.pending_list == []


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_releasing_nothing_is_a_no_op(cache_type):
    """The failure paths call it unconditionally (they cannot know whether a reservation was
    taken), so ``None`` has to be free rather than a branch every caller repeats."""
    cm, _tm, pm = _rig(cache_type)
    unlocks = _count_unlocks(cm)
    pm.release_reservation(None)
    assert unlocks == []


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_an_abort_for_an_unknown_uid_releases_nothing(cache_type):
    """A reservation belongs to ITS request: an abort naming another uid must leave it alone."""
    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    msg = _msg(2, _T2, seeds=[1, 5])
    reservation = pm.reserve_prefix(msg)
    msg.mm_embeds = _soft(3)
    pm.add_one_req(msg, reservation)
    locked = _evictable(cm, cache_type)

    unlocks = _count_unlocks(cm)
    assert pm.abort_req(999) is None
    assert unlocks == []
    assert _evictable(cm, cache_type) == locked
    assert pm.pending_list[-1].reservation is reservation


# --------------------------------------------------------------------------- #
# 6. end to end: a re-sent picture is not encoded again
# --------------------------------------------------------------------------- #
# The whole round, walked the way a request travels: the tokenizer worker's UserMsg goes into
# ``Scheduler._process_one_msg``, which derives the cache key, reserves the prefix, runs the
# tower on what is left, and admits. The model is fake; everything else is the served code.
class _CountingTower:
    """``model.encode_images`` -- records each image it is handed, one call per image (0018)."""

    def __init__(self, fail: bool = False):
        self.images: list[int] = []
        self.fail = fail

    def __call__(self, pixels, positions):
        assert pixels.shape[0] == 1, "0018 hands the tower one image at a time"
        if self.fail:
            raise RuntimeError("tower fell over")
        count = int(pixels.shape[1])
        self.images.append(count)
        # rows tagged by the call that produced them, so a mis-sliced row is visible
        base = 100 * len(self.images)
        return (base + torch.arange(count, dtype=torch.float32)).view(count, 1).expand(
            count, HIDDEN
        ).contiguous()


def _scheduler(pm, tower, *, prompt_limit=None):
    """A ``Scheduler`` with nothing real but the prefill manager -- the ``__new__`` + namespace
    rig ``test_cost_accounting_core.py`` and ``test_abort_inflight_prefill.py`` both use."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sch = Scheduler.__new__(Scheduler)
    sch.engine = SimpleNamespace(
        max_seq_len=4096, model=SimpleNamespace(encode_images=tower)
    )
    sch.config = SimpleNamespace(
        image_soft_token_limit=lambda: None,
        multimodal_prompt_limit=lambda: prompt_limit,
    )
    sch.device = "cpu"
    sch.prefill_manager = pm
    sch.sent = []
    sch.send_result = sch.sent.extend
    sch._abort_tombstones = {}
    return sch


def _turn(sch, cm, pm, msg, cache_type, boundary, budget: int = 64):
    """One whole turn: through the scheduler, then prefilled, decoded and committed."""
    from freetoken.scheduler.scheduler import Scheduler

    Scheduler._process_one_msg(sch, msg)
    req, admitted_at = _drain(cm, pm, budget)
    if req is not None:
        _finish(cm, req, cache_type, boundary=boundary)
    return req, admitted_at


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_re_sent_picture_is_not_encoded_again(cache_type):
    """⭐⭐ THE ROUND, END TO END. Turn 2 re-sends turn 1's picture and adds a new one; the tower
    must run on the new one ALONE. #843 priced what that saves at 0.27 s per image per turn."""
    cm, tm, pm = _rig(cache_type)
    tower = _CountingTower()
    sch = _scheduler(pm, tower)

    req1, admitted1 = _turn(sch, cm, pm, _msg(1, _T1, seeds=[1]), cache_type, len(_T1))
    assert tower.images == [8], "a cold picture is encoded"
    assert admitted1 == 0 and req1.mm_skipped_rows == 0

    tower.images.clear()
    msg2 = _msg(2, _T2, seeds=[1, 5])
    req2, admitted2 = _turn(sch, cm, pm, msg2, cache_type, len(_T2))
    assert tower.images == [3], "only the NEW picture reached the tower"
    assert admitted2 == len(_T1), "and the turn was admitted on the prefix that held the old one"
    assert req2.mm_skipped_rows == 8
    assert msg2.mm_embeds.shape == (3, HIDDEN)
    # the rows the batch scattered are the new image's, not the old one's
    assert torch.equal(msg2.mm_embeds[:, 0], torch.tensor([100.0, 101.0, 102.0]))
    cm.check_integrity()


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_turn_whose_every_picture_is_already_held_runs_the_tower_on_none(cache_type):
    """⛔ The k == N case, which ``encode_one_at_a_time`` REFUSES rather than answer with a
    guessed shape -- the hidden size belongs to the tower and the tower never ran. It is sound
    to skip the encode outright here: with every image inside the match, no placeholder falls in
    the extend region, so no chunk asks for a row. ⚠ And ``mm_embeds`` being None must not turn
    the request into the #791 cache BYPASS -- that needs a missing KEY, which this has."""
    cm, tm, pm = _rig(cache_type)
    tower = _CountingTower()
    sch = _scheduler(pm, tower)
    _turn(sch, cm, pm, _msg(1, _T1, seeds=[1]), cache_type, len(_T1))

    tower.images.clear()
    text_only = _T1 + "tttt"
    msg2 = _msg(2, text_only, seeds=[1])
    req2, admitted2 = _turn(sch, cm, pm, msg2, cache_type, len(text_only))
    assert tower.images == [], "every picture was inside the match; the tower had nothing to run"
    assert msg2.mm_embeds is None
    assert admitted2 == len(_T1) and req2 is not None

    # ⚠ it still INSERTED its KV under its marker stream: a third turn matches past turn 2's text
    third = pm.reserve_prefix(_msg(3, text_only + "tt", seeds=[1]))
    assert third is not None and third.cached_len > len(_T1)
    cm.check_integrity()


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_failed_encode_gives_the_reservation_s_lock_back(cache_type):
    """⭐ Moved here from bullet 5: this is the first pass where a reservation EXISTS to leak.
    ⛔ That ``except`` is also #871's TP>1 desync -- the only early return in ``_attach_mm_embeds``
    whose outcome can differ per rank -- so what it leaves behind is worth pinning."""
    from freetoken.message.tokenizer import ErrorReplyMsg

    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    free_before = _evictable(cm, cache_type)
    sch = _scheduler(pm, _CountingTower(fail=True))

    unlocks = _count_unlocks(cm)
    from freetoken.scheduler.scheduler import Scheduler

    Scheduler._process_one_msg(sch, _msg(2, _T2, seeds=[1, 5]))
    assert len(sch.sent) == 1 and isinstance(sch.sent[0], ErrorReplyMsg)
    assert pm.pending_list == [], "a failed encode admits nothing"
    assert len(unlocks) == 1, "and leaves no lock behind"
    assert _evictable(cm, cache_type) == free_before


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_ceiling_refusal_never_took_a_lock_to_give_back(cache_type):
    """Ordering, pinned: the ceiling refuses BEFORE anything reserves, so the refusal has
    nothing to release -- the cheapest way for that path to stay leak-free."""
    from freetoken.message.tokenizer import ErrorReplyMsg
    from freetoken.scheduler.scheduler import Scheduler

    cm, tm, pm = _rig(cache_type)
    _turn_one(cm, tm, pm, cache_type)
    free_before = _evictable(cm, cache_type)
    tower = _CountingTower()
    sch = _scheduler(pm, tower, prompt_limit=4)

    unlocks = _count_unlocks(cm)
    Scheduler._process_one_msg(sch, _msg(2, _T2, seeds=[1, 5]))
    assert len(sch.sent) == 1 and isinstance(sch.sent[0], ErrorReplyMsg)
    assert tower.images == [] and unlocks == []
    assert _evictable(cm, cache_type) == free_before


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_a_text_turn_reaches_neither_the_tower_nor_a_reservation(cache_type):
    """⚠ Decision 2 of the round, at the seam that decides it: a text request must not acquire
    a tree lock across the pending window."""
    from freetoken.core import SamplingParams
    from freetoken.message.backend import UserMsg
    from freetoken.scheduler.scheduler import Scheduler

    cm, tm, pm = _rig(cache_type)
    tower = _CountingTower()
    sch = _scheduler(pm, tower)

    ids = torch.full((24,), TXT, dtype=torch.int32)
    Scheduler._process_one_msg(
        sch, UserMsg(uid=1, input_ids=ids, sampling_params=SamplingParams(max_tokens=4))
    )
    assert tower.images == [] and sch.sent == []
    pending = pm.pending_list[-1]
    assert pending.reservation is None and pending.cache_key_ids is None
    req, admitted_at = _drain(cm, pm)
    assert admitted_at == 0 and req.mm_skipped_rows == 0
