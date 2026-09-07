"""Patch 0018 (llm-server #890): pack the multimodal batch, encode one image at a time.

What these tests hold down, in the order the request travels:

1. ``multimodal._pack_batch`` keeps HF's packed ``[sum(P), D]`` run and never allocates the
   ``n_images x P_max`` tray -- the allocation #883 fitted at ``alpha ~ 1.0`` of the worker's
   floor and #883 rung B6 rode into an 8.81 GiB kill at 51% of the deployed soft-token cap.
2. ``mm_encode.encode_one_at_a_time`` hands the tower ONE image per call, so the host->device
   copy is that image's patches and not the request's -- #871's 2.21 GiB.
3. The prefix-cache key is BYTE-IDENTICAL either side of the change, so no deployed row's cache
   is invalidated by 0018.
4. The padded ``[N, P, 2]`` tray still splits correctly: the offline ``LLM.generate`` API
   documents it and still hands one over.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.message.backend import BaseBackendMsg, UserMsg
from freetoken.core import SamplingParams
from freetoken.multimodal import ImageError, _pack_batch
from freetoken.scheduler.mm_encode import encode_one_at_a_time, split_patch_counts
from freetoken.scheduler.mm_key import image_cache_key_ids, image_digest, placeholder_runs

D = 8          # stands in for the served 1536 = 3 channels * 2 temporal * 16**2
MERGE = 2      # so one soft token is 4 patches, as on the row
IMG, TXT = 101, 7


def _grid(patch_counts):
    """``image_grid_thw`` whose rows multiply out to ``patch_counts`` (t=1, w=MERGE)."""
    return torch.tensor([[1, c // MERGE, MERGE] for c in patch_counts], dtype=torch.int64)


def _packed(patch_counts, seed=0):
    total = sum(patch_counts)
    g = torch.Generator().manual_seed(seed)
    return torch.rand(total, D, generator=g), _grid(patch_counts)


def _tray(pixels, positions, counts):
    """The pre-0018 right-padded batch, rebuilt from a packed run -- the reference shape."""
    p_max = max(counts)
    tray = pixels.new_zeros((len(counts), p_max, pixels.shape[-1]))
    tray_pos = positions.new_full((len(counts), p_max, positions.shape[-1]), -1)
    offset = 0
    for index, count in enumerate(counts):
        tray[index, :count] = pixels[offset : offset + count]
        tray_pos[index, :count] = positions[offset : offset + count]
        offset += count
    return tray, tray_pos


# --------------------------------------------------------------------------- #
# 1. the tray is gone
# --------------------------------------------------------------------------- #
def test_pack_batch_returns_the_packed_run_and_where_it_splits():
    counts = [16, 4, 4]
    pixels, grid = _packed(counts)
    out_pixels, out_pos, out_counts = _pack_batch(pixels, grid, MERGE)
    assert out_counts == counts
    assert out_pixels.shape == (sum(counts), D)          # NOT (3, 16, D)
    assert out_pos.shape == (sum(counts), 2)
    assert out_pixels.dtype is torch.float32 and out_pos.dtype is torch.int64
    assert torch.equal(out_pixels, pixels)               # content, unreordered


def test_the_packed_batch_costs_its_content_where_the_tray_cost_n_times_the_largest():
    """#883 rung B4's shape: one big image beside six small ones.

    This is the whole patch in one assertion -- the tray charged 7 x the biggest image for a
    request carrying 1.43x its content, and #883 rung B6 pushed the same ratio to 25.6x.
    """
    counts = [16] + [4] * 6
    pixels, grid = _packed(counts)
    packed_pixels, _, out_counts = _pack_batch(pixels, grid, MERGE)
    tray_elements = len(counts) * max(counts) * D
    assert packed_pixels.numel() == sum(counts) * D
    assert tray_elements == 7 * 16 * D
    # the ratio exactly: 112 padded patches for 40 patches of content
    assert tray_elements == packed_pixels.numel() * 2.8


def test_pack_batch_still_refuses_a_grid_that_disagrees_with_the_patches():
    pixels, grid = _packed([16, 4])
    with pytest.raises(ImageError, match="patches and"):
        _pack_batch(pixels[:-1], grid, MERGE)


def test_pack_batch_refuses_a_count_that_is_not_whole_merge_blocks():
    """The tower asserts on this; saying it here keeps it a 4xx instead of an engine assertion."""
    pixels = torch.rand(6, D)
    grid = torch.tensor([[1, 3, 2]], dtype=torch.int64)   # 6 patches, merge unit is 4
    with pytest.raises(ImageError, match="spatial-merge"):
        _pack_batch(pixels, grid, MERGE)


# --------------------------------------------------------------------------- #
# 2. one image at a time
# --------------------------------------------------------------------------- #
class _RecordingTower:
    """Stands in for ``model.encode_images``: records every batch it is handed."""

    def __init__(self, merge_unit=MERGE * MERGE):
        self.calls: list[tuple[int, ...]] = []
        self.merge_unit = merge_unit

    def __call__(self, pixels, positions):
        self.calls.append(tuple(pixels.shape))
        assert pixels.shape[:2] == positions.shape[:2]
        soft = pixels.shape[1] // self.merge_unit
        # a deterministic per-image "embedding" so the concatenation is checkable
        return pixels.reshape(-1, D)[: soft].clone()


def test_the_tower_sees_one_image_per_call_and_never_the_whole_request():
    counts = [16, 4, 4]
    pixels, grid = _packed(counts)
    packed_pixels, packed_pos, out_counts = _pack_batch(pixels, grid, MERGE)
    tower = _RecordingTower()
    encode_one_at_a_time(tower, packed_pixels, packed_pos, "cpu", out_counts)
    assert tower.calls == [(1, 16, D), (1, 4, D), (1, 4, D)]
    # the pre-0018 single call was (3, 16, D): 48 patches for a request holding 24.
    assert max(shape[0] * shape[1] for shape in tower.calls) == 16
    assert sum(shape[0] * shape[1] for shape in tower.calls) == sum(counts)


def test_the_concatenated_result_matches_encoding_each_image_by_hand():
    counts = [16, 4, 8]
    pixels, grid = _packed(counts, seed=3)
    packed_pixels, packed_pos, out_counts = _pack_batch(pixels, grid, MERGE)
    got = encode_one_at_a_time(_RecordingTower(), packed_pixels, packed_pos, "cpu", out_counts)
    expected = []
    offset = 0
    for count in counts:
        chunk = packed_pixels[offset : offset + count].unsqueeze(0)
        expected.append(_RecordingTower()(chunk, packed_pos[offset : offset + count].unsqueeze(0)))
        offset += count
    assert torch.equal(got, torch.cat(expected, dim=0))


def test_a_single_image_request_is_not_concatenated():
    pixels, grid = _packed([16])
    packed_pixels, packed_pos, counts = _pack_batch(pixels, grid, MERGE)
    tower = _RecordingTower()
    encode_one_at_a_time(tower, packed_pixels, packed_pos, "cpu", counts)
    assert tower.calls == [(1, 16, D)]


def test_the_padded_offline_tray_still_splits_per_image():
    """``LLM.generate``'s documented ``[N, P, D]`` input, with no counts to go on."""
    counts = [16, 4]
    pixels, grid = _packed(counts)
    packed_pixels, packed_pos, _ = _pack_batch(pixels, grid, MERGE)
    tray, tray_pos = _tray(packed_pixels, packed_pos, counts)
    assert split_patch_counts(tray_pos, None) == counts
    tower = _RecordingTower()
    got = encode_one_at_a_time(tower, tray, tray_pos, "cpu")
    assert tower.calls == [(1, 16, D), (1, 4, D)]       # trimmed, not padded to 16 twice
    packed_out = encode_one_at_a_time(
        _RecordingTower(), packed_pixels, packed_pos, "cpu", counts
    )
    assert torch.equal(got, packed_out)


def test_a_packed_run_without_counts_is_refused_rather_than_guessed():
    pixels, grid = _packed([16, 4])
    packed_pixels, packed_pos, _ = _pack_batch(pixels, grid, MERGE)
    with pytest.raises(ValueError, match="per-image patch counts"):
        encode_one_at_a_time(_RecordingTower(), packed_pixels, packed_pos, "cpu")


def test_counts_that_do_not_add_up_are_refused():
    pixels, grid = _packed([16, 4])
    packed_pixels, packed_pos, _ = _pack_batch(pixels, grid, MERGE)
    with pytest.raises(ValueError, match="patch counts total"):
        encode_one_at_a_time(_RecordingTower(), packed_pixels, packed_pos, "cpu", [16, 8])


# --------------------------------------------------------------------------- #
# 3. the prefix-cache key does not move
# --------------------------------------------------------------------------- #
def _prompt(layout):
    return torch.tensor([IMG if c == "I" else TXT for c in layout], dtype=torch.int32)


def test_the_key_is_byte_identical_across_the_packed_and_padded_shapes():
    """0018 must not invalidate a deployed row's prefix cache."""
    counts = [16, 4]
    layout = "tt" + "I" * (counts[0] // 4) + "t" + "I" * (counts[1] // 4) + "tt"
    prompt = _prompt(layout)
    pixels, grid = _packed(counts, seed=11)
    packed_pixels, packed_pos, out_counts = _pack_batch(pixels, grid, MERGE)
    tray, tray_pos = _tray(packed_pixels, packed_pos, counts)
    packed_key = image_cache_key_ids(prompt, IMG, packed_pixels, packed_pos, out_counts)
    padded_key = image_cache_key_ids(prompt, IMG, tray, tray_pos)
    assert packed_key is not None and torch.equal(packed_key, padded_key)


def test_a_neighbour_no_longer_exists_to_change_a_picture_s_digest():
    """The padded tray's hazard -- hashing a neighbour's padding -- cannot arise in a packed run,
    and the digest of a picture alone still equals its digest in company."""
    alone_pixels, alone_grid = _packed([4], seed=5)
    a_pix, a_pos, a_counts = _pack_batch(alone_pixels, alone_grid, MERGE)
    together = torch.cat([a_pix, torch.rand(16, D, generator=torch.Generator().manual_seed(6))])
    t_pix, t_pos, t_counts = _pack_batch(together, _grid([4, 16]), MERGE)
    assert image_digest(a_pix[:4], a_pos[:4]) == image_digest(t_pix[:4], t_pos[:4])


def test_a_packed_run_without_counts_bypasses_the_cache_instead_of_keying_wrongly():
    prompt = _prompt("ttIIIItt")
    pixels, grid = _packed([16])
    packed_pixels, packed_pos, _ = _pack_batch(pixels, grid, MERGE)
    assert image_cache_key_ids(prompt, IMG, packed_pixels, packed_pos, None) is None


def test_one_image_still_keys_when_its_counts_arrive():
    prompt = _prompt("ttIIIItt")
    pixels, grid = _packed([16])
    packed_pixels, packed_pos, counts = _pack_batch(pixels, grid, MERGE)
    key = image_cache_key_ids(prompt, IMG, packed_pixels, packed_pos, counts)
    assert key is not None and int(key[2]) < 0 and int(key[3]) < 0
    assert len(placeholder_runs(prompt, IMG)) == 1


# --------------------------------------------------------------------------- #
# 4. the wire
# --------------------------------------------------------------------------- #
def test_the_packed_batch_and_its_counts_cross_the_wire():
    counts = [16, 4]
    pixels, grid = _packed(counts)
    packed_pixels, packed_pos, out_counts = _pack_batch(pixels, grid, MERGE)
    msg = UserMsg(
        uid=9,
        input_ids=torch.arange(4, dtype=torch.int32),
        sampling_params=SamplingParams(),
        pixel_values=packed_pixels,
        image_position_ids=packed_pos,
        image_patch_counts=out_counts,
    )
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, UserMsg)
    assert out.image_patch_counts == counts
    assert torch.equal(out.pixel_values, packed_pixels)
    assert torch.equal(out.image_position_ids, packed_pos)
    # what the wire buffer no longer carries: the tray's padding
    assert out.pixel_values.numel() == sum(counts) * D < len(counts) * max(counts) * D


# --------------------------------------------------------------------------- #
# 5. the scheduler drops the pixels at admission (#876's carried sub-question)
# --------------------------------------------------------------------------- #
def test_add_one_req_never_carries_the_pixels_into_the_request():
    """The pending request keeps the embeddings and the key, never the pixels.

    This is why an abort cannot 'fail to free the tray' on the scheduler side: there is nothing
    holding it once admission returns. ⚠ The tokenizer worker's residue is a different thing --
    an allocator floor, not a live reference -- and no abort returns that one either.
    """
    from dataclasses import fields

    from freetoken.scheduler.utils import PendingReq

    carried = {f.name for f in fields(PendingReq)}
    assert "pixel_values" not in carried and "image_position_ids" not in carried
    assert "mm_embeds" in carried and "cache_key_ids" in carried
