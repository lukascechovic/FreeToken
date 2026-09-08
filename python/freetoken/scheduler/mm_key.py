"""Image-aware prefix-cache keys (llm-server #791, map #790).

The prefix cache keys on token ids. Every picture is tokenised into the SAME run of
``image_token_id`` placeholders, so two prompts with different pictures behind one text prefix
have identical ids -- a plain prefix match would hand the second picture the first one's KV.
Until now the scheduler therefore matched an image request against the EMPTY prefix and never
inserted it: correct, and zero reuse -- once a picture is in a long session every later turn
re-prefilled the whole conversation (#789 gate 2: 55 s wall at 42k tokens).

This module builds the request's *cache key stream*: ``input_ids`` with the first one or two
placeholders of every image's run replaced by ids derived from a hash of that image's pixels.
The token pool and the model still see the real ids (the scatter in ``slice_mm_embeds`` and
the model's ``masked_scatter`` count real placeholders); only the radix trees see the key stream.
Two different pictures diverge at their first placeholder; the same picture, same prefix, hits.

Why hash the PIXELS here and not the bytes in the tokenizer worker: every TP rank receives the
same ``UserMsg`` and must derive the same key without another wire field, and the preprocessed
patches are what the tower actually sees (a re-encoded JPEG of the same picture keys the same
way its patches do). Only that image's OWN patches are hashed, never a neighbour's -- otherwise
a picture's key would depend on what it was sent beside.

⭐ #890 (patch 0018) changed the batch this reads from a right-padded ``[N, P_max, D]`` tray to a
packed ``[sum(P), D]`` run plus per-image patch counts. ⛔ The DIGEST IS UNCHANGED, deliberately:
the padded path already hashed exactly ``pixels[:valid]``, so the same picture hashes to the same
8 bytes either side of 0018 and no deployed row's prefix cache is invalidated by the patch.
⭐ #898 (patch 0020) holds that same digest while dropping the redundant host copy it was built
through; see ``image_digest``.

Marker ids are NEGATIVE so they can never collide with a vocabulary id; they stay inside int32
because ``input_ids`` is int32 on the wire and the trees key on ``tuple(ids.tolist())``. A run
of >= 2 placeholders carries 62 bits of the digest (two markers), a 1-token run 31 bits.
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import torch

_MARK_BITS = 31
_MARK_MASK = (1 << _MARK_BITS) - 1
MARKERS_PER_IMAGE = 2


def placeholder_runs(input_ids: torch.Tensor, image_token_id: int) -> List[Tuple[int, int]]:
    """``[(start, length), ...]`` of every maximal run of ``image_token_id`` in ``input_ids``."""
    is_img = input_ids == image_token_id
    if not bool(is_img.any()):
        return []
    prev = torch.cat([is_img.new_zeros(1), is_img[:-1]])
    nxt = torch.cat([is_img[1:], is_img.new_zeros(1)])
    starts = torch.nonzero(is_img & ~prev).flatten().tolist()
    ends = torch.nonzero(is_img & ~nxt).flatten().tolist()
    return [(s, e - s + 1) for s, e in zip(starts, ends)]


def cached_leading_images(
    input_ids: torch.Tensor, image_token_id: int | None, cached_len: int
) -> Tuple[int, int]:
    """``(n_images, n_rows)`` that a matched prefix ``[0, cached_len)`` ALREADY HOLDS.

    ⭐ #892 (patch 0019). "Is this image's KV already held?" is not an image-keyed question --
    an image's KV is reusable only as part of a *matching prefix* -- so the answer is read off
    the prefix match, and because the match is a prefix the images it covers are exactly a
    LEADING RUN. That makes the verdict per image and the skip a leading slice of the request's
    images, which is what ``encode_one_at_a_time`` and ``slice_mm_embeds`` can both act on.

    ⛔ A run that STRADDLES ``cached_len`` is NOT held: the chunk carrying its tail still
    consumes that image's rows, so it must be encoded whole -- and it stops the verdict for
    every image after it, which the prefix property makes free rather than conservative.

    ``n_rows`` counts PLACEHOLDERS, which is one row of ``mm_embeds`` each: the model scatters
    positionally, one soft token per placeholder (``models/qwen4_exp/model.py:119-125``).
    """
    if image_token_id is None or cached_len <= 0:
        return 0, 0
    n_images = n_rows = 0
    for start, length in placeholder_runs(input_ids, image_token_id):
        if start + length > cached_len:
            break
        n_images += 1
        n_rows += length
    return n_images, n_rows


def image_digest(pixels: torch.Tensor, position_ids: torch.Tensor) -> bytes:
    """8-byte digest of ONE image's valid patches (``position_ids`` row -1 marks padding).

    Takes one image: a packed run's slice (every row valid) or a padded tray's row (valid
    prefix, then ``(-1, -1)``). Both reduce to the same ``pixels[:valid]`` bytes.

    ⭐ #898 (patch 0020): the patches are hashed IN PLACE. ``.numpy()`` is already zero-copy on
    a CPU tensor, so the ``.tobytes()`` this used to call was a full extra host copy of the
    patches -- 96 MiB per image at llama.cpp parity (16,384 patches), per turn, per rank, and
    #883/#887 measured that it is paid on CACHED images too. ⛔ THE DIGEST IS UNCHANGED,
    deliberately: ``hashlib.update`` reads the array through the buffer protocol and sees exactly
    the bytes ``.tobytes()`` built, so the same picture hashes to the same 8 bytes either side of
    0020 and no deployed row's prefix cache is invalidated
    (``tests/scheduler/test_mm_key_digest_898.py``).

    ⛔⛆ Hand the array to ``update`` RAW, never as ``memoryview(arr).cast("B")``: the cast raises
    ``TypeError: cannot cast view with zeros in shape or strides`` on a zero-length buffer -- an
    image with no valid patches -- which the ``.tobytes()`` path digested silently. Raw is also
    fail-LOUD on a non-C-contiguous buffer (``ValueError``) instead of hashing the wrong bytes,
    which is what makes the ``.contiguous()`` below load-bearing rather than decoration.
    """
    valid = int((position_ids[:, 0] >= 0).sum().item())
    patches = pixels[:valid].detach().to("cpu", torch.float32).contiguous().numpy()
    h = hashlib.blake2b(digest_size=8)
    h.update(valid.to_bytes(4, "little"))
    h.update(patches)
    return h.digest()


def markers_from_digest(digest: bytes) -> Tuple[int, int]:
    """Two negative int32 marker ids from an 8-byte digest: bits [0,31) and [31,62)."""
    h = int.from_bytes(digest, "little")
    return -(h & _MARK_MASK) - 1, -((h >> _MARK_BITS) & _MARK_MASK) - 1


def image_cache_key_ids(
    input_ids: torch.Tensor,
    image_token_id: int | None,
    pixel_values: torch.Tensor | None,
    image_position_ids: torch.Tensor | None,
    image_patch_counts: List[int] | None = None,
) -> torch.Tensor | None:
    """The cache key stream for an image request, or ``None`` to keep today's bypass.

    ``None`` (bypass: empty-prefix match, no insert) whenever the key cannot be derived
    soundly -- no placeholder id declared, no pixels (the offline path attaches embeddings with
    nothing to hash), or the number of placeholder runs differs from the number of images.
    A bypass is always safe; only a wrong key is not.
    """
    if image_token_id is None or pixel_values is None or image_position_ids is None:
        return None
    try:
        images = _per_image(pixel_values, image_position_ids, image_patch_counts)
    except ValueError:
        # A packed run whose counts did not arrive cannot be split soundly. Bypass, never guess:
        # a bypass costs reuse, a wrong key serves another picture's KV.
        return None
    runs = placeholder_runs(input_ids, image_token_id)
    n_images = len(images)
    if n_images == 0 or len(runs) != n_images:
        return None
    key = input_ids.clone()
    for (start, length), (pixels, pos) in zip(runs, images):
        marks = markers_from_digest(image_digest(pixels, pos))
        for k in range(min(MARKERS_PER_IMAGE, length)):
            key[start + k] = marks[k]
    return key


def _per_image(
    pixel_values: torch.Tensor,
    image_position_ids: torch.Tensor,
    image_patch_counts: List[int] | None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """``[(pixels_i, positions_i), ...]`` from either batch shape -- packed run or padded tray.

    Slicing a packed run is a view, so this adds no copy to the admission path -- and since #898
    (patch 0020) the digest does not materialise the bytes either: over the CPU float32 run the
    tokenizer worker sends, its ``.to("cpu", float32).contiguous().numpy()`` is a chain of no-ops
    over this same storage, and ``hashlib`` reads that storage directly.
    """
    from freetoken.scheduler.mm_encode import split_patch_counts

    counts = split_patch_counts(image_position_ids, image_patch_counts)
    if image_position_ids.dim() == 3:
        return [(pixel_values[i], image_position_ids[i]) for i in range(len(counts))]
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    offset = 0
    for count in counts:
        out.append(
            (pixel_values[offset : offset + count], image_position_ids[offset : offset + count])
        )
        offset += count
    return out


def extend_key(key_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """The key stream for a request whose ``input_ids`` grew past the prompt (decode appends):
    the prompt's markers, then the real ids after them."""
    n = len(key_ids)
    if len(input_ids) <= n:
        return key_ids[: len(input_ids)]
    return torch.cat([key_ids, input_ids[n:]])
