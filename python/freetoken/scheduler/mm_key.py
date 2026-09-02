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
way its patches do). Only the VALID patches are hashed -- ``_pad_batch`` right-pads every image
to the widest one in the request, so hashing the padded row would make a picture's key depend
on its neighbours.

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


def image_digest(pixels: torch.Tensor, position_ids: torch.Tensor) -> bytes:
    """8-byte digest of ONE image's valid patches (``position_ids`` row -1 marks padding)."""
    valid = int((position_ids[:, 0] >= 0).sum().item())
    payload = pixels[:valid].detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    h = hashlib.blake2b(digest_size=8)
    h.update(valid.to_bytes(4, "little"))
    h.update(payload)
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
) -> torch.Tensor | None:
    """The cache key stream for an image request, or ``None`` to keep today's bypass.

    ``None`` (bypass: empty-prefix match, no insert) whenever the key cannot be derived
    soundly -- no placeholder id declared, no pixels (the offline path attaches embeddings with
    nothing to hash), or the number of placeholder runs differs from the number of images.
    A bypass is always safe; only a wrong key is not.
    """
    if image_token_id is None or pixel_values is None or image_position_ids is None:
        return None
    runs = placeholder_runs(input_ids, image_token_id)
    n_images = int(pixel_values.shape[0])
    if n_images == 0 or len(runs) != n_images:
        return None
    key = input_ids.clone()
    for (start, length), pixels, pos in zip(runs, pixel_values, image_position_ids):
        marks = markers_from_digest(image_digest(pixels, pos))
        for k in range(min(MARKERS_PER_IMAGE, length)):
            key[start + k] = marks[k]
    return key


def extend_key(key_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """The key stream for a request whose ``input_ids`` grew past the prompt (decode appends):
    the prompt's markers, then the real ids after them."""
    n = len(key_ids)
    if len(input_ids) <= n:
        return key_ids[: len(input_ids)]
    return torch.cat([key_ids, input_ids[n:]])
