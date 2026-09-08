"""Run the vision tower ONE IMAGE AT A TIME, from the packed batch (llm-server #890, patch 0018).

The scheduler is where a request's pixels cross onto the device. It used to cross in one move::

    encode(msg.pixel_values.to(self.device), msg.image_position_ids.to(self.device))

with ``pixel_values`` the right-padded ``[N, P_max, D]`` tray. Two costs rode on that:

* ⛔ the copy is ``n_images x P_max`` float32 patches, not ``sum(P)``. #871 measured **2.21 GiB
  landing on the card before the tower split anything** -- 5.89 publisher-max images at
  384 MiB each (one patch is 6,144 B). ``FREETOKEN_VIT_GROUP=1`` (patch 0014) cannot bound it,
  because 0014 takes ``pixel_values`` as an *input*: it is downstream of this copy.
* ⛔ on TP>1 that copy is where a vision encode runs out of card, and a CAUGHT encode failure is
  the one early return in ``_attach_mm_embeds`` whose outcome can differ per rank -- the ranks
  desync and NCCL's watchdog takes the process down two minutes later (#871). 0018 removes most
  of the TRIGGER; it does not remove the desync, which is #871's own ticket.

⭐ Per image, the transfer is that image's own patches and nothing else -- at llama.cpp parity
(#886: 4,096 soft tokens per image, 16,384 patches) that is **96 MiB**, flat in ``n_images``,
where the tray was ``n_images x 96 MiB``.

⭐ This is the SAME COMPUTATION as the deployed row already performs. With
``FREETOKEN_VIT_GROUP=1`` live (every deployed vision row since 2026-09-05) the tower already
trims each group to its own longest image and runs the images one at a time
(``models/qwen4_exp/vision.py``, patch 0014) -- so the tensors the tower's blocks see here are
bit-identical to the ones they saw before. What changes is only WHERE the padding is dropped:
before the host->device copy instead of after it. ⚠ Against a row with ``FREETOKEN_VIT_GROUP``
unset the tower used to run the whole padded batch at once, and 0014 already recorded what that
costs -- ~1.5e-08 in float32, matmul-tiling rounding, not a logic change.

⚠ ``FREETOKEN_VIT_GROUP`` is therefore inert on the ONLINE path after 0018 (N is always 1 here).
It stays live for the offline ``LLM.generate`` API and for any caller that hands the tower a
whole batch, and leaving it set on a deployed row is correct and free.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch


def split_patch_counts(
    position_ids: torch.Tensor, patch_counts: Sequence[int] | None
) -> list[int]:
    """Per-image patch counts, for either batch shape.

    ``patch_counts`` is what the packed wire carries beside a 2-D ``[sum(P), 2]`` run
    (``multimodal._pack_batch``). A 3-D ``[N, P_max, 2]`` ``position_ids`` is the historical
    right-padded tray -- the offline ``LLM.generate`` API still documents and accepts one -- and
    its counts are the rows that are not ``(-1, -1)``.
    """
    if patch_counts is not None:
        return [int(count) for count in patch_counts]
    if position_ids.dim() == 3:
        return [int(v) for v in (position_ids[..., 0] >= 0).sum(dim=1).tolist()]
    raise ValueError(
        "a packed [sum(P), 2] position_ids needs its per-image patch counts; only a padded "
        "[N, P, 2] batch can be split without them"
    )


def encode_one_at_a_time(
    encode: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pixel_values: torch.Tensor,
    position_ids: torch.Tensor,
    device: torch.device | str,
    patch_counts: Sequence[int] | None = None,
    skip_images: int = 0,
) -> torch.Tensor:
    """``encode`` each image alone; returns ``[sum(soft), hidden]`` for the images NOT skipped.

    Only one image's pixels are on the device at a time: the previous iteration's slice drops its
    last reference at the top of the next one, so torch's caching allocator can hand the same
    block back instead of growing to hold the whole request.

    ⭐ #892 (patch 0019): ``skip_images`` drops the first *k* images -- the LEADING RUN a prefix
    match already holds (``mm_key.cached_leading_images``). llama.cpp has skipped this encode
    since it began keying chunks by hash; FreeToken re-ran the whole tower on every image in
    every turn, at #843's 0.27 s per image per turn. ⛔ The skipped images' patches are still in
    the batch -- 0019 skips the ENCODE, not the shipping, and the whole-batch validation below
    stays whole-batch for exactly that reason. The rows returned are the TAIL of the full-request
    result, in prompt order, because everything downstream consumes them positionally.
    """
    counts = split_patch_counts(position_ids, patch_counts)
    if not counts:
        raise ValueError("multimodal request carries no images")
    packed = position_ids.dim() == 2
    if packed and sum(counts) != int(position_ids.shape[0]):
        raise ValueError(
            f"patch counts total {sum(counts)} but the packed batch holds "
            f"{int(position_ids.shape[0])} patches"
        )
    if skip_images >= len(counts):
        # ⛔ There is no honest empty result to return: the hidden size is the TOWER's and the
        # tower never ran. A caller whose every image is inside the matched prefix must skip the
        # encode entirely -- which is sound, because then no placeholder falls in the extend
        # region either (scheduler.py, patch 0019).
        raise ValueError(
            f"every image in this request is already held by the prefix cache "
            f"(skip_images={skip_images} of {len(counts)}); skip the encode instead of asking "
            f"for a zero-row result"
        )
    outputs: list[torch.Tensor] = []
    offset = sum(counts[:skip_images])
    for index in range(skip_images, len(counts)):
        count = counts[index]
        if packed:
            pixels = pixel_values[offset : offset + count]
            pos = position_ids[offset : offset + count]
            offset += count
        else:
            pixels = pixel_values[index, :count]
            pos = position_ids[index, :count]
        # unsqueeze(0) is a view: the tower's [1, P, *] contract with no second host copy, and
        # `.to(device)` then moves exactly this image's patches.
        outputs.append(encode(pixels.unsqueeze(0).to(device), pos.unsqueeze(0).to(device)))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


__all__ = ["encode_one_at_a_time", "split_patch_counts"]
