"""Patch 0020 (llm-server #898): hash the patches in place, never through a second copy.

``image_digest`` fed ``hashlib`` a ``bytes`` object built by ``.tobytes()``. On a CPU tensor
``.numpy()`` is already zero-copy, so that call was a full extra host copy of the patches --
**96 MiB per image at llama.cpp parity (16,384 patches), per turn, per rank**, and #883/#887
measured it is paid on CACHED images too (a turn at 72% cache-served cost within 9% of the same
images sent cold). ``hashlib.update`` takes the buffer protocol directly, so the copy is removable
without touching a byte of the digest.

Two different things are held down here, and they are NOT the same kind of test:

* ⛔ **The bar.** The digest must stay byte-identical or every deployed row's prefix cache is
  invalidated -- the whole point of the image-aware key (patch 0010) is that a returning
  conversation hits. ``test_the_digest_is_byte_identical_to_the_pre_0020_path`` is a GUARD: it
  passes on both sides of the patch, by design, and its job is to keep passing forever.
* ⭐ **The change.** ``test_the_digest_no_longer_copies_the_patches_to_hash_them`` is the one that
  was RED before the patch: ``tracemalloc`` sees the ``bytes`` object, and on a 12 MiB run the old
  path peaks at the run's full size where the new one peaks at approximately nothing.

⛔⛆ **Why the array is handed to ``update`` raw and not as ``memoryview(arr).cast("B")``**: the
cast form raises ``TypeError: memoryview: cannot cast view with zeros in shape or strides`` on a
zero-length array, i.e. on an image with no valid patches -- a case the ``.tobytes()`` path handled
silently and ``test_an_image_with_no_valid_patches_still_digests`` keeps reachable. Handing the
array over raw is byte-identical on every shape AND raises loudly on a non-C-contiguous buffer
instead of hashing the wrong bytes; ``image_digest``'s own ``.contiguous()`` is what keeps that
unreachable, which ``test_a_non_contiguous_pixel_view_hashes_its_c_order_bytes`` asserts.

⚠ **Scope of the saving**: it lands on the CPU/float32/contiguous run the tokenizer worker
actually sends, where ``.detach().to("cpu", torch.float32).contiguous().numpy()`` is a chain of
no-ops. A bf16 or fp16 ``pixels`` still allocates inside ``.to(torch.float32)`` -- that is a real
conversion, not a redundant copy, and removing it is not what this patch is about. The
byte-identity guards cover those dtypes; the no-copy assertion deliberately does not.

⚠ This is host RAM and host time on the tokenizer/scheduler path. It is NOT VRAM and NOT decode,
and it must not be read as a decode result.
"""
from __future__ import annotations

import hashlib
import tracemalloc

import pytest
import torch

from freetoken.scheduler.mm_key import image_digest

D = 1536       # the served patch width: 3 channels * 2 temporal * 16**2


def _reference_digest_pre_0020(pixels: torch.Tensor, position_ids: torch.Tensor) -> bytes:
    """``image_digest`` exactly as it stood before patch 0020, transcribed rather than imported.

    ⛔ Do not replace this with a call into ``mm_key`` -- a guard that imports the thing it guards
    cannot fail. This is the frozen pre-0020 body, ``.tobytes()`` and all."""
    valid = int((position_ids[:, 0] >= 0).sum().item())
    payload = pixels[:valid].detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    h = hashlib.blake2b(digest_size=8)
    h.update(valid.to_bytes(4, "little"))
    h.update(payload)
    return h.digest()


def _positions(total: int, valid: int) -> torch.Tensor:
    """``[P, 2]`` positions: ``valid`` real rows, then the ``(-1, -1)`` the padded tray uses."""
    pos = torch.full((total, 2), -1, dtype=torch.int64)
    pos[:valid, 0] = torch.arange(valid)
    pos[:valid, 1] = 0
    return pos


def _patches(total: int, width: int = 8, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(total, width, generator=g).to(dtype)


# ------------------------------------------------------------------ the bar: the digest cannot move

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "total, valid",
    [
        (16, 16),   # a packed run's slice -- every row valid, the shape 0018 sends
        (16, 9),    # a padded tray's row -- valid prefix, then (-1, -1)
        (16, 1),    # a one-patch picture in a wide tray
        (1, 1),     # the narrowest image there is
    ],
)
def test_the_digest_is_byte_identical_to_the_pre_0020_path(total, valid, dtype):
    """⛔ The bar #890 cleared explicitly and #898 must clear again: the same picture hashes to the
    same 8 bytes either side of the patch, so no deployed row's prefix cache is invalidated."""
    pixels = _patches(total, dtype=dtype, seed=total * 10 + valid)
    pos = _positions(total, valid)
    assert image_digest(pixels, pos) == _reference_digest_pre_0020(pixels, pos)


def test_the_digest_is_byte_identical_at_the_served_patch_width():
    """The parametrised cases run at width 8 to stay cheap; the row's patches are 1536 wide and a
    stride bug would only show on a real row."""
    pixels = _patches(64, width=D, seed=1)
    pos = _positions(64, 40)
    assert image_digest(pixels, pos) == _reference_digest_pre_0020(pixels, pos)


def test_an_image_with_no_valid_patches_still_digests():
    """⛔⛆ The case that rules out ``memoryview(arr).cast("B")``: a zero-length buffer. The
    ``.tobytes()`` path hashed ``b""`` here, so the new path must too -- not raise."""
    pixels = _patches(8)
    pos = _positions(8, 0)
    assert image_digest(pixels, pos) == _reference_digest_pre_0020(pixels, pos)


def test_a_different_picture_still_gets_a_different_digest():
    """Removing a copy must not accidentally hash something shape-only."""
    pos = _positions(16, 16)
    assert image_digest(_patches(16, seed=1), pos) != image_digest(_patches(16, seed=2), pos)


def test_the_valid_count_is_still_hashed_alongside_the_pixels():
    """``valid`` goes into the hash separately, so two runs whose pixel bytes coincide but whose
    valid counts differ cannot collide."""
    pixels = torch.zeros(8, 4, dtype=torch.float32)
    assert image_digest(pixels, _positions(8, 3)) != image_digest(pixels, _positions(8, 5))


def test_a_non_contiguous_pixel_view_hashes_its_c_order_bytes():
    """Handing a non-C-contiguous array to ``hashlib`` raises ``ValueError``. ``image_digest``'s
    own ``.contiguous()`` is what keeps that unreachable -- assert it, so a later 'simplification'
    that drops the call fails here instead of on a row."""
    pixels = _patches(16, width=8, seed=3).T.contiguous().T   # same values, column-major storage
    assert not pixels.is_contiguous()
    pos = _positions(16, 12)
    assert image_digest(pixels, pos) == _reference_digest_pre_0020(pixels, pos)


# ------------------------------------------------------------------ the change: the copy is gone

def _peak_bytes_hashing(digest_fn, pixels: torch.Tensor, pos: torch.Tensor) -> int:
    """Host bytes ``tracemalloc`` sees allocated while ``digest_fn`` takes the digest.

    ⚠ ``tracemalloc`` traces CPython's allocator, not torch's -- so a tensor conversion would be
    invisible here. That is exactly why callers pass an ALREADY CPU/float32/contiguous tensor, the
    shape the tokenizer worker actually sends: the only allocation left to see is the ``bytes``
    object, which is the thing under test.

    ⛔ BOTH the measurement and its vacuity control go through THIS function, on purpose. A control
    that re-inlines the same four lines only proves ITS OWN copy of the instrument is sighted, and
    two copies drift.

    ⛔⛆ ``tracemalloc.start()`` on an interpreter that is ALREADY tracing (``PYTHONTRACEMALLOC``,
    ``-X tracemalloc``, a memory plugin) neither resets the peak nor nests -- the high-water would
    be a stale one from before this call, and a bare ``stop()`` would kill the outer tracer. Hence
    ``reset_peak()``, and hence leaving tracing on if we did not turn it on."""
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        digest_fn(pixels, pos)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        if started_here:
            tracemalloc.stop()


def test_the_digest_no_longer_copies_the_patches_to_hash_them():
    """⭐ The red one. Before patch 0020 this peaked at the run's full size -- 12 MiB here, and
    **96 MiB on a parity-sized image, per image, per turn, per rank, paid on cached images too**."""
    patches = 2048
    pixels = _patches(patches, width=D)
    assert pixels.is_contiguous() and pixels.dtype is torch.float32 and pixels.device.type == "cpu"
    payload = patches * D * 4
    assert payload == 12 * 2 ** 20, payload

    peak = _peak_bytes_hashing(image_digest, pixels, _positions(patches, patches))

    # The old path allocated `payload` exactly. An eighth of it is far below that and far above
    # the few KB of bookkeeping the buffer-protocol path costs -- no threshold tuning either way.
    assert peak < payload // 8, (
        f"digest allocated {peak} host bytes for a {payload}-byte patch run: "
        "the .tobytes() copy patch 0020 removed is back"
    )


def test_the_copy_the_reference_path_makes_is_what_this_test_can_see():
    """⛔ A negative-space assertion is worthless if the instrument is blind. Run the SAME
    measurement over the frozen pre-0020 body and require it to see the whole copy -- if this ever
    fails, ``test_the_digest_no_longer_copies_the_patches_to_hash_them`` is passing vacuously."""
    patches = 2048
    pixels = _patches(patches, width=D)
    pos = _positions(patches, patches)
    payload = patches * D * 4

    peak = _peak_bytes_hashing(_reference_digest_pre_0020, pixels, pos)

    assert peak >= payload, (peak, payload)
