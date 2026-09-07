"""Image ingest: wire payload -> the padded batch ``encode_images`` expects.

The one place that knows how a client's image reaches the vision tower. Both protocol
adapters (``server/openai_api.py``, ``server/anthropic_api.py``) decode through
``decode_image_url`` / ``decode_anthropic_source``; the tokenizer worker preprocesses
through ``MultimodalProcessor``. Nothing here imports a wire request type.

Two contracts this module exists to keep:

* A dropped image is never a success. Every failure below raises ``ValueError`` with a
  legible message, which each adapter turns into a 4xx. Silently continuing past an image
  the server cannot serve is what made an operator believe a text-only row had vision.
* The tower takes ONE image at a time, ``[1, P, D]`` + ``[1, P, 2]``, and HF's processor
  emits every image packed into one ``[sum(P), ...]`` run. ``_pack_batch`` keeps the packed
  shape and records where each image starts, so the two meet per image at the device edge
  (``scheduler/mm_encode.py``) instead of in a right-padded ``[N, P_max, D]`` tray here.

  ⭐⭐ #890 (patch 0018) removed that tray. It cost ``n_images x P_max`` float32 patches of
  host RAM -- allocated, then copied again onto the wire, then decoded on the scheduler side
  -- for a request whose real content is ``sum(P)``. #883 measured the retained copy at
  ``alpha ~ 1.0`` of the tray and fitted the tokenizer worker's floor at
  ``405 + 1.02*(n x P_max) + 0.35*Sum`` MiB, and #883 rung B6 (one big image beside six tiny
  ones) drove that worker to 8.81 GiB and KILLED THE ROW on a request at 51% of the deployed
  soft-token cap -- because the cap bounds ``Sum`` and NOTHING bounds ``n_images``. Packing
  makes the cost track the content, which is the only term a soft-token cap can see.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlsplit

import torch

# A generous per-image cap (OpenAI documents 20 MB; Anthropic 5 MB of base64). This bounds the
# decode, not the token cost -- the prompt ceiling is enforced on the tokenized length, after
# the processor has said how many soft tokens the image actually costs.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_DATA_URL_RE = re.compile(r"^data:(?P<media>[\w.+-]+/[\w.+-]+)?(?P<params>;[^,]*)?,", re.IGNORECASE)
_REMOTE_SCHEMES = ("http://", "https://")
_MAX_REDIRECTS = 3


class ImageError(ValueError):
    """A client-classifiable problem with a supplied image (4xx, never a 500)."""


def _size_field(size: Any, field: str) -> int | None:
    """Read one field off a processor's ``size`` dict-or-object, or None if absent.

    HF image processors expose ``size`` as either a plain dict or a ``SizeDict``-like object
    depending on version; this reads either without assuming which.
    """
    if size is None:
        return None
    getter = getattr(size, "get", None)
    value = getter(field) if callable(getter) else getattr(size, field, None)
    return int(value) if isinstance(value, (int, float)) else None


def estimate_soft_tokens(
    width: int,
    height: int,
    *,
    patch_size: int,
    merge_size: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> int:
    """Upper-bound soft-token cost of one image from its pixel dimensions alone.

    Mirrors the resize-then-patchify arithmetic a real Qwen2VL-style processor performs
    (#840 bullet 1: pixel count capped at ``max_pixels``, patches = pixels / patch_size**2,
    soft tokens = patches / merge_size**2) but every division rounds up, and the pixel count
    is also floored at ``min_pixels`` (the real processor upscales a tiny image to meet it,
    which would otherwise make this under-count). Both adjustments push the result up, never
    down, so it always returns >= what the real processor would produce for the same image --
    never less. That direction is the only one that matters: this function exists to gate a
    request before the real, expensive decode runs (``MultimodalProcessor.encode_chat``), and
    an under-count would let an oversized request slip past the gate it exists to be.
    """
    pixels = width * height
    if max_pixels is not None:
        pixels = min(pixels, max_pixels)
    if min_pixels is not None:
        pixels = max(pixels, min_pixels)
    patches = -(-pixels // (patch_size * patch_size))
    return -(-patches // (merge_size * merge_size))


def images_too_large_message(soft_tokens: int, limit: int) -> str:
    """The one wording for a request whose images alone are over the ceiling.

    Deliberately distinct from ``prompt_too_long_message``: that one is about how much
    conversation the client sent, this one is about how big the pictures are, and telling a
    client to "shorten the prompt" when the fix is to send a smaller image sends it in the
    wrong direction. Both the frontend pre-check (``server/generation.py``) and the
    authoritative admission check (``scheduler/scheduler.py``) refuse with this text.
    """
    return (
        f"the images in this request expand to {soft_tokens} soft tokens > {limit} maximum "
        f"(this server's --max-image-soft-tokens cap); send fewer or smaller images"
    )


def prompt_too_long_message(input_len: int, limit: int) -> str:
    """The one wording for an over-ceiling image prompt.

    Both the frontend pre-check (``server/generation.py``) and the authoritative admission
    check (``scheduler/scheduler.py``) refuse the same request for the same reason; a client
    that hits one and then the other must not be told two different stories.
    """
    return (
        f"prompt is too long for an image request: {input_len} tokens > {limit} maximum "
        f"(this server's --max-multimodal-prompt-tokens cap); shorten the prompt or send a "
        f"smaller image"
    )


# --------------------------------------------------------------------------- #
# Decode: what a client put on the wire -> raw encoded image bytes.
# --------------------------------------------------------------------------- #
def decode_image_url(url: Any, *, allow_remote: bool = False) -> bytes:
    """OpenAI ``image_url.url`` -> encoded image bytes.

    Accepts a ``data:`` URL and a bare base64 payload. A remote ``http(s)`` URL is refused
    unless ``allow_remote``: this server binds to loopback behind a proxy, so fetching a
    client-supplied URL would make it an SSRF pivot into whatever else the box serves.
    """
    if not isinstance(url, str) or not url.strip():
        raise ImageError("image_url.url must be a non-empty string")
    url = url.strip()
    if url.lower().startswith(_REMOTE_SCHEMES):
        if not allow_remote:
            raise ImageError(
                "remote image URLs are not fetched by this server (start it with "
                "--allow-remote-images to enable); send a data: URL or base64 instead"
            )
        return _fetch_remote(url)
    match = _DATA_URL_RE.match(url)
    if match is not None:
        params = (match.group("params") or "").lower()
        if "base64" not in params:
            raise ImageError("only base64-encoded data: URLs are supported")
        return _b64(url[match.end():])
    if "://" in url[:16]:
        raise ImageError(f"unsupported image URL scheme in {url[:32]!r}")
    return _b64(url)


def decode_anthropic_source(source: Any, *, allow_remote: bool = False) -> bytes:
    """Anthropic image block ``source`` -> encoded image bytes.

    ``{"type": "base64", "media_type": ..., "data": ...}`` and ``{"type": "url", "url": ...}``
    are the two shapes the Messages API defines; ``"file"`` needs the Files API this server
    does not implement, and is refused rather than dropped.
    """
    if not isinstance(source, dict):
        raise ImageError("image block requires a source object")
    stype = source.get("type")
    if stype == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data.strip():
            raise ImageError("image source.data must be a non-empty base64 string")
        return _b64(data)
    if stype == "url":
        return decode_image_url(source.get("url"), allow_remote=allow_remote)
    if stype == "file":
        raise ImageError("image source type 'file' requires the Files API, which is not served here")
    raise ImageError(f"unsupported image source type {stype!r}; use 'base64' or 'url'")


def _b64(payload: str) -> bytes:
    # Tolerate the whitespace a pretty-printed JSON body can leave in a long base64 field, and
    # missing padding (some clients strip '='); validate=False would silently accept garbage.
    cleaned = "".join(payload.split())
    if not cleaned:
        raise ImageError("image payload is empty")
    cleaned += "=" * (-len(cleaned) % 4)
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError(f"image payload is not valid base64: {exc}") from exc
    if not raw:
        raise ImageError("image payload decoded to zero bytes")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(
            f"image is {len(raw)} bytes, over the {MAX_IMAGE_BYTES}-byte per-image limit"
        )
    return raw


def check_remote_url(url: str) -> None:
    """Refuse a URL that --allow-remote-images must still not fetch.

    Turning the flag on buys "fetch a picture off the public internet", not "make a request
    to anything this host can reach". Two things are checked: the scheme (the default opener
    also speaks ftp/file, which a redirect could otherwise reach), and every address the host
    resolves to -- loopback, RFC1918, link-local (169.254.169.254 is the cloud metadata
    service) and the other reserved ranges are refused.

    This narrows the surface, it does not seal it: the name is resolved here and again by the
    connect, so a DNS entry that changes between the two is not caught. Sealing that needs the
    connection pinned to the address checked, which urllib does not expose.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise ImageError(f"image URL scheme {parts.scheme!r} is not fetched; use http or https")
    host = parts.hostname
    if not host:
        raise ImageError("image URL has no host")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as exc:
        raise ImageError(f"could not resolve image URL host {host!r}: {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            raise ImageError(
                f"image URL host {host!r} resolves to the non-public address {address}; "
                "this server only fetches images from public addresses"
            )


def _fetch_remote(url: str) -> bytes:
    import urllib.request

    class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
        # A redirect is a second client-supplied URL: check it exactly like the first, or the
        # scheme and address rules above are one hop deep.
        max_redirections = _MAX_REDIRECTS

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check_remote_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    check_remote_url(url)
    opener = urllib.request.build_opener(_ValidatingRedirectHandler)
    try:
        with opener.open(url, timeout=10) as response:  # noqa: S310 -- opt-in, scheme-checked
            raw = response.read(MAX_IMAGE_BYTES + 1)
    except ImageError:
        raise
    except Exception as exc:  # noqa: BLE001 -- a bad client URL is a client error
        raise ImageError(f"could not fetch image URL: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(f"fetched image exceeds the {MAX_IMAGE_BYTES}-byte per-image limit")
    if not raw:
        raise ImageError("fetched image is empty")
    return raw


# --------------------------------------------------------------------------- #
# Preprocess: encoded bytes -> the tower's padded batch, and the expanded prompt.
# --------------------------------------------------------------------------- #
@dataclass
class EncodedPrompt:
    """One tokenized prompt plus the vision inputs its placeholders are waiting for."""

    input_ids: torch.Tensor                      # 1-D int32
    # ⭐ #890: PACKED, not padded -- every image's patches back to back, in document order.
    pixel_values: torch.Tensor | None = None     # [sum(P), D] float32
    image_position_ids: torch.Tensor | None = None  # [sum(P), 2] int64, no padding rows
    image_patch_counts: list[int] | None = None  # P per image; where the packed run splits


class MultimodalProcessor:
    """The checkpoint's own HF processor, loaded once per worker.

    ``apply_chat_template(tokenize=True)`` renders **and** expands each template-emitted
    ``<|image_pad|>`` into the right number of placeholders for that image's grid. Doing the
    expansion by hand is the classic off-by-one here, and ``_merge_multimodal`` asserts on it
    rather than answering wrongly -- so we let the processor own the count and only *check* it.
    """

    def __init__(self, model_path: str) -> None:
        from transformers import AutoProcessor

        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError(f"{model_path} has no image processor; this checkpoint is text-only")
        self.merge_size = int(getattr(image_processor, "merge_size", 2))
        self.patch_size = int(getattr(image_processor, "patch_size", 14))
        size = getattr(image_processor, "size", None)
        self.max_pixels = _size_field(size, "longest_edge")
        self.min_pixels = _size_field(size, "shortest_edge")
        self.image_token_id = _image_token_id(model_path, self.processor)

    def open_images(self, blobs: Sequence[bytes]) -> list[Any]:
        from PIL import Image, UnidentifiedImageError

        images = []
        for index, blob in enumerate(blobs):
            try:
                image = Image.open(io.BytesIO(blob))
                image.load()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ImageError(f"image {index} could not be decoded: {exc}") from exc
            images.append(image.convert("RGB"))
        return images

    def estimate_prompt_soft_tokens(self, blobs: Sequence[bytes]) -> int:
        """Upper bound on this batch's total image soft-token cost, from headers alone.

        Reads each image only far enough to learn its pixel dimensions -- ``Image.open``
        parses just the header; the actual decode (``.load()``) never runs, so cost stays
        flat regardless of image size or count. Exists so an oversized multimodal request can
        be refused *before* paying for real decode+patchify (#840: that step is what pushed a
        15-image worst-case request over this box's last few GiB of host RAM and OOM-killed
        the whole machine, not just the request). See ``estimate_soft_tokens`` for why the
        result never falls below what ``encode_chat`` would actually produce.
        """
        from PIL import Image, UnidentifiedImageError

        total = 0
        for index, blob in enumerate(blobs):
            try:
                with Image.open(io.BytesIO(blob)) as image:
                    width, height = image.size
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ImageError(f"image {index} could not be decoded: {exc}") from exc
            total += estimate_soft_tokens(
                width,
                height,
                patch_size=self.patch_size,
                merge_size=self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        return total

    def encode_chat(
        self,
        messages: list[dict],
        images: Sequence[bytes],
        chat_template_kwargs: dict,
    ) -> EncodedPrompt:
        """Render + tokenize a conversation whose image parts are bare ``{"type": "image"}``
        markers, re-attaching ``images`` to them in document order."""
        opened = self.open_images(images)
        attached, used = _attach_images(messages, opened)
        if used != len(opened):
            raise ImageError(
                f"{len(opened)} images supplied but the conversation has {used} image parts"
            )
        encoded = self.processor.apply_chat_template(
            attached,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        )
        input_ids = encoded["input_ids"][0].reshape(-1).to(torch.int32)
        pixel_values, position_ids, patch_counts = _pack_batch(
            encoded["pixel_values"], encoded["image_grid_thw"], self.merge_size
        )
        unit = self.merge_size * self.merge_size
        slots = int((input_ids == self.image_token_id).sum())
        soft = sum(count // unit for count in patch_counts)
        # The scatter in `_merge_multimodal` asserts this equality and an off-by-one is an
        # engine-side assertion, not a wrong answer. Name it here, where it is still a 4xx.
        if slots != soft:
            raise ImageError(
                f"prompt has {slots} image placeholders but the images produce {soft} soft "
                "tokens; the chat template and the image processor disagree"
            )
        return EncodedPrompt(input_ids, pixel_values, position_ids, patch_counts)


def _image_token_id(model_path: str, processor: Any) -> int:
    import json
    import os

    token_id = getattr(processor, "image_token_id", None)
    if isinstance(token_id, int):
        return token_id
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    token_id = config.get("image_token_id")
    if not isinstance(token_id, int):
        raise RuntimeError(f"{model_path} declares no image_token_id")
    return token_id


def _attach_images(messages: list[dict], images: Sequence[Any]) -> tuple[list[dict], int]:
    """Replace each ``{"type": "image"}`` marker with the matching PIL image, in order.

    The wire payload travels beside the conversation rather than inside it so the prompt the
    template renders stays small and JSON-shaped; this is where the two meet again.
    """
    out: list[dict] = []
    used = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                if used >= len(images):
                    used += 1
                    parts.append(part)
                    continue
                parts.append({"type": "image", "image": images[used]})
                used += 1
            else:
                parts.append(part)
        out.append({**message, "content": parts})
    return out, used


def _pack_batch(
    packed: torch.Tensor, grid_thw: torch.Tensor, merge_size: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """HF's packed ``[sum(P), D]`` -> the same packed run, checked, plus where it splits.

    Returns ``(pixel_values, position_ids, patch_counts)``. ``patch_counts[i]`` is image ``i``'s
    patch count, so the run ``pixel_values[sum(counts[:i]) : sum(counts[:i+1])]`` is that image
    and nothing else. ``scheduler/mm_encode.py`` is what splits it, one image at a time, on the
    device edge.

    ⭐⭐ #890 (patch 0018). This function used to build the tower's right-padded
    ``[N, P_max, D]`` batch here, in the tokenizer worker, on the host:

        pixel_values = packed.new_zeros((n_images, p_max, packed.shape[-1]))

    ⛔ That allocation is ``n_images x P_max``, not ``sum(P)``, and the two diverge without
    limit as soon as a request mixes image sizes -- #883 rung B4 (one 2048 sq. + six 1024 sq.)
    paid 4x its own content, and rung B6 (one big image beside six thumbnails) asked for
    ``7 x P_big`` and took the worker to 8.81 GiB, killing the row at 51% of the deployed
    ``--max-image-soft-tokens``. The cap could not see it: the cap bounds ``sum(P)`` and the
    allocation is ``n_images x P_max``, so the two are ANTI-correlated (#883 section 5).
    ⛔ And nothing anywhere bounds ``n_images``.

    ⚠ Keeping the packed run is not free of the tray's other costs by accident -- it removes
    them by construction. The tray was allocated once here, copied a second time into the
    message buffer (``message/utils.py`` ``serialize_type`` -> ``tobytes()``), and copied a
    third time when the scheduler decoded it. All three now carry ``sum(P)``.

    ⚠ What this does NOT remove: HF's own packed tensor, #883's sum-proportional
    ``beta ~ 0.35`` term. It is the content, so no packing fix can reach it.
    """
    from transformers.vision_utils import get_vision_position_ids

    counts = [int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()]
    # The tower merges each image over whole spatial-merge blocks and asserts on a remainder.
    # Checked here, where it is still a 4xx -- and BEFORE the position ids are built, so a grid
    # the position helper would itself reject is named as the image problem it is.
    unit = merge_size * merge_size
    for index, count in enumerate(counts):
        if count % unit:
            raise ImageError(
                f"image {index}: {count} patches is not a multiple of {unit}; the processor "
                "emits whole spatial-merge blocks"
            )
    position_ids = get_vision_position_ids(grid_thw, merge_size)
    total = sum(counts)
    if packed.shape[0] != total or position_ids.shape[0] != total:
        raise ImageError(
            f"processor emitted {packed.shape[0]} patches and {position_ids.shape[0]} position "
            f"ids for a grid totalling {total}"
        )
    return (
        packed.to(torch.float32).contiguous(),
        position_ids.to(torch.int64).contiguous(),
        counts,
    )


__all__ = [
    "EncodedPrompt",
    "ImageError",
    "MAX_IMAGE_BYTES",
    "MultimodalProcessor",
    "check_remote_url",
    "decode_anthropic_source",
    "decode_image_url",
    "prompt_too_long_message",
]
