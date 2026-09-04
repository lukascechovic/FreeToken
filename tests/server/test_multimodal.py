"""Image ingest without a GPU, a model, or a network.

Everything between "a client put an image on the wire" and "the tokenizer worker has bytes"
is pure: the decoders, the URL policy, the part walker, the prompt ceiling and the message
wire. These pin the halves that a deployed text-only row shares with the image path -- an
image part that decodes to nothing must never become a 200, and a text request must be
untouched by any of it.
"""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib
from types import SimpleNamespace

import pytest
import torch
from freetoken.core import SamplingParams
from freetoken.message import BaseBackendMsg, UserMsg
from freetoken.multimodal import (
    MAX_IMAGE_BYTES,
    ImageError,
    MultimodalProcessor,
    check_remote_url,
    decode_anthropic_source,
    decode_image_url,
    estimate_soft_tokens,
    prompt_too_long_message,
)
from freetoken.scheduler.config import SchedulerConfig
from freetoken.server.api_models import MessageContent
from freetoken.server.generation import (
    GenSpec,
    prerender_error,
    render_messages,
    render_messages_multimodal,
)

PNG = bytes.fromhex("89504e470d0a1a0a")  # just a header: nothing here opens the image
B64 = base64.b64encode(PNG).decode()


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
def test_data_url_and_bare_base64_decode_to_the_same_bytes():
    assert decode_image_url(f"data:image/png;base64,{B64}") == PNG
    assert decode_image_url(B64) == PNG


def test_base64_survives_whitespace_and_stripped_padding():
    padded = base64.b64encode(b"abcde").decode()
    assert decode_image_url(f"  {padded[:4]}\n{padded[4:].rstrip('=')} ") == b"abcde"


@pytest.mark.parametrize("url", ["", "   ", None, 42])
def test_an_empty_or_non_string_url_is_a_client_error(url):
    with pytest.raises(ImageError):
        decode_image_url(url)


def test_a_non_base64_payload_is_named_as_such():
    with pytest.raises(ImageError, match="not valid base64"):
        decode_image_url("!!!! not base64 !!!!")


def test_a_data_url_must_be_base64():
    with pytest.raises(ImageError, match="only base64"):
        decode_image_url("data:image/png,literal-bytes")


def test_an_oversize_payload_is_refused_before_pillow_sees_it():
    with pytest.raises(ImageError, match="per-image limit"):
        decode_image_url(base64.b64encode(b"\0" * (MAX_IMAGE_BYTES + 1)).decode())


def test_remote_urls_are_refused_by_default_and_name_the_flag():
    with pytest.raises(ImageError, match="--allow-remote-images"):
        decode_image_url("https://example.com/cat.png")


def test_an_unknown_scheme_is_refused_even_with_remote_allowed():
    with pytest.raises(ImageError, match="scheme"):
        decode_image_url("file:///etc/passwd", allow_remote=True)


def test_anthropic_sources():
    assert decode_anthropic_source({"type": "base64", "data": B64}) == PNG
    with pytest.raises(ImageError, match="Files API"):
        decode_anthropic_source({"type": "file", "file_id": "f_1"})
    with pytest.raises(ImageError, match="unsupported image source type"):
        decode_anthropic_source({"type": "smoke-signal"})
    with pytest.raises(ImageError, match="source object"):
        decode_anthropic_source("data:image/png;base64,xx")
    with pytest.raises(ImageError, match="--allow-remote-images"):
        decode_anthropic_source({"type": "url", "url": "https://example.com/cat.png"})


# --------------------------------------------------------------------------- #
# What --allow-remote-images still refuses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/x.png",       # the server's own loopback
        "http://[::1]/x.png",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/x.png",
        "http://192.168.1.10/x.png",
    ],
)
def test_a_non_public_address_is_refused_even_when_remote_is_allowed(url):
    with pytest.raises(ImageError, match="non-public address"):
        check_remote_url(url)


@pytest.mark.parametrize("url", ["ftp://example.com/x.png", "file:///etc/passwd", "gopher://x/1"])
def test_only_http_and_https_are_fetchable(url):
    with pytest.raises(ImageError, match="scheme"):
        check_remote_url(url)


def test_a_public_literal_address_passes_the_policy():
    check_remote_url("https://93.184.216.34/cat.png")  # no request is made, only the policy


# --------------------------------------------------------------------------- #
# The part walker: markers out, bytes alongside, text untouched
# --------------------------------------------------------------------------- #
def test_a_text_only_conversation_renders_exactly_as_before():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    ]
    rendered, images = render_messages_multimodal(messages)
    assert images == []
    assert rendered[1]["content"] == "ab"          # collapsed to a string, not a part list
    assert rendered == render_messages(messages)   # the text-only entry point agrees


def test_an_image_part_becomes_a_marker_and_its_bytes_come_back_in_order():
    rendered, images = render_messages_multimodal(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}},
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "data": B64}},
                ],
            }
        ]
    )
    assert images == [PNG, PNG]
    assert rendered[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "what is this?"},
        {"type": "image"},
    ]


def test_a_surface_that_does_not_accept_images_refuses_one_rather_than_dropping_it():
    with pytest.raises(ValueError, match="Unsupported content part type"):
        render_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": B64}}]}]
        )


def test_an_unknown_part_type_is_still_refused_on_the_image_surface():
    with pytest.raises(ValueError, match="Unsupported content part type"):
        render_messages_multimodal(
            [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "x"}}]}]
        )


def test_an_undecodable_image_never_renders_a_prompt_without_it():
    with pytest.raises(ImageError):
        render_messages_multimodal(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "%%%%"}}]}]
        )


# --------------------------------------------------------------------------- #
# The wire shape a client may send
# --------------------------------------------------------------------------- #
def test_image_url_accepts_both_the_object_and_the_bare_string_spelling():
    # `image_url` was `Any` before image ingest existed, so the bare-string spelling reached
    # the server (and was ignored). Typing the field must not turn it into a 422.
    part = MessageContent(type="image_url", image_url={"url": B64, "detail": "high"})
    assert part.image_url.url == B64 and part.image_url.detail == "high"
    assert MessageContent(type="image_url", image_url=B64).image_url.url == B64


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #
def _config(**kwargs) -> SchedulerConfig:
    from freetoken.distributed import DistributedInfo

    return SchedulerConfig(
        model_path="/unused", tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16,
        **kwargs,
    )


def test_the_image_prompt_ceiling_is_the_operator_cap_or_nothing():
    """Policy only: the engine chunks a multimodal prefill, so the prefill budget no longer
    bounds an image prompt. Without a cap there is no image-specific ceiling at all."""
    config = _config(max_extend_tokens=8192)
    assert config.multimodal_prompt_limit() is None
    capped = _config(max_extend_tokens=8192, max_multimodal_prompt_tokens=20480)
    assert capped.multimodal_prompt_limit() == 20480          # may exceed the prefill chunk
    assert _config(max_extend_tokens=8192, max_multimodal_prompt_tokens=2048).multimodal_prompt_limit() == 2048


def test_both_refusals_tell_the_client_the_same_story():
    message = prompt_too_long_message(9000, 8192)
    assert "9000" in message and "8192" in message and "--max-multimodal-prompt-tokens" in message
    assert "prefill chunk" not in message and "--max-extend-tokens" not in message


# --------------------------------------------------------------------------- #
# #840: the pre-decode estimate -- must reject an oversized request WITHOUT ever
# opening/patchifying the images (that decode is itself what OOM-killed the host).
# --------------------------------------------------------------------------- #
def test_estimate_soft_tokens_matches_the_bullet1_worst_case_derivation():
    # docs/research/freetoken-cache-sizing-worst-case-840/README.md §1: a 4096x4096 image
    # against this checkpoint's real preprocessor_config.json costs exactly 16,384 soft tokens.
    assert (
        estimate_soft_tokens(
            4096, 4096, patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16_777_216
        )
        == 16384
    )


def test_estimate_soft_tokens_caps_at_max_pixels_instead_of_growing_unbounded():
    small = estimate_soft_tokens(4096, 4096, patch_size=16, merge_size=2, max_pixels=16_777_216)
    huge = estimate_soft_tokens(20000, 20000, patch_size=16, merge_size=2, max_pixels=16_777_216)
    assert huge == small  # the real processor resizes down to fit; a bigger source image is free


def test_estimate_soft_tokens_floors_at_min_pixels_instead_of_undercounting_a_tiny_image():
    # The real processor upscales a tiny image to meet min_pixels; without the floor here this
    # would report a lower cost than the image actually has once the real processor is done.
    assert estimate_soft_tokens(1, 1, patch_size=16, merge_size=2, min_pixels=65536) == (
        estimate_soft_tokens(256, 256, patch_size=16, merge_size=2)  # 256**2 == 65536
    )


def test_estimate_soft_tokens_rounds_up_never_down():
    # 17x17 px at patch_size=16 doesn't divide evenly -- ceiling division must still count it as
    # (at least) one patch. A floor-dividing version of this function would silently return 0 and
    # let an image's cost vanish from the sum, defeating the whole gate.
    assert estimate_soft_tokens(17, 17, patch_size=16, merge_size=2) > 0


def test_estimate_soft_tokens_with_no_configured_bounds_uses_raw_pixels():
    assert estimate_soft_tokens(32, 32, patch_size=16, merge_size=2) == 1  # 1024px -> 4 patches -> 1


def _png(width: int, height: int) -> bytes:
    """A minimal but fully valid 8-bit RGB PNG of the given size -- real enough for
    ``Image.open`` to parse a header from, with no dependency on Pillow to build it."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    idat = zlib.compress(raw)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_estimate_prompt_soft_tokens_sums_a_batch_from_headers_alone():
    # A bare SimpleNamespace stands in for MultimodalProcessor -- proves this method needs
    # nothing but the four config numbers it reads in __init__, not a loaded HF checkpoint.
    processor = SimpleNamespace(patch_size=2, merge_size=2, min_pixels=None, max_pixels=None)
    blob = _png(8, 8)  # 64px -> patches=ceil(64/4)=16 -> soft=ceil(16/4)=4, per image
    total = MultimodalProcessor.estimate_prompt_soft_tokens(processor, [blob, blob])
    assert total == 8


def test_estimate_prompt_soft_tokens_names_an_undecodable_image():
    processor = SimpleNamespace(patch_size=2, merge_size=2, min_pixels=None, max_pixels=None)
    with pytest.raises(ImageError, match="image 1 could not be decoded"):
        MultimodalProcessor.estimate_prompt_soft_tokens(processor, [_png(8, 8), b"not an image"])


class _FakeMultimodalManager:
    """Stands in for ``TokenizeManager`` in ``prerender_error`` -- ``encode_calls`` is the
    load-bearing assertion: it must stay 0 whenever the cheap estimate alone is enough to
    refuse, because #840's whole point is that the real ``encode`` (which decodes and
    patchifies every image) must never run for a request already known to be oversized."""

    def __init__(self, *, estimate: int, encoded_len: int) -> None:
        self._processor = SimpleNamespace(estimate_prompt_soft_tokens=lambda blobs: estimate)
        self._encoded_len = encoded_len
        self.encode_calls = 0

    def multimodal_processor(self):
        return self._processor

    def encode(self, msg):
        self.encode_calls += 1
        return SimpleNamespace(input_ids=torch.zeros(self._encoded_len))


def _mm_state(manager: _FakeMultimodalManager, *, limit: int) -> SimpleNamespace:
    return SimpleNamespace(
        frontend_tokenizer=lambda: manager,
        config=SimpleNamespace(multimodal_prompt_limit=lambda: limit),
    )


def _image_spec() -> GenSpec:
    return GenSpec(
        messages=[{"role": "user", "content": [{"type": "image"}]}],
        sampling_params=SamplingParams(),
        images=[PNG],
    )


def test_prerender_error_rejects_before_the_real_decode_when_the_estimate_is_already_over():
    manager = _FakeMultimodalManager(estimate=50_000, encoded_len=1)  # #840's worst-case shape
    err = asyncio.run(prerender_error(_image_spec(), _mm_state(manager, limit=8192)))
    assert err is not None and "50000" in str(err)
    assert manager.encode_calls == 0  # the expensive step this ticket exists to prevent


def test_prerender_error_still_runs_the_real_check_when_the_estimate_is_within_budget():
    # The estimate alone must not be the final word -- an accepted request still gets the
    # exact, authoritative check against the real tokenized length (unchanged behavior).
    manager = _FakeMultimodalManager(estimate=100, encoded_len=9000)
    err = asyncio.run(prerender_error(_image_spec(), _mm_state(manager, limit=8192)))
    assert manager.encode_calls == 1
    assert err is not None and "9000" in str(err)


def test_prerender_error_admits_a_request_under_both_checks():
    manager = _FakeMultimodalManager(estimate=100, encoded_len=100)
    err = asyncio.run(prerender_error(_image_spec(), _mm_state(manager, limit=8192)))
    assert err is None
    assert manager.encode_calls == 1


# --------------------------------------------------------------------------- #
# The message wire: pixels cross it, text is unchanged
# --------------------------------------------------------------------------- #
def _roundtrip(msg: UserMsg) -> UserMsg:
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, UserMsg)
    return out


def test_a_text_user_msg_crosses_the_wire_unchanged():
    ids = torch.arange(7, dtype=torch.int32)
    out = _roundtrip(UserMsg(uid=1, input_ids=ids, sampling_params=SamplingParams()))
    assert torch.equal(out.input_ids, ids)
    assert out.pixel_values is None and out.image_position_ids is None


def test_the_padded_batch_keeps_its_shape_across_the_wire():
    pixels = torch.randn(3, 5, 8, dtype=torch.float32)
    positions = torch.randint(-1, 4, (3, 5, 2), dtype=torch.int64)
    out = _roundtrip(
        UserMsg(
            uid=2,
            input_ids=torch.arange(4, dtype=torch.int32),
            sampling_params=SamplingParams(),
            pixel_values=pixels,
            image_position_ids=positions,
        )
    )
    assert out.pixel_values.shape == (3, 5, 8) and out.image_position_ids.shape == (3, 5, 2)
    assert torch.equal(out.pixel_values, pixels)
    assert torch.equal(out.image_position_ids, positions)


def test_a_non_contiguous_tensor_survives_the_wire():
    pixels = torch.randn(4, 6, 8).transpose(0, 1)  # a view, not contiguous
    out = _roundtrip(
        UserMsg(
            uid=3,
            input_ids=torch.arange(2, dtype=torch.int32),
            sampling_params=SamplingParams(),
            pixel_values=pixels,
        )
    )
    assert torch.equal(out.pixel_values, pixels)


def test_a_message_written_before_shape_existed_still_decodes():
    from freetoken.message.utils import deserialize_type, serialize_type

    payload = serialize_type(torch.arange(5, dtype=torch.int32))
    payload.pop("shape")  # the pre-image-ingest encoding
    assert torch.equal(deserialize_type({}, payload), torch.arange(5, dtype=torch.int32))
