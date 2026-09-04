"""``FREETOKEN_VIT_GROUP`` -- encoding a request's images in bounded groups.

The tower used to run on every image of a request at once, right-padded to a common ``P``, so its
peak memory is linear in the image count.  Grouping bounds that.  It is only sound because NOTHING
IN THIS TOWER CROSSES IMAGES -- attention masks to each image's own patches, the position embedding
is interpolated per image, and the merger already loops per image.  This asserts that against the
real module rather than a reading of it.

⛔⛔ IT IS *NOT* BIT-IDENTICAL when it splits, and an earlier draft of #841 claimed it was.  A
   different batch shape changes matmul tiling and SDPA's reduction order, so results move at float
   rounding (~1e-08 in float32 on this fixture).  "Equivalent" here means NUMERICALLY equivalent.

⭐ THE DISCRIMINATOR IS PRECISION, NOT SIZE.  If the gap is rounding it must collapse in float64;
   a real logic error -- a dropped image, a mis-sliced group, a live padding row -- would not.
   Both precisions run and the ratio is asserted, so the claim is measured, not asserted.

⚠ The unset path must stay on the historical unsplit branch: that is what makes the change inert
  until it is switched on.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen4_exp.config import VisionConfig
from freetoken.models.qwen4_exp.vision import Qwen4ExpVisionModel

ENV = "FREETOKEN_VIT_GROUP"
# Ragged, every count a multiple of spatial_merge_size**2 = 4.
COUNTS = [16, 8, 24, 8]


def _config() -> VisionConfig:
    """Small but structurally real: merge**2 divides every patch count, head_dim stays even."""
    return VisionConfig(
        hidden_size=32, depth=2, num_heads=4, intermediate_size=64,
        patch_size=14, temporal_patch_size=2, in_channels=3,
        spatial_merge_size=2, num_position_embeddings=64,
        out_hidden_size=48, hidden_act="gelu_pytorch_tanh", rope_theta=10000.0,
    )


def _batch(config: VisionConfig, counts, dtype: torch.dtype, seed: int = 7):
    """Ragged images right-padded to a common ``P`` -- exactly the caller's contract."""
    generator = torch.Generator().manual_seed(seed)
    width = max(counts)
    pixel_values = torch.randn(
        len(counts), width, config.patch_input_dim, generator=generator, dtype=dtype
    )
    position_ids = torch.full((len(counts), width, 2), -1, dtype=torch.int64)
    for row, count in enumerate(counts):
        grid = torch.stack(
            torch.meshgrid(torch.arange(count // 2), torch.arange(2), indexing="ij"), dim=-1
        )
        position_ids[row, :count] = grid.reshape(-1, 2)
        pixel_values[row, count:] = 0.0  # padding must never be read; a leak would show
    return pixel_values, position_ids


def _model(config: VisionConfig, dtype: torch.dtype, seed: int = 1234) -> Qwen4ExpVisionModel:
    # ⚠ The tower is a freetoken ``BaseOP``, not an ``nn.Module`` -- there is no ``.to()`` to cast
    #   it afterwards, so the default dtype has to be set BEFORE it is constructed.
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    model = Qwen4ExpVisionModel(config)
    state = {
        key: (torch.randn_like(value, dtype=dtype) * 0.05)
        if value.dtype.is_floating_point else value
        for key, value in model.state_dict().items()
    }
    model.load_state_dict(state)
    return model


def _forward(model, pixel_values, position_ids, group, monkeypatch) -> torch.Tensor:
    if group is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, str(group))
    with torch.no_grad():
        return model.forward(pixel_values, position_ids)


def _max_diff(config, dtype, group, monkeypatch) -> tuple[float, torch.Size, torch.Size]:
    """One arm at one precision, against the unsplit reference built identically."""
    torch.set_default_dtype(dtype)
    pixel_values, position_ids = _batch(config, COUNTS, dtype)
    model = _model(config, dtype)
    reference = _forward(model, pixel_values, position_ids, None, monkeypatch)
    actual = _forward(model, pixel_values, position_ids, group, monkeypatch)
    return (actual - reference).abs().max().item(), actual.shape, reference.shape


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", 3), ("1", 1), ("0", 0),
        # ⛔ A malformed or nonsensical value must DEGRADE TO THE HISTORICAL PATH, never raise:
        #    this runs inside the tower's forward, so an exception here takes the row down.
        ("", 0), ("nonsense", 0), ("2.5", 0), ("-5", 0), (None, 0),
    ],
)
def test_group_size_degrades_to_unsplit_on_a_bad_value(value, expected, monkeypatch):
    """The env knob is read per forward, so every reachable value must resolve to an int."""
    model = _model(_config(), torch.float32)
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert model._group_size() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", [0, len(COUNTS), len(COUNTS) + 1, 99])
def test_does_not_split_stays_bit_identical(dtype, group, monkeypatch):
    """``0``/unset and any ``G >= N`` take the historical unsplit branch -- exactly, not nearly.

    This is the property that makes the change inert until it is deliberately switched on.
    """
    diff, actual_shape, reference_shape = _max_diff(_config(), dtype, group, monkeypatch)
    assert actual_shape == reference_shape
    assert diff == 0.0, f"G={group} split when it should not have (max|diff| {diff:.3e})"


@pytest.mark.parametrize("group", [1, 2, 3])
def test_splitting_is_rounding_not_a_logic_error(group, monkeypatch):
    """A split must move the result only at float rounding, and precision proves which it is.

    Rounding shrinks with the mantissa; a dropped image or a live padding row would not.
    """
    config = _config()
    float32_diff, actual_shape, reference_shape = _max_diff(
        config, torch.float32, group, monkeypatch
    )
    float64_diff, _, _ = _max_diff(config, torch.float64, group, monkeypatch)

    assert actual_shape == reference_shape
    assert float32_diff < 1e-5, f"G={group}: fp32 gap {float32_diff:.3e} is too large for rounding"
    assert float64_diff == 0.0 or float32_diff / float64_diff > 1e3, (
        f"G={group}: fp32 {float32_diff:.3e} vs fp64 {float64_diff:.3e} does not collapse with "
        "precision -- this is a logic error, not rounding"
    )


def test_padding_is_trimmed_not_read(monkeypatch):
    """A group is trimmed to its OWN longest image, so a short group stops paying the batch's max.

    Encoding the ragged batch one image at a time must agree with encoding each image alone in a
    width-1 batch of its own natural size -- which is only true if the trim drops padding only.
    """
    config = _config()
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    pixel_values, position_ids = _batch(config, COUNTS, dtype)
    model = _model(config, dtype)
    grouped = _forward(model, pixel_values, position_ids, 1, monkeypatch)

    merge = config.spatial_merge_size ** 2
    offset = 0
    for row, count in enumerate(COUNTS):
        alone = _forward(
            model, pixel_values[row : row + 1, :count], position_ids[row : row + 1, :count],
            1, monkeypatch,
        )
        tokens = count // merge
        assert torch.equal(grouped[offset : offset + tokens], alone), (
            f"image {row}: encoding it inside the ragged batch differs from encoding it alone, "
            "so the trim is reading or emitting padding"
        )
        offset += tokens
    assert offset == grouped.shape[0]
