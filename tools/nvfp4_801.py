"""#801 round 2 bullet 2 — NVFP4 quantise/dequantise that reproduces the publisher's own bytes.

The MTP head this issue wants to wire up ships **bf16** (`hf_quant_config.json` excludes `mtp.*`
from quantisation), while the engine's offload cache only knows how to stream NVFP4 expert banks.
So the head's 512 experts have to be requantised, and the requantiser has to agree with the
checkpoint producer *exactly* -- a schema that is merely plausible loads, runs, and returns noise.

⭐ THE SCHEMA, read off the publisher's bytes and the engine's kernels rather than assumed. For an
expert weight ``W[N, K]``:

    W[n, k] = E2M1[code[n, k]] * block_scale[n, k // 16] * global[n]

  * ``packed[n, k // 2]`` uint8 -- two 4-bit codes per byte, **low nibble = even k**
    (`kernel/triton/nvfp4_fused_moe.py`'s own docstring).
  * ``block_scale[n, k // 16]`` fp8-e4m3 -- one scale per 16 values, the checkpoint's declared
    ``group_size``.
  * ``global`` fp32 -- the publisher's ``weight_scale_2``.

⭐ THE RECIPE, MEASURED (this bullet's finding; the previous round could only model it):

    global       = amax(|W_bf16|) / (6 * 448)          # e2m1 max * fp8-e4m3 max
    block_scale  = to_fp8(block_amax / 6 / global)
    code         = to_e2m1(W / (block_scale * global))

`test_nvfp4_801.py` pins the first line against the checkpoint: ``weight_scale_2 * 6 * 448`` lands
within one fp32 ULP of a **bf16** number for every published expert-projection, which an arbitrary
float does not do.

⛔ THE GRANULARITY IS NOT WHAT THE ISSUE ASSUMED. The publisher does not write one global per
expert-projection. `gate_proj` and `up_proj` **share** one global across all 128 experts of a shard
(ModelOpt quantised them as a single fused module), and there are exactly 4 distinct values across
the 512 experts of layer 0 -- one per shard. `down_proj` has its own per expert. The engine's bank,
meanwhile, stores a global per **row** (`[E, 2I] fp16`). :func:`global_scale_for` therefore takes
the granularity as a parameter, and bullet 3 picks between them on measured round-trip error
rather than on either side's convention.

⚠ Pure python + numpy: the host has neither torch nor safetensors, and numpy has no fp8 dtype.
Both number formats are built here from their bit layouts, once, as lookup tables.

Usage:
    python3 nvfp4_801.py [model_path]     # the round trip against a published expert
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from head_801 import (  # noqa: E402
    NVFP4_BLOCK,
    HeadManifest,
    safetensors_memmap,
    safetensors_tensor,
    widen_bf16,
    write_safetensors,
)

# The e2m1 code table, copied from the image's `kernel/triton/nvfp4_fused_moe.py::_E2M1_VALUES`.
# ⛔ This is the table the inference kernel gathers through, so it -- not the IEEE paper -- is the
# definition our written bytes have to satisfy. Index = the 4-bit code; codes 8..15 are the
# negatives of 0..7, so bit 3 is the sign and -0.0 is a real, used code.
E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0

_E2M1_TABLE = np.array(E2M1_VALUES, dtype=np.float32)
_E2M1_MAGNITUDES = _E2M1_TABLE[:8].astype(np.float64)

_E4M3_NAN_CODES = (0x7F, 0xFF)


def _build_e4m3_table() -> np.ndarray:
    """Every fp8-e4m3 code's value: 4 exponent bits (bias 7), 3 mantissa bits, `fn` flavour.

    `fn` means *finite*: the pattern the wider formats spend on infinity is spent on values
    instead, the maximum is 448 rather than 240, and the single NaN is the all-ones mantissa at
    the top exponent.
    """
    code = np.arange(256, dtype=np.uint32)
    sign, exp, mant = (code >> 7) & 1, (code >> 3) & 0xF, code & 0x7
    subnormal = (mant / 8.0) * 2.0 ** -6
    normal = (1.0 + mant / 8.0) * np.exp2(exp.astype(np.float64) - 7)
    value = np.where(exp == 0, subnormal, normal)
    value = np.where(sign == 1, -value, value)
    value[list(_E4M3_NAN_CODES)] = np.nan
    return value.astype(np.float32)


_E4M3_TABLE = _build_e4m3_table()
# Codes 0x00..0x7E are the non-negative finite values in increasing order, which is what makes a
# searchsorted encoder possible at all.
_E4M3_MAGNITUDES = _E4M3_TABLE[:_E4M3_NAN_CODES[0]].astype(np.float64)


def _nearest_even_index(magnitude: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Round each magnitude to the index of the nearest level, ties to the **even** index.

    ⚠ [MODELLED] where it bites -- see the module docstring's note and the tie test. For both
    formats here the mantissa's low bit is the level index's low bit, so "even index" is
    "round-to-nearest-even" in the IEEE sense.

    A tie is detected rather than nudged: `searchsorted` disagrees with itself between `left` and
    `right` exactly when the value sits on a midpoint.
    """
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    lower = np.searchsorted(midpoints, magnitude, side="left")
    upper = np.searchsorted(midpoints, magnitude, side="right")
    tie_target = np.where(lower % 2 == 0, lower, lower + 1)
    return np.where(lower == upper, upper, tie_target)


def e4m3_to_float32(raw: np.ndarray) -> np.ndarray:
    """Decode raw fp8-e4m3 bytes. numpy has no fp8 dtype, so the block scales travel as `uint8`."""
    return _E4M3_TABLE[np.asarray(raw, dtype=np.uint8)]


def float32_to_e4m3(values: np.ndarray) -> np.ndarray:
    """Encode to fp8-e4m3 bytes, round-to-nearest-even, **saturating** at ±448.

    ⛔ Saturating, not wrapping: the codes just past 448 are the format's NaN, and a block scale
    that silently became NaN would turn 16 weights into NaN with nothing raised anywhere.
    """
    values = np.asarray(values, dtype=np.float32)
    magnitude = np.minimum(np.abs(values).astype(np.float64), FP8_E4M3_MAX)
    code = _nearest_even_index(magnitude, _E4M3_MAGNITUDES).astype(np.uint8)
    return np.where(np.signbit(values), code | 0x80, code).astype(np.uint8)


_GRANULARITIES = ("tensor", "row")


def global_scale_for(w: np.ndarray, granularity: str = "tensor") -> np.ndarray:
    """The ``weight_scale_2`` that maps ``w``'s largest element onto the top of both formats.

    ``"tensor"`` is the publisher's own granularity -- one scalar over everything handed in, which
    for the head's stacked `[E, 2I, H]` experts means one scalar for all 512 at once, matching what
    ModelOpt did to the fused gate/up module. ``"row"`` is the engine bank's, one per output row.

    ⛔ An unknown granularity raises rather than falling back: a typo that quietly reverted to
    per-tensor would come back as a fidelity *result* in bullet 3, not as an error.
    """
    if granularity not in _GRANULARITIES:
        raise ValueError(f"granularity must be one of {_GRANULARITIES}, got {granularity!r}")
    w = np.asarray(w, dtype=np.float32)
    axis = None if granularity == "tensor" else tuple(range(1, w.ndim))
    amax = np.abs(w).max(axis=axis).astype(np.float64)
    return (amax / (E2M1_MAX * FP8_E4M3_MAX)).astype(np.float32)


@dataclass(frozen=True)
class QuantisedNVFP4:
    """One NVFP4 weight, in the three pieces the engine's bank stores it as.

    ``scale`` is deliberately raw `uint8` rather than a decoded float: it is compared byte-for-byte
    against the publisher's, and it is written to the artefact as the bytes the loader `memcpy`s.
    """

    packed: np.ndarray       # uint8 [N, K // 2], low nibble = even k
    scale: np.ndarray        # uint8 [N, K // NVFP4_BLOCK], raw fp8-e4m3
    global_scale: np.ndarray  # float32, scalar or [N]

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.scale.nbytes + np.asarray(self.global_scale).nbytes


def _as_row_scale(global_scale, rows: int) -> np.ndarray:
    """Broadcast a scalar or per-row global to a `[rows, 1]` column, so both call shapes divide."""
    g = np.asarray(global_scale, dtype=np.float64)
    if g.ndim == 0:
        return np.full((rows, 1), float(g))
    if g.shape != (rows,):
        raise ValueError(f"global_scale must be a scalar or shape ({rows},), got {g.shape}")
    return g.reshape(rows, 1)


def quantise_nvfp4(w: np.ndarray, *, global_scale) -> QuantisedNVFP4:
    """Quantise ``w[N, K]`` to packed e2m1 codes + fp8 block scales at the given global.

    The global is an input, not something derived here, because the caller has to be able to hand
    in the publisher's own -- that is what makes the round-trip gate a test of *our* arithmetic
    rather than a test of two independent amax reductions agreeing.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D [N, K] weight, got shape {w.shape}")
    rows, k = w.shape
    if k % NVFP4_BLOCK:
        raise ValueError(f"K={k} is not a multiple of the {NVFP4_BLOCK}-wide NVFP4 block")

    g = _as_row_scale(global_scale, rows)
    blocks = np.abs(w.astype(np.float64)).reshape(rows, k // NVFP4_BLOCK, NVFP4_BLOCK)
    # ⛔ `where=g > 0` rather than a bare divide: an all-zero row has an amax of 0 and so a global
    # of 0, and `0 / 0` would hand the fp8 encoder a NaN that becomes a NaN block scale -- 16
    # weights turned to NaN with nothing raised anywhere.
    block_amax = blocks.max(axis=2)
    scale = float32_to_e4m3(np.divide(block_amax, E2M1_MAX * g,
                                      out=np.zeros_like(block_amax), where=g > 0.0))

    # The divisor the codes are measured against is the scale as STORED, not as computed: the fp8
    # rounding above already happened, and the kernel will read back the stored byte.
    divisor = (e4m3_to_float32(scale).astype(np.float64) * g).repeat(NVFP4_BLOCK, axis=1)
    ratio = np.divide(np.abs(w.astype(np.float64)), divisor,
                      out=np.zeros_like(divisor), where=divisor > 0.0)
    code = _nearest_even_index(np.minimum(ratio, E2M1_MAX), _E2M1_MAGNITUDES).astype(np.uint8)
    code |= (np.signbit(w) << 3).astype(np.uint8)

    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return QuantisedNVFP4(packed=packed, scale=scale,
                          global_scale=np.asarray(global_scale, dtype=np.float32))


def dequantise_nvfp4(packed: np.ndarray, scale: np.ndarray, *, global_scale) -> np.ndarray:
    """The kernel's own product, in numpy: ``E2M1[code] * block_scale * global`` -> `[N, K]`."""
    packed = np.asarray(packed, dtype=np.uint8)
    rows, half_k = packed.shape
    code = np.empty((rows, half_k * 2), dtype=np.uint8)
    code[:, 0::2] = packed & 0xF          # low nibble = even k
    code[:, 1::2] = packed >> 4
    block_scale = e4m3_to_float32(scale).repeat(NVFP4_BLOCK, axis=1)
    return (_E2M1_TABLE[code] * block_scale * _as_row_scale(global_scale, rows).astype(np.float32)
            ).astype(np.float32)


# ======================================================================================
# Bullet 3 -- the head's stacked experts, requantised into the engine's six-tensor bank
# ======================================================================================

# The bank the engine allocates, as `models/nvfp4_banks.py::_alloc_nvfp4_host_banks` names it:
# ``{bank: (safetensors dtype, numpy dtype)}``. ⛔ `gate_up_scale` and `down_scale` are `F8_E4M3`
# and travel as `uint8` because numpy has no fp8 -- the dtype lives in the header, not in the
# array, which is why `write_safetensors` takes it explicitly.
BANK_SCHEMA: dict[str, tuple[str, type]] = {
    "gate_up_packed": ("U8", np.uint8),
    "gate_up_scale": ("F8_E4M3", np.uint8),
    "gate_up_global": ("F16", np.float16),
    "down_packed": ("U8", np.uint8),
    "down_scale": ("F8_E4M3", np.uint8),
    "down_global": ("F16", np.float16),
}

_HEAD_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
_HEAD_DOWN = "mtp.layers.0.mlp.experts.down_proj"
_ARTEFACT_NAME = "mtp-experts-nvfp4.safetensors"

# One global for the whole stacked tensor / one per expert-projection / one per output row. The
# first two are both the publisher's -- it uses per-shard-fused for gate/up and per-expert for
# down (bullet 2's finding) -- and the third is the engine bank's own.
BANK_GRANULARITIES = ("tensor", "expert", "row")

# Roughly this many elements per expert go into the p99 sample, by a fixed stride. Deterministic
# on purpose: a seeded RNG would still make two runs of the same arm disagree in the last digit.
_P99_SAMPLES_PER_EXPERT = 10_000


@dataclass(frozen=True)
class BankError:
    """Requantisation error for one bank, against the checkpoint's own bf16 as the control.

    ⚠ ``max_abs`` and ``mean_abs`` are exact over every element. ``p99_abs`` is over a fixed-stride
    sample (~10k per expert, ~5M for the head) -- an exact percentile over 1.7e9 elements would
    need the whole error tensor resident, and the sample is stated rather than rounded away.
    """

    bank: str
    amax: float        # amax of the SOURCE, so the absolute figures can be read relatively
    # ⭐ rms of the SOURCE weights. Error as a fraction of `amax` flatters NVFP4 badly -- expert
    # weights are heavy-tailed, so `amax` is an outlier and dividing by it turns a 9 % error into
    # a 0.1 % one. `rms_over_source` is the figure a fidelity claim should quote.
    source_rms: float
    max_abs: float
    mean_abs: float
    rms: float
    p99_abs: float
    # ⭐ The worst output row's error as a fraction of that row's own size. The absolute figures
    # above are dominated by the largest rows and are BLIND to the failure that actually matters
    # here: a row whose block scales underflow fp8 quantises to all zeros, which is a 100 % error
    # on that output channel and a rounding error in `rms`. This is the number the global-scale
    # granularity is chosen on.
    max_row_rel: float

    @property
    def max_rel(self) -> float:
        return self.max_abs / self.amax if self.amax else 0.0

    @property
    def rms_rel(self) -> float:
        return self.rms / self.amax if self.amax else 0.0

    @property
    def rms_over_source(self) -> float:
        """The honest one: requantisation error as a fraction of the weights' own spread."""
        return self.rms / self.source_rms if self.source_rms else 0.0


@dataclass(frozen=True)
class RequantReport:
    """What one requantisation arm produced and what it cost in fidelity."""

    granularity: str
    num_experts: int
    artefact: Path | None
    artefact_bytes: int
    shapes: dict[str, tuple[int, ...]]
    errors: tuple[BankError, ...]
    # ⚠ Pooled standard deviation of the two halves of the stacked `[E, 2I, H]` gate_up, lower
    # rows then upper. Circumstantial evidence only -- see `requantise_head_experts`.
    gate_up_half_std: tuple[float, float]

    def error(self, bank: str) -> BankError:
        for err in self.errors:
            if err.bank == bank:
                return err
        raise KeyError(f"no error banked for {bank!r}")


class _ErrorAccumulator:
    """Running exact max/mean/rms plus a fixed-stride p99 sample, over experts streamed one by one."""

    def __init__(self, bank: str):
        self.bank = bank
        self.amax = 0.0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.source_sum_sq = 0.0
        self.count = 0
        self.max_row_rel = 0.0
        self.samples: list[np.ndarray] = []

    def note(self, source: np.ndarray, error: np.ndarray) -> None:
        err = np.abs(error, dtype=np.float64)
        self.amax = max(self.amax, float(np.abs(source).max()))
        self.max_abs = max(self.max_abs, float(err.max()))
        self.sum_abs += float(err.sum())
        self.sum_sq += float((err ** 2).sum())
        self.source_sum_sq += float((source.astype(np.float64) ** 2).sum())
        self.count += err.size
        stride = max(1, err.size // _P99_SAMPLES_PER_EXPERT)
        self.samples.append(err.ravel()[::stride].astype(np.float32))

        # Per-row rms error over per-row rms magnitude. Rows that are entirely zero in the source
        # have nothing to be wrong about and are skipped rather than counted as perfect.
        row_err = np.sqrt((err ** 2).mean(axis=1))
        row_mag = np.sqrt((source.astype(np.float64) ** 2).mean(axis=1))
        live = row_mag > 0.0
        if live.any():
            self.max_row_rel = max(self.max_row_rel,
                                   float((row_err[live] / row_mag[live]).max()))

    def finish(self) -> BankError:
        return BankError(
            bank=self.bank,
            amax=self.amax,
            source_rms=(self.source_sum_sq / self.count) ** 0.5,
            max_abs=self.max_abs,
            mean_abs=self.sum_abs / self.count,
            rms=(self.sum_sq / self.count) ** 0.5,
            p99_abs=float(np.percentile(np.concatenate(self.samples), 99)),
            max_row_rel=self.max_row_rel,
        )


class _HalfSpread:
    """Pooled std of the lower and upper halves of the stacked gate_up rows."""

    def __init__(self):
        self.n = [0, 0]
        self.total = [0.0, 0.0]
        self.total_sq = [0.0, 0.0]

    def note(self, rows: np.ndarray) -> None:
        half = rows.shape[0] // 2
        for i, part in enumerate((rows[:half], rows[half:])):
            part = part.astype(np.float64)
            self.n[i] += part.size
            self.total[i] += float(part.sum())
            self.total_sq[i] += float((part ** 2).sum())

    def finish(self) -> tuple[float, float]:
        out = []
        for i in (0, 1):
            mean = self.total[i] / self.n[i]
            out.append(max(self.total_sq[i] / self.n[i] - mean ** 2, 0.0) ** 0.5)
        return (out[0], out[1])


def _bank_global(expert: np.ndarray, granularity: str, stacked_amax: float) -> np.ndarray:
    """The global scale for one expert's `[N, K]` weight, as fp16 -- the bank's own width.

    ⭐ Rounded to fp16 **before** quantising, not after. The engine reads the fp16 bank, so a
    quantiser that used the fp32 value and then stored a rounded one would write codes that
    dequantise to something the engine never reproduces -- and the measured error would be a
    number nothing on the card can achieve. The real checkpoint makes this concrete: published
    `down_proj` globals run down to 2.27e-5, which is inside fp16's subnormals.
    """
    if granularity == "tensor":
        g = np.full(expert.shape[0], stacked_amax / (E2M1_MAX * FP8_E4M3_MAX), dtype=np.float64)
    elif granularity == "expert":
        g = np.full(expert.shape[0], float(global_scale_for(expert, "tensor")), dtype=np.float64)
    else:
        g = global_scale_for(expert, "row").astype(np.float64)
    return g.astype(np.float16)


def _stacked_amax(mm: np.memmap) -> float:
    """amax over every expert of a stacked bf16 tensor, one expert at a time."""
    return max(float(np.abs(widen_bf16(np.asarray(mm[e]))).max()) for e in range(mm.shape[0]))


def requantise_head_experts(
    manifest: HeadManifest,
    out_dir: str | Path | None,
    *,
    granularity: str = "row",
) -> RequantReport:
    """Requantise the head's 512 stacked bf16 experts into the engine's six NVFP4 banks.

    ``out_dir=None`` measures without writing -- a granularity sweep is three 4.8 GiB reads and
    should not also be three 1.4 GiB writes.

    ⛔ ROW ORDER IS PRESERVED VERBATIM, and that is a decision rather than an oversight. The bank's
    `gate_up_packed[e, :I]` is what the engine treats as *gate* and `[I:]` as *up*
    (`models/nvfp4_banks.py`, and every adapter that fuses a separate pair does so in that order).
    The head ships one stacked `[E, 2I, H]` tensor, and nothing in its bytes says which half is
    which; requantisation does not care (it is row-independent), the forward pass does. So the
    halves are copied in the order the checkpoint wrote them and their spreads are banked in
    `gate_up_half_std` as circumstantial evidence. ⚠ **Only a forward pass settles it** -- the
    issue's own bullet 3, where a module exists to run one.
    """
    if granularity not in BANK_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {BANK_GRANULARITIES}, got {granularity!r}"
        )
    gate_up_t = manifest.by_name(_HEAD_GATE_UP)
    down_t = manifest.by_name(_HEAD_DOWN)
    num_experts = gate_up_t.shape[0]

    sources = {
        "gate_up": safetensors_memmap(manifest.model_path / gate_up_t.file, _HEAD_GATE_UP),
        "down": safetensors_memmap(manifest.model_path / down_t.file, _HEAD_DOWN),
    }
    shapes: dict[str, tuple[int, ...]] = {}
    for bank, mm in sources.items():
        _, rows, k = mm.shape
        shapes[f"{bank}_packed"] = (num_experts, rows, k // 2)
        shapes[f"{bank}_scale"] = (num_experts, rows, k // NVFP4_BLOCK)
        shapes[f"{bank}_global"] = (num_experts, rows)

    banks = None
    if out_dir is not None:
        banks = {name: np.zeros(shapes[name], dtype=BANK_SCHEMA[name][1]) for name in BANK_SCHEMA}

    # A per-tensor global needs the whole stacked amax before any expert can be quantised; the
    # other two granularities are decided per expert and need no prepass.
    stacked_amax = {
        bank: (_stacked_amax(mm) if granularity == "tensor" else 0.0)
        for bank, mm in sources.items()
    }

    accumulators = {bank: _ErrorAccumulator(bank) for bank in sources}
    halves = _HalfSpread()
    for bank, mm in sources.items():
        for e in range(num_experts):
            w = widen_bf16(np.asarray(mm[e]))
            if bank == "gate_up":
                halves.note(w)
            g16 = _bank_global(w, granularity, stacked_amax[bank])
            q = quantise_nvfp4(w, global_scale=g16.astype(np.float32))
            accumulators[bank].note(
                w, dequantise_nvfp4(q.packed, q.scale, global_scale=g16.astype(np.float32)) - w
            )
            if banks is not None:
                banks[f"{bank}_packed"][e] = q.packed
                banks[f"{bank}_scale"][e] = q.scale
                banks[f"{bank}_global"][e] = g16

    artefact, written = None, 0
    if banks is not None:
        artefact = Path(out_dir) / _ARTEFACT_NAME
        written = write_safetensors(
            artefact, {name: (BANK_SCHEMA[name][0], banks[name]) for name in BANK_SCHEMA}
        )

    return RequantReport(
        granularity=granularity,
        num_experts=num_experts,
        artefact=artefact,
        artefact_bytes=written,
        shapes=shapes,
        errors=tuple(accumulators[bank].finish() for bank in ("gate_up", "down")),
        gate_up_half_std=halves.finish(),
    )



# ======================================================================================
# CLI
# ======================================================================================

_DEFAULT_MODEL = "/home/luka/models/Qwen3.8-Flash-Next-NVFP4"
_DEFAULT_BANK = "/home/luka/freetoken-mtp801/mtp-bank"
_PUB_SHARD = "layer-00000-experts-0000-0127.safetensors"


def _gate(model: Path) -> int:
    """Run the gate on a published expert and print what it found, so the number can be banked."""
    shard = model / _PUB_SHARD
    print(f"# round trip against {shard.name}, layer 0 expert 0\n")
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.language_model.layers.0.mlp.experts.0.{projection}."
        packed = safetensors_tensor(shard, base + "weight")
        scale = safetensors_tensor(shard, base + "weight_scale")
        g = safetensors_tensor(shard, base + "weight_scale_2")

        w = dequantise_nvfp4(packed, scale, global_scale=g)
        q = quantise_nvfp4(w, global_scale=g)
        bad_codes = int(np.count_nonzero(q.packed != packed))
        bad_scales = int(np.count_nonzero(q.scale != scale))
        ours = global_scale_for(w, "tensor")
        print(f"{projection:<10s} {str(list(w.shape)):<14s} global {float(g):.9g}  "
              f"code mismatches {bad_codes:>10d}  scale mismatches {bad_scales:>8d}")
        print(f"           amax(|W|) {np.abs(w).max():.6g}   our per-tensor global would be "
              f"{float(ours):.9g} ({float(g) / float(ours):.4g}x the publisher's)")
    return 0


def _report_line(report: RequantReport) -> str:
    parts = [f"{report.granularity:<8s}"]
    for err in report.errors:
        parts.append(
            f"{err.bank:>8s} rms {err.rms:.6g} = {100 * err.rms_over_source:5.2f}% of the "
            f"weights' own rms {err.source_rms:.6g} ({100 * err.rms_rel:.2f}% of amax {err.amax:.4g})"
            f"\n           {'':>8s} max {err.max_abs:.6g}  p99 {err.p99_abs:.6g}  "
            f"worst row {100 * err.max_row_rel:6.2f}%"
        )
    return "\n           ".join(parts)


def _requant(model: Path, out_dir: Path | None, granularities: tuple[str, ...]) -> int:
    """Sweep the global-scale granularities, and optionally write the artefact for the last one."""
    from head_801 import head_manifest

    man = head_manifest(model)
    print(f"# requantising {man.by_name(_HEAD_GATE_UP).shape[0]} head experts from {model}\n")
    for granularity in granularities:
        started = time.monotonic()
        write_to = out_dir if granularity == granularities[-1] else None
        report = requantise_head_experts(man, write_to, granularity=granularity)
        print(_report_line(report) + f"   [{time.monotonic() - started:.0f}s]")
        if report.artefact is not None:
            print(f"\n  wrote {report.artefact}  {report.artefact_bytes / 2**30:.3f} GiB")
            for name, shape in sorted(report.shapes.items()):
                print(f"    {name:<16s} {list(shape)}  {BANK_SCHEMA[name][0]}")
        lower, upper = report.gate_up_half_std
        print(f"  gate_up stacked halves: rows[:I] std {lower:.6g}   rows[I:] std {upper:.6g}\n")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default=_DEFAULT_MODEL)
    parser.add_argument("--requant", action="store_true",
                        help="sweep the global-scale granularities over the head's experts")
    parser.add_argument("--write", nargs="?", const=_DEFAULT_BANK, default=None,
                        help="also write the bank artefact (implies --requant)")
    parser.add_argument("--granularity", action="append", choices=BANK_GRANULARITIES,
                        help="restrict the sweep; repeatable, the LAST one is what gets written")
    args = parser.parse_args(argv[1:])

    if not (args.requant or args.write):
        return _gate(Path(args.model))
    return _requant(Path(args.model),
                    Path(args.write) if args.write else None,
                    tuple(args.granularity or BANK_GRANULARITIES))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
