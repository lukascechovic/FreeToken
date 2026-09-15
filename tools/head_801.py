"""#801 round 2 bullet 1 — what the checkpoint's MTP head actually is, read from its own bytes.

The `Qwen3.8-Flash-Next-NVFP4` checkpoint ships a multi-token-prediction head that our FreeToken
build drops on load (`models/qwen4_exp/weight.py:300`). #801 asks whether wiring it up raises
decode on the offload row. Everything downstream -- the NVFP4 requantiser, the loader flag, the
draft module -- is priced on what that head weighs and what shape it is, so this module reads that
off the shards rather than carrying it from a desk table.

⚠ Pure python + numpy on purpose: the box's host python has **neither torch nor safetensors**, and
a safetensors file needs neither. It is an 8-byte little-endian header length, that many bytes of
JSON naming every tensor's dtype, shape and byte range, then the raw bytes.

Usage:
    python3 head_801.py [model_path]        # the manifest table
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HEADER_LEN_BYTES = 8

# The NVFP4 block: 16 values share one FP8_E4M3 scale. `config.json`'s quantization_config
# declares `group_size: 16` for both weights and activations, and the engine's bank schema is
# written against the same 16.
NVFP4_BLOCK = 16


def safetensors_header(path: str | Path) -> dict[str, dict]:
    """The tensor table of one safetensors shard: ``{name: {"dtype", "shape", "data_offsets"}}``.

    ``__metadata__`` is dropped -- it is a free-form string map the format stores beside the
    tensors, and a caller walking the header would otherwise treat it as one.
    """
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(_HEADER_LEN_BYTES))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header



# The on-disk element types this checkpoint uses, as little-endian numpy dtypes. ⚠ Two of them
# have no numpy dtype at all and are handed back as their raw bytes or widened:
#   * `F8_E4M3` -- numpy has no fp8; the caller decodes the byte (`nvfp4_801.e4m3_to_float32`).
#   * `BF16`    -- numpy has no bf16 either, but widening is exact and needs no library: a bf16 is
#                  the top 16 bits of the float32 with the same value, so `uint16 << 16` viewed as
#                  float32 IS the number, with no rounding anywhere.
_NUMPY_DTYPE = {
    "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4",
    "U8": "|u1", "F8_E4M3": "|u1",
}


def widen_bf16(raw: np.ndarray) -> np.ndarray:
    """bf16 `uint16` bits -> float32, exactly and with no library.

    A bf16 IS the top 16 bits of the float32 with the same value, so this is a shift and a
    reinterpretation: no rounding, no approximation, nothing to get wrong.
    """
    return (np.asarray(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def safetensors_memmap(path: str | Path, name: str) -> np.memmap:
    """One tensor's raw elements as a read-only memmap, for tensors too big to hold.

    The head's stacked `gate_up_proj` is 3.2 GiB and its `down_proj` 1.6 GiB; the requantiser walks
    them one expert at a time and never materialises either whole. ⚠ `BF16` comes back as `uint16`
    -- numpy has no bf16 dtype -- so the caller widens each slice with :func:`widen_bf16`.
    """
    path = Path(path)
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(_HEADER_LEN_BYTES))
        entry = json.loads(f.read(header_len))[name]
    lo, _ = entry["data_offsets"]
    dtype = "<u2" if entry["dtype"] == "BF16" else _NUMPY_DTYPE[entry["dtype"]]
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(entry["shape"]),
                     offset=_HEADER_LEN_BYTES + header_len + lo)


def safetensors_layout(
    tensors: dict[str, tuple[str, tuple[int, ...]]]
) -> tuple[bytes, int, int]:
    """``{name: (dtype_string, shape)}`` -> (padded header blob, data bytes, total file bytes).

    ⭐ Factored out so a PLAN and the WRITE cannot disagree. Predicting an artefact's size with
    arithmetic beside the writer is how a published recipe comes to promise a number the tool does
    not produce: the header is JSON whose length depends on the names, the shapes and the offsets
    themselves, so it is not something to estimate. ⛔ Anything that quotes a byte total must get it
    from here.
    """
    header: dict[str, dict] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _ITEMSIZE[dtype]
        for dim in shape:
            nbytes *= dim
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    return blob, offset, _HEADER_LEN_BYTES + len(blob) + offset


def write_safetensors(path: str | Path, tensors: dict[str, tuple[str, np.ndarray]]) -> int:
    """Write a safetensors file: ``{name: (dtype_string, array)}`` -> total bytes written.

    ⭐ The dtype string is given, never inferred. Two of the six banks this round writes are
    `F8_E4M3` and travel as `uint8`, and `U8` is also `uint8`; a writer that guessed from the numpy
    dtype would label the block scales as plain bytes and the engine's loader would reject the
    file -- or worse, read them as integers.

    The format wants the data section 8-byte aligned, so the JSON header is padded with spaces
    (which JSON ignores) rather than with NULs.
    """
    contiguous: dict[str, tuple[str, np.ndarray]] = {}
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array)
        if raw.itemsize != _ITEMSIZE[dtype]:
            raise ValueError(
                f"{name}: {dtype} is {_ITEMSIZE[dtype]} B/element but the array is {raw.itemsize}"
            )
        contiguous[name] = (dtype, raw)

    # ⛔ Header AND total come from safetensors_layout, the one place that knows the format -- see
    #   its docstring. Recomputing the total here is how a writer and a --plan drift apart.
    blob, _data_bytes, total = safetensors_layout(
        {name: (dtype, raw.shape) for name, (dtype, raw) in contiguous.items()}
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _dtype, raw in contiguous.values():
            raw.tofile(f)
    return total


def safetensors_tensor(path: str | Path, name: str) -> np.ndarray:
    """One tensor's data, read out of a safetensors shard with no safetensors library.

    The header gives the tensor's byte range *relative to the end of the header*, so the absolute
    offset is ``8 + header_len + lo``. Returns a read-only view's copy, shaped as the header says;
    a scalar tensor (``shape: []``) comes back as a 0-d array, which ``float()`` accepts.
    """
    path = Path(path)
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(_HEADER_LEN_BYTES))
        entry = json.loads(f.read(header_len))[name]
        lo, hi = entry["data_offsets"]
        f.seek(_HEADER_LEN_BYTES + header_len + lo)
        raw = f.read(hi - lo)

    shape = tuple(entry["shape"])
    if entry["dtype"] == "BF16":
        return widen_bf16(np.frombuffer(raw, dtype="<u2")).reshape(shape)
    return np.frombuffer(raw, dtype=_NUMPY_DTYPE[entry["dtype"]]).reshape(shape)


_MTP_PREFIX = "mtp."
_INDEX = "model.safetensors.index.json"

# Bytes per element, for the dtypes this checkpoint actually uses. The head is entirely BF16;
# the publisher's routed experts are the NVFP4 triple (U8 packed / F8_E4M3 block scales / F32
# global), which bullet 2 reads out of the same shards.
_ITEMSIZE = {"BF16": 2, "F32": 4, "F16": 2, "F8_E4M3": 1, "U8": 1, "I64": 8, "I32": 4}


@dataclass(frozen=True)
class HeadTensor:
    """One checkpoint tensor of the MTP head, as the shard header describes it."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    file: str

    @property
    def is_expert(self) -> bool:
        """The stacked routed experts -- the 4,800 MiB that has to be requantised.

        ⚠ The head stores them STACKED (``mtp.layers.0.mlp.experts.gate_up_proj``, one tensor for
        all 512 experts), while the main layers store one tensor per expert per projection. That
        difference is why `weight.py`'s ``^model\\.language_model\\.`` anchor excludes them and why
        they cannot ride the existing per-expert bank reader.
        """
        return ".mlp.experts." in self.name


@dataclass(frozen=True)
class HeadGeometry:
    """The head's shape, as its own tensors state it.

    ``hc_count`` is the multi-stream hyper-connection width the head inherits from the backbone:
    ``pre_fc_norm_hidden`` is ``hc_count * hidden_size`` wide because the fusion reads the main
    model's PRE-final-mixer multi-stream residual, not the collapsed hidden state.
    """

    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hc_count: int
    num_layers: int

    def nvfp4_bank_shapes(self) -> dict[str, tuple[int, ...]]:
        """The six NVFP4 source banks the engine allocates for one MoE layer, at this geometry.

        Mirrors `models/nvfp4_banks.py::_alloc_nvfp4_host_banks` -- gate and up fused on the
        output-row axis, down separate, e2m1 packed two values per byte along the reduction axis,
        `NVFP4_BLOCK`-wide block scales, and a per-row FP16 global.

        ⚠ The globals are per ROW here because that is the bank's own granularity; the publisher
        writes one scalar per expert-projection and broadcasts it. Which of the two the head
        should use is a fidelity question bullet 2 measures rather than assumes.
        """
        E, H, I = self.num_experts, self.hidden_size, self.moe_intermediate_size
        return {
            "gate_up_packed": (E, 2 * I, H // 2),
            "gate_up_scale": (E, 2 * I, H // NVFP4_BLOCK),
            "gate_up_global": (E, 2 * I),
            "down_packed": (E, H, I // 2),
            "down_scale": (E, H, I // NVFP4_BLOCK),
            "down_global": (E, H),
        }


@dataclass(frozen=True)
class HeadManifest:
    """Every ``mtp.*`` tensor in the checkpoint, with the geometry implied by their shapes."""

    model_path: Path
    tensors: tuple[HeadTensor, ...]


    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def expert_bytes(self) -> int:
        """The stacked routed experts -- the part that has to be requantised or pinned as bf16."""
        return sum(t.nbytes for t in self.tensors if t.is_expert)

    @property
    def dense_bytes(self) -> int:
        """Everything else: the QSA layer, the two fusion projections, the HC mixers, the norms.

        This is the part #801 prices as resident GPU weight (~170 MiB, about 60 expert cache
        slots), against the experts' 4.7 GiB of pinned host RAM.
        """
        return self.total_bytes - self.expert_bytes

    @property
    def geometry(self) -> HeadGeometry:
        """The head's own shapes, read as geometry. ⭐ Every figure here is compared against the
        checkpoint's `config.json` by the tests: if the head's expert geometry did not match the
        main layers', the existing NVFP4 bank schema would not transfer and bullet 2 would need a
        second schema rather than a requantiser."""
        gate_up = self.by_name("mtp.layers.0.mlp.experts.gate_up_proj")
        down = self.by_name("mtp.layers.0.mlp.experts.down_proj")
        num_experts, two_i, hidden = gate_up.shape
        layers = {
            int(t.name.split(".")[2])
            for t in self.tensors
            if t.name.startswith("mtp.layers.")
        }
        return HeadGeometry(
            num_experts=num_experts,
            hidden_size=hidden,
            # ⚠ Taken from `down_proj`'s [E, H, I], not from `gate_up_proj`'s 2I halved: the fused
            # tensor only tells us 2I, and reading I out of it would assume the [gate | up] split
            # this round deliberately does NOT assume (it is bullet 3's question).
            moe_intermediate_size=down.shape[2],
            shared_expert_intermediate_size=self.by_name(
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight").shape[0],
            hc_count=self.by_name("mtp.pre_fc_norm_hidden.weight").shape[0] // hidden,
            num_layers=max(layers) + 1,
        )

    def by_name(self, name: str) -> HeadTensor:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(f"{name} is not in the head's manifest")


def head_manifest(model_path: str | Path) -> HeadManifest:
    """Read the checkpoint index, then the header of each shard holding an ``mtp.*`` tensor."""
    model_path = Path(model_path)
    weight_map = json.loads((model_path / _INDEX).read_text())["weight_map"]
    names = sorted(n for n in weight_map if n.startswith(_MTP_PREFIX))

    headers: dict[str, dict] = {}
    tensors = []
    for name in names:
        file = weight_map[name]
        if file not in headers:
            headers[file] = safetensors_header(model_path / file)
        entry = headers[file][name]
        lo, hi = entry["data_offsets"]
        tensors.append(HeadTensor(
            name=name,
            dtype=entry["dtype"],
            shape=tuple(entry["shape"]),
            nbytes=hi - lo,
            file=file,
        ))
    return HeadManifest(model_path=model_path, tensors=tuple(tensors))


# ======================================================================================
# CLI
# ======================================================================================

_DEFAULT_MODEL = "/home/luka/models/Qwen3.8-Flash-Next-NVFP4"


def main(argv: list[str]) -> int:
    man = head_manifest(argv[1] if len(argv) > 1 else _DEFAULT_MODEL)
    print(f"# {man.model_path}\n")
    for t in man.tensors:
        mark = "E" if t.is_expert else " "
        print(f"{mark} {t.name:<66s} {t.dtype:<8s} {str(list(t.shape)):<24s} "
              f"{t.nbytes / 2**20:10.2f} MiB  {t.file}")
    print(f"\n{len(man.tensors)} tensors   total {man.total_bytes / 2**20:.1f} MiB "
          f"({man.total_bytes} B)")
    print(f"  experts (E) {man.expert_bytes / 2**20:.1f} MiB   "
          f"dense {man.dense_bytes / 2**20:.2f} MiB")

    g = man.geometry
    print(f"\ngeometry: E={g.num_experts} H={g.hidden_size} I={g.moe_intermediate_size} "
          f"shared_I={g.shared_expert_intermediate_size} hc_count={g.hc_count} "
          f"layers={g.num_layers}")
    print("\nNVFP4 bank shapes this geometry implies:")
    for name, shape in g.nvfp4_bank_shapes().items():
        print(f"  {name:<16s} {list(shape)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
