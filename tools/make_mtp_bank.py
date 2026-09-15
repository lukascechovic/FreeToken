#!/usr/bin/env python3
"""Rebuild the MTP draft head's NVFP4 expert bank from a published checkpoint.

The `-mtp-` row needs one derived file that is not in the checkpoint and is not in the image:

    mtp-experts-nvfp4.safetensors   1,419,510,312 B

⛔ **It is deliberately not published.** It is derived from the checkpoint's own `mtp.*` tensors --
the roughly one third of the weights that `models/qwen4_exp/weight.py` drops on load -- so shipping
it as bytes would raise a redistribution question in exchange for nothing this script cannot
rebuild. Pure python and numpy, no torch, no safetensors library, about 152 s of CPU on one core.

    python3 make_mtp_bank.py /path/to/Qwen3.8-Flash-Next-NVFP4 --plan
    python3 make_mtp_bank.py /path/to/Qwen3.8-Flash-Next-NVFP4 --out /path/to/mtp-bank

Then point the row at it:  FREETOKEN_MTP_BANK=/path/to/mtp-bank/mtp-experts-nvfp4.safetensors

⛔⛆ **THE MODEL PATH IS REQUIRED AND THERE IS NO DEFAULT.** The research CLI this wraps
(`nvfp4_801.py`) defaults it to one machine's directory, which is right for a research round and
wrong for a recipe: on anybody else's machine a default does not refuse, it goes looking for
somebody else's filesystem and reports a missing file -- which reads as the reader's setup being
broken rather than the tool assuming something it had no business assuming.

⚠ `nvfp4_801.py` and `head_801.py` beside this file are DERIVED COPIES of llm-server's
`docs/research/freetoken-mtp-801/`. Regenerate them, never edit them here: two producers for one
file is how a fully-declared artefact quietly stops being reproducible.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from head_801 import head_manifest, safetensors_layout  # noqa: E402
from nvfp4_801 import (  # noqa: E402
    BANK_GRANULARITIES,
    BANK_SCHEMA,
    NVFP4_BLOCK,
    requantise_head_experts,
)

#: The head's two stacked expert tensors, as the checkpoint names them.
HEAD_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
HEAD_DOWN = "mtp.layers.0.mlp.experts.down_proj"

ARTEFACT_NAME = "mtp-experts-nvfp4.safetensors"

#: ⛔⛆ What the DEPLOYED bank was built at, and the only default this tool has.
#: `nvfp4_801.py`'s CLI sweeps all three granularities -- three 4.8 GiB reads -- and writes only the
#: last. That is a research instrument. Rebuilding one artefact must cost one pass, and must not
#: quietly hand back a granularity no measurement in this repo was taken at.
DEPLOYED_GRANULARITY = "row"


class BankError(ValueError):
    """A refusal. ⛔ Always names the path or value that did not line up."""


@dataclass(frozen=True)
class BankPlan:
    """What the write WILL produce, computed from the checkpoint's geometry alone.

    ⭐ `total_bytes` comes from `safetensors_layout` -- the writer's own header code -- rather than
    from arithmetic beside it. The header is JSON whose length depends on the tensor names, their
    shapes and the offsets themselves, so a recipe that estimated it would promise a size the tool
    does not produce.
    """

    model: Path
    granularity: str
    num_experts: int
    tensors: dict[str, tuple[str, tuple[int, ...]]]
    total_bytes: int

    def describe(self) -> str:
        lines = [
            f"# {ARTEFACT_NAME} from {self.model}",
            f"#   {self.num_experts} head experts, global scale per {self.granularity}",
            "",
        ]
        for name, (dtype, shape) in self.tensors.items():
            lines.append(f"    {name:<16s} {str(list(shape)):<22s} {dtype}")
        lines += ["", f"    total {self.total_bytes} B  ({self.total_bytes / 2**30:.3f} GiB)"]
        return "\n".join(lines)


def plan_bank(model: str | Path, granularity: str = DEPLOYED_GRANULARITY) -> BankPlan:
    """Read the head's geometry and say exactly what the artefact will be. Writes nothing."""
    if granularity not in BANK_GRANULARITIES:
        raise BankError(
            f"granularity must be one of {BANK_GRANULARITIES}, got {granularity!r}"
        )
    model = Path(model)
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        raise BankError(
            f"{model} does not look like a published checkpoint: no {index.name} in it"
        )
    manifest = head_manifest(model)
    try:
        gate_up = manifest.by_name(HEAD_GATE_UP)
        down = manifest.by_name(HEAD_DOWN)
    except Exception as exc:  # the manifest raises its own message; keep the path in it
        raise BankError(f"{model} carries no MTP head: {exc}") from exc

    num_experts = gate_up.shape[0]
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for bank, tensor in (("gate_up", gate_up), ("down", down)):
        _, rows, k = tensor.shape
        tensors[f"{bank}_packed"] = (BANK_SCHEMA[f"{bank}_packed"][0], (num_experts, rows, k // 2))
        tensors[f"{bank}_scale"] = (
            BANK_SCHEMA[f"{bank}_scale"][0], (num_experts, rows, k // NVFP4_BLOCK)
        )
        tensors[f"{bank}_global"] = (BANK_SCHEMA[f"{bank}_global"][0], (num_experts, rows))
    # ⛔ The write emits BANK_SCHEMA's order, so the plan has to as well: the header's JSON -- and
    #   therefore the file's size -- depends on the order the names go in.
    ordered = {name: tensors[name] for name in BANK_SCHEMA}
    _, _, total = safetensors_layout(ordered)
    return BankPlan(
        model=model,
        granularity=granularity,
        num_experts=num_experts,
        tensors=ordered,
        total_bytes=total,
    )


def write_bank(model: str | Path, out_dir: str | Path, granularity: str) -> BankPlan:
    """Do the work, then assert the artefact is the one the plan promised."""
    plan = plan_bank(model, granularity)
    print(plan.describe())
    print(f"\n  requantising {plan.num_experts} experts, one pass -- this takes a few minutes")
    started = time.monotonic()
    report = requantise_head_experts(head_manifest(plan.model), out_dir, granularity=granularity)
    elapsed = time.monotonic() - started

    if report.artefact_bytes != plan.total_bytes:
        raise BankError(
            f"wrote {report.artefact_bytes} B but the plan promised {plan.total_bytes} B "
            f"-- the artefact is not what this tool said it would be"
        )
    print(f"\n  wrote {report.artefact}  {report.artefact_bytes} B  [{elapsed:.0f}s]")
    print(f"\n  FREETOKEN_MTP_BANK={report.artefact}")
    return plan


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="make_mtp_bank.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ⛔ Positional and REQUIRED. See the module docstring: a default here would mean this box.
    parser.add_argument("model", help="the published checkpoint directory (no default, on purpose)")
    parser.add_argument("--out", metavar="DIR", help=f"write {ARTEFACT_NAME} into this directory")
    parser.add_argument("--plan", action="store_true",
                        help="print what would be written, read nothing but the index, write nothing")
    parser.add_argument("--granularity", default=DEPLOYED_GRANULARITY, choices=BANK_GRANULARITIES,
                        help=f"global-scale granularity (default {DEPLOYED_GRANULARITY}, "
                             "which is what the deployed bank was built at)")
    args = parser.parse_args(argv[1:])

    try:
        if args.plan:
            print(plan_bank(args.model, args.granularity).describe())
            return 0
        if not args.out:
            parser.error("--out DIR is required to write the bank (or pass --plan to look first)")
        write_bank(args.model, args.out, args.granularity)
    except BankError as exc:
        print(f"\nFATAL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
