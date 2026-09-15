# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `models/qwen4_exp/ple_disk.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 82839118e9b58e6cb56faea541ba781c,
# 272 lines) BIND-MOUNTED over the installed package, plus this block, `_decode_runs`,
# `_undrained_context` and the `.spec` import it needs, the two lines of `host_fill_batch` that
# call it, the readback width, and one assertion in `fill`.
# ⛔ In no image, in no Dockerfile ladder: `arm_mtp_801.sh`'s OVERLAY and ORIGS lists or
# nothing (#866).
#
# ⛔⛆ WHY IT EXISTS, AND WHY IT IS BULLET 9C AND NOT BULLET 8B. This is PLE's OTHER half. Bullet
#   8b fixed `ple.py` -- the compute half, the conv and the n-gram context -- and reasoned
#   carefully about capture; it never opened this file, because nothing named it. `--ple-backend
#   disk` is what the DEPLOYED ROW and `arm_mtp_801.sh` both run, so the whole of bullet 8b's
#   work reaches the box through here. The round's second load found it ~90 s in, and it is the
#   sixth silent site by count and bullet 8b's own rule turned back on bullet 8b: **a hazard
#   carried forward by name is a claim until someone opens the file** -- nobody had opened this
#   one, because no name pointed at it.
#
# ⛔⛆ THE TWO BUGS, AND THEY ARE BULLET 8C'S EXACT SHAPE. One line of the decode branch, written
#   twice (once in the flag-sync path, once in launch-gating):
#
#       runs = [torch.tensor([*_context(r.input_ids, r.device_len - 1, eos), t], ...)
#               for r, t in zip(reqs, tokens)]
#
#   ⛔⛆ AND BUG 2 HAS A CROSS-STEP HALF THAT BULLET 9C DID NOT SEE — #801 bullet 9j. Moving the
#   context back by `extend_len` fixed the read WITHIN a verify step. It does not help the step
#   AFTER one: `commit_verify` advances `cached_len` by TWO on an acceptance while the drain is a
#   whole iteration behind, so the next fill asks for an id `req.input_ids` does not hold yet and
#   load 5 died on exactly that, ~63 steps in. `_undrained_context` + `spec.host_token_id` close
#   it, and the id they need is host-known because it is the accepted DRAFT.
#
#   1. `tokens` is the per-ROW device readback and `reqs` is per REQUEST, so `zip` TRUNCATES.
#      At bs=1 T=2 it would take the committed token's row and drop the draft's entirely -- the
#      forward then hashes one window where it has two token rows, and `lookup` copies the second
#      row out of whatever the pinned staging happened to hold.
#   2. `r.device_len - 1` is the position the NEW token sits at only when the step forwards ONE
#      token. On a verify step `device_len == cached_len + 2`, so it reads `input_ids[cached_len]`
#      -- one PAST the last host-known id, because the host ids only catch up at the drain
#      (`scheduler.overlap_loop` runs `_forward` before `_process_last_data`). That raises
#      `IndexError: index 66 is out of bounds for dimension 0 with size 66`, which is how the bug
#      announced itself rather than serving quietly.
#
# ⭐ THE FIX IS THE SHAPE `fill` ALREADY WANTED. `fill` stages `run.numel() - 2` hash rows per run
#   -- a run is `[ctx0, ctx1, tok...]` and the C++ store slides the window itself -- and the
#   PREFILL branch in this same file already hands it exactly that: a context taken at
#   `req.cached_len` followed by EVERY new token. So a verify step wants ONE run per REQUEST
#   carrying BOTH tokens, not one run per token, with the context taken `n` positions back.
#   ⭐⭐ At `n == 1` that is byte-identically today's arithmetic, which is the check that the
#   plain decode path -- every row this box serves -- is untouched;
#   `test_ple_verify_801.py::TestAPlainDecodeStepDidNotMove` proves it as a DIFFERENTIAL against
#   `ple_disk.py.orig` rather than asserting it.
#
# ⭐ THE BUFFERS ARE NOT WIDENED, DELIBERATELY. `max_graph_rows` is
#   `max(256, cuda_graph_max_bs or 0)` (`model.py`) against the EIGHT rows a bs=4 T=2 step needs,
#   so `_graph_pinned`, `_graph_dev` and `_token_readback` all already hold a verify step. ⛔ The
#   `fill` assertion below is the LOCK on that reasoning, not a second copy of it: it cannot fire
#   while `mtp_verify_graph_max_bs` stays at 4, and it fires loudly instead of writing past a
#   pinned allocation if a later change raises either dial.
#
# ⛔ `device_len` is READ here and never written: the staging that advanced it is
#   `spec.stage_verify`, and `Req.extend_len` is a read-only property over the two fields.

"""Disk-backed PLE table (--ple-backend disk): the C++ store hashes n-gram windows and batch-reads rows from the checkpoint's fp8 shard tensors into pinned staging; the captured ``lookup`` is a fixed-shape H2D copy + dequant.

Hash windows are pure functions of ``req.input_ids`` + ``device_len`` (prefix hits, restores and COW forks need no bookkeeping); the decode input token lives device-side under overlap scheduling and is read back here.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import safetensors
import torch

from freetoken.core import Batch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

from .spec import UNDRAINED_IDS, host_token_id
from .weight import (
    _PLE_SCALE_SUFFIX,
    _PLE_SHARD_RE,
    _PLE_ST_DTYPE,
    _ple_table_files,
    _safetensors_header,
)

_IO_URING_ENV = "FREETOKEN_PLE_IO_URING"
_SYNC_ENV = "FREETOKEN_PLE_SYNC"  # auto | wait | gate

logger = init_logger(__name__)


def _context(ids: torch.Tensor, position: int, eos: int) -> list[int]:
    """The two token ids before ``position``; eos pads past the start."""
    return [int(ids[position - 2]) if position >= 2 else eos,
            int(ids[position - 1]) if position >= 1 else eos]


def _undrained_context(req: "Req", position: int, eos: int) -> list[int]:
    """`_context`, over the ids the host knows INCLUDING the ones the drain still owes.

    ⛔⛆ #801 bullet 9j. `req.input_ids` does not see the batch still in flight — `overlap_loop`
    runs `_forward` before `_process_last_data` — and at one token per step that leaves this read
    sitting EXACTLY on the last host-known id. An ACCEPTED verify step commits two, and the next
    fill wants one the drain has not shipped: load 5 died here with
    `IndexError: index 123 is out of bounds for dimension 0 with size 123`, on both ranks.
    `spec.host_token_id` resolves it from what `commit_verify` recorded.

    ⛔ It RAISES rather than padding with ``eos`` when an id is genuinely absent. Padding is the
    cheap move and it would turn a loud host error into a wrong hash window — the store would
    hash a run the model never saw, on every accepted step, in silence. ⚠ ``eos`` still pads the
    START of a sequence, which is `_context`'s own rule and is not a missing id.
    """
    out: list[int] = []
    for back in (2, 1):
        at = position - back
        if at < 0:
            out.append(eos)
            continue
        token = host_token_id(req, at)
        if token is None:
            raise RuntimeError(
                f"#801: req {req.uid} has no host id at position {at} — input_ids holds "
                f"{req.input_ids.numel()} and the undrained record holds "
                f"{sorted(getattr(req, UNDRAINED_IDS, {}))}; cached_len={req.cached_len}, "
                f"device_len={req.device_len}"
            )
        out.append(token)
    return out


def _decode_runs(reqs: Sequence["Req"], tokens: Sequence[int], eos: int) -> list[torch.Tensor]:
    """#801 bullet 9c: this decode forward's hash runs — ONE per request, carrying every token
    that request is forwarding, in the batch's own flat row order.

    ``tokens`` is the per-ROW readback of ``batch.input_ids``; ``reqs`` is per REQUEST. A plain
    decode step is one row each and this is the image's own list comprehension spelled as a loop;
    a verify step is two rows for a request whose draft is being checked and one for a request
    beside it whose draft was not produced, and the two cannot be ``zip``ped.

    ⛔ The context is taken ``n`` positions back — ``device_len - n`` IS ``cached_len``, the
    position this request's FIRST new token sits at — because the run is a window, not a token:
    `fill` hands the store ``numel() - 2`` rows and the store slides ``(ctx0, ctx1, tok0)``,
    ``(ctx1, tok0, tok1)`` itself. Taken one back regardless of width, the draft row would hash
    the committed token's window and the first row would index past the host ids.

    ⛔ The ids come through `_undrained_context`, not `req.input_ids` directly: under overlap the
    host list is missing the in-flight batch's tokens, and after an ACCEPTED step the one it is
    missing is one this read needs (#801 bullet 9j).

    ⚠ ``reqs`` is the UNPADDED list on purpose: a padded decode lane stages nothing and reads the
    zeroed staging, which is what `DiskRowTable.__init__` allocates it for. Padding is appended
    after the real requests, so walking ``reqs`` consumes the readback's leading rows.
    """
    runs: list[torch.Tensor] = []
    offset = 0
    for req in reqs:
        count = req.extend_len
        runs.append(torch.tensor(
            [*_undrained_context(req, req.device_len - count, eos),
             *tokens[offset : offset + count]],
            dtype=torch.int64,
        ))
        offset += count
    return runs


@dataclass(frozen=True)
class PleRowSource:
    """On-disk row layout: equal extents, row i of an extent at ``base + i * row_stride`` (a repacked flat file is one extent with its own stride)."""

    paths: list[str]
    extent_file: list[int]
    extent_base: list[int]
    rows_per_extent: int
    row_bytes: int
    row_stride: int
    scale: float

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent


def source_from_safetensors(folder: str) -> PleRowSource:
    """Map the checkpoint's ``ngram_embedding.shard_<i>`` tensors in place: one extent per shard, no copy."""
    rows = cols = 0
    scale: torch.Tensor | None = None
    paths: list[str] = []
    path_idx: dict[str, int] = {}
    shards: dict[int, tuple[int, int]] = {}
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE shard {key} has dtype {meta['dtype']}, expected {_PLE_ST_DTYPE}")
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            if path not in path_idx:
                path_idx[path] = len(paths)
                paths.append(path)
            idx = int(match.group("shard"))
            if idx in shards:
                raise ValueError(f"duplicate PLE shard {idx} in {path}")
            shards[idx] = (path_idx[path], base + meta["data_offsets"][0])
    if sorted(shards) != list(range(len(shards))) or not shards:
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")
    order = [shards[i] for i in range(len(shards))]
    return PleRowSource(paths, [f for f, _ in order], [b for _, b in order], rows, cols, cols, float(scale))


def resolve_row_source(folder: str) -> PleRowSource:
    """Pick the row source for a checkpoint; the seam where a repacked format would plug in."""
    return source_from_safetensors(folder)


class DiskRowTable:
    """``PLETableBackend`` whose rows are read from disk per fill (--ple-backend disk)."""
    # every rank stages all hash heads from the host store; NGramEmbedding.lookup must not all-gather
    serves_all_heads = True

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from freetoken.kernel import _ple_store

        self.num_rows = source.total_rows
        self.head_dim = source.row_bytes  # fp8: one byte per element
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._store = _ple_store.PleStore(
            paths=list(source.paths),
            extent_file=list(source.extent_file),
            extent_base=list(source.extent_base),
            rows_per_extent=source.rows_per_extent,
            row_bytes=source.row_bytes,
            row_stride=source.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv(_IO_URING_ENV, "1") != "0",
        )
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._token_bytes = self.heads * self.head_dim
        # allocated up front: pinned alloc inside stream capture is illegal; one replay consumes it at a time
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * self._token_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()  # padded decode lanes read whatever sits here
        # outlives any one graph: a cache rebuild recaptures against the same pointer
        self._graph_dev = torch.empty(
            max_graph_rows * self._token_bytes, dtype=torch.uint8, device=self._device
        )
        eager_bytes = max_extend_tokens * self._token_bytes
        self._eager_pinned = alloc_pinned_tensor(eager_bytes, dtype=torch.uint8)
        self._eager_pinned.zero_()  # the warmup prefill stages nothing and reads whatever sits here
        self._eager_dev = torch.empty(eager_bytes, dtype=torch.uint8, device=self._device)
        # probe picks flag-sync (graph WAITs at the consume, host fills then signals) or launch-gating
        self._wait_sync = self._probe_wait_sync(os.getenv(_SYNC_ENV, "auto"))
        # one flag for all graphs: the readback event orders a fill after the previous graph, so signals never overlap
        self._flag = alloc_pinned_tensor(1, dtype=torch.int64)
        self._flag.zero_()
        self._token_readback = alloc_pinned_tensor(max_graph_rows, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        sync = "wait-sync" if self._wait_sync else "launch-gating"
        logger.info_rank0(f"PLE disk backend: {self._store.io_backend()}, {sync}")

    def _probe_wait_sync(self, mode: str) -> bool:
        from freetoken.kernel import _ple_store

        if mode == "gate":
            return False
        scratch = alloc_pinned_tensor(1, dtype=torch.int64)
        scratch.zero_()
        stream = torch.cuda.current_stream(self._device)
        ok = (
            _ple_store.memop_write(stream.cuda_stream, scratch.data_ptr(), 7) == 0
            and _ple_store.memop_wait_geq(stream.cuda_stream, scratch.data_ptr(), 7) == 0
        )
        if ok:
            stream.synchronize()
            ok = int(scratch[0]) == 7
        if mode == "wait" and not ok:
            raise RuntimeError("FREETOKEN_PLE_SYNC=wait but stream memops are unavailable")
        return ok

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order."""
        pinned = self._graph_pinned if graph else self._eager_pinned
        # ⛔ #801 bullet 9c: a verify step stages `sum(tokens_per_req)` hash rows, not `bs` of
        #   them, and `stage` writes straight into the pinned allocation. Derived from the buffer
        #   rather than from a remembered constant so it cannot go stale; see the module header
        #   for why it cannot fire at today's dials.
        rows = sum(int(run.numel()) - 2 for run in runs)
        capacity = pinned.numel() // self._token_bytes
        assert rows <= capacity, (
            f"#801: this step stages {rows} PLE hash rows into a "
            f"{'graph' if graph else 'eager'} buffer sized for {capacity}"
        )
        offset = 0
        for run in runs:
            self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
            offset += run.numel() - 2
        self._store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def host_fill_batch(self, batch: Batch, use_graph: bool):
        """Stage this batch's rows; returns the post-dispatch fill callable under flag-sync, else None."""
        eos = self.eos_token_id
        if batch.is_decode:
            reqs = list(batch.reqs)
            if use_graph and self._wait_sync:
                # ⛔ #801 bullet 9c: ROWS, not requests. `batch.input_ids` is one entry per
                #   forwarded TOKEN (`scheduler._make_input_tuple` walks `Req.extend_len`), so
                #   `padded_size` here sized the readback for half of a T=2 step and `copy_`
                #   raised on the shape. Same arithmetic as `_make_positions`.
                rows = sum(r.extend_len for r in batch.padded_reqs)
                self._token_readback[:rows].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self._device))

                def _complete() -> None:
                    try:
                        self._readback_event.synchronize()
                        tokens = self._token_readback[:rows].to(torch.int64).tolist()
                        self.fill(_decode_runs(reqs, tokens, eos), graph=True)
                    except BaseException:
                        from freetoken.kernel import _ple_store

                        # unblock the stream before surfacing; the step's output is discarded
                        _ple_store.signal_flag(self._flag.data_ptr())
                        raise

                return _complete
            # launch-gating: this D2H is the step's readback and orders the fill after sampling
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            self.fill(_decode_runs(reqs, tokens, eos), graph=use_graph)
            return None
        runs = [
            torch.cat((
                torch.tensor(_context(req.input_ids, req.cached_len, eos), dtype=torch.int64),
                req.input_ids[req.cached_len : req.device_len].to(torch.int64),
            ))
            for req in batch.padded_reqs
        ]
        self.fill(runs, graph=False)
        return None

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage on enter, run the deferred fill+signal on exit."""
        deferred = self.host_fill_batch(batch, use_graph)
        yield
        # no try/finally: a failed launch leaves no WAIT pending, so the fill must not run
        if deferred is not None:
            deferred()

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = row_ids.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel import _ple_store

            _ple_store.memop_wait_reset(
                torch.cuda.current_stream(self._device).cuda_stream, self._flag.data_ptr()
            )
        pinned, dev = (
            (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned, self._eager_dev)
        )
        nbytes = rows * self._token_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        values = dev[:nbytes].view(torch.float8_e4m3fn).to(self.dtype)
        if self.scale != 1.0:
            values = values * self.scale
        values = values.view(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None
