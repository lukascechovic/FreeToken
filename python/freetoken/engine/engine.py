from __future__ import annotations

import gc
import math
import os
from datetime import timedelta
from typing import Any, Dict, Iterable, NamedTuple, Tuple

import torch
from freetoken.attention import AttnType, attention_backend_info, create_attention_backend
from freetoken.core import Batch, Context, Req, set_global_ctx
from freetoken.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from freetoken.gpu_select import gpu_identity
from freetoken.layers import set_rope_device
from freetoken.models import create_model, load_weight
from freetoken.moe import create_moe_backend, is_offload_moe_backend
from freetoken.moe.expert_banks import load_expert_banks
from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache
from freetoken.utils import align_ceil, init_logger, is_sm90_family, is_sm100_family, mem_GB, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, _determine_cuda_graph_bs, get_free_memory
from .sample import BatchSamplingArgs, Sampler
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.kvcache.cache_status import _supports_swa_ratio
from freetoken.kvcache.linear_state_pool import (
    _linear_pool_min_slots, _linear_pool_num_slots, state_pool_bytes,
)

logger = init_logger(__name__)

# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `engine/engine.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 02902f4aa338362de63767c0523dbfd4,
# 1524 lines) BIND-MOUNTED over the installed package, plus this block and the draft-head pool
# sizing marked `#801 bullet 7` below. ⛔ It is NOT a patch in the Dockerfile ladder and is in NO
# image. ⚠ A `-v` that silently does not take leaves the row running the IMAGE's engine while
# every log line looks like the arm we think we launched (#866) -- hence a marker, on stderr,
# before the engine's logging is up.
import json as _ft801_json
import sys as _ft801_sys

print(
    "[#801] overlay ACTIVE: engine/engine.py bind-mounted from the repo "
    f"(pid {os.getpid()}, base md5 02902f4aa338362de63767c0523dbfd4, "
    f"FREETOKEN_LOAD_MTP={os.getenv('FREETOKEN_LOAD_MTP', '<unset>')}, "
    f"FREETOKEN_MTP801_RUN_HEAD={os.getenv('FREETOKEN_MTP801_RUN_HEAD', '<unset>')})",
    file=_ft801_sys.stderr,
    flush=True,
)


def _require_offload_cache_size(cache_size: int, num_experts: int) -> None:
    """The offload MoE cache needs at least one slot per expert per layer. A too-small size
    (e.g. a bare offload run with moe_cache_size unset and auto disabled) must fail loudly."""
    if cache_size < num_experts:
        raise ValueError(
            f"moe_cache_size={cache_size} is too small: need at least num_experts={num_experts} "
            f"slots. Pass --moe-cache-size/--moe-cache-rate, or use --moe-cache-auto "
            f"(the default for offload/hybrid backends when no cache-sizing flag is given; "
            f"--moe-backend cpu always sizes its own fixed two-layer buffer and ignores "
            f"cache-sizing flags)."
        )


def _flashinfer_available() -> bool:
    from freetoken.kernel.backend import is_flashinfer_installed

    return is_flashinfer_installed()


def _sgl_flash_attn_available() -> bool:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
        logger.warning_rank0(
            "sgl_kernel.flash_attn is unavailable; auto attention backend falls back to fi "
            f"({type(exc).__name__}: {detail})"
        )
        return False
    return True


def _startup_kv_budget(memory_ratio: float, init_free_memory: int, new_free_memory: int) -> int:
    """Bytes available to the KV pool at startup: ratio-scaled pre-load free memory minus
    what the resident model consumed. Kept as a pure function so the composition with the
    pool families' ``solve_num_pages`` stays CPU-testable."""
    return int(memory_ratio * init_free_memory) - (init_free_memory - new_free_memory)


def _page_table_width(max_seq_len: int, page_size: int) -> int:
    """Column count for the page table. ``_write_page_table`` writes WHOLE trailing pages, so the
    highest column touched is ``align_ceil(max_seq_len, page_size) - 1`` -- which the 32-alignment
    alone does not cover once page_size > 32 (an unaligned --max-seq-len-override on DSV4's P=128
    or trtllm's forced 64 would index past the row)."""
    return align_ceil(align_ceil(max_seq_len, page_size), 32)


def _required_attn_types(model_config) -> frozenset[AttnType]:
    """Backend-driving attention types of this model, from the group-spec walk
    (single source shared with the pool factory and the KV cost model). getattr
    fallbacks: duck-typed test configs may not implement the spec walk; for those,
    dsv4_args marks DSV4 (the real config declares a DSV4 attention group)."""
    specs_fn = getattr(model_config, "kv_cache_group_specs", None)
    if specs_fn is None:
        if getattr(model_config, "dsv4_args", None) is not None:
            return frozenset({AttnType.DSV4})
        return frozenset({AttnType.FULL})
    types = frozenset(
        spec.attn_type for spec in specs_fn() if spec.attn_type.backend_driven
    )
    return types or frozenset({AttnType.FULL})


def _backend_parts_serve(name: str, required: frozenset[AttnType]) -> bool:
    return all(
        required <= attention_backend_info(part).supported_types
        for part in name.split(",")
    )


def _backend_requirements_met(name: str) -> bool:
    # flashinfer first across ALL parts: the sgl probe logs a "falls back to fi" warning,
    # which would mislead when the candidate is about to fail on flashinfer anyway.
    infos = [attention_backend_info(part) for part in name.split(",")]
    if any(i.requires_flashinfer for i in infos) and not _flashinfer_available():
        return False
    if any(i.requires_sgl_kernel for i in infos) and not _sgl_flash_attn_available():
        return False
    if any(i.requires_sm100 for i in infos) and not is_sm100_family():
        return False
    return True


def _resolve_auto_attention_backend(required: frozenset[AttnType]) -> str:
    """First candidate (in per-type priority order) whose arch condition holds,
    whose packages are installed, and whose every comma part serves ALL required
    types. Reproduces the historical hardware tree for FULL-only models:
    sm_100 -> trtllm, sm_90+sgl_kernel -> "fa,fi", flashinfer -> fi, else triton."""
    candidates: list[tuple[str, bool]] = []
    if AttnType.DSV4 in required:
        candidates.append(("dsv4_sparse", True))
    if required & {AttnType.MLA, AttnType.DSA}:
        candidates.append(("dsa", True))
    if AttnType.BSA in required:
        candidates.append(("m3_sparse", True))
    if AttnType.QSA in required:
        candidates.append(("qsa_sparse", True))
    if AttnType.SWA in required:
        candidates.append(("triton", True))
    if AttnType.FULL in required:
        candidates += [
            ("trtllm", is_sm100_family()),
            ("fa,fi", is_sm90_family()),
            ("fi", True),
            ("triton", True),
        ]
    for name, arch_ok in candidates:
        if not arch_ok:
            continue
        if not _backend_parts_serve(name, required):
            continue
        if not _backend_requirements_met(name):
            continue
        return name
    raise RuntimeError(
        "No attention backend can serve attention types "
        f"{sorted(t.value for t in required)} on this machine."
    )


def _validate_attention_backend_choice(config, override, required: frozenset[AttnType]) -> None:
    """Config-time type x backend capability check for the resolved (or explicit)
    backend string: every comma part must serve every required type and have its
    packages/arch available. Replaces the per-model gates; in particular this is
    where a DSV4 or MLA checkpoint rejects a generic backend before weights load,
    and where a generic model rejects dsa/dsv4_sparse."""
    from freetoken.attention import validate_attn_backend

    # Name membership first (ArgumentTypeError listing the supported names): the CLI already
    # ran this, but the programmatic EngineConfig path reaches here unvalidated and would
    # otherwise die on a bare KeyError from the info lookup below.
    validate_attn_backend(config.attention_backend, allow_auto=False)

    model_config = config.model_config
    backend_parts = [p.strip() for p in config.attention_backend.split(",")]
    for part in backend_parts:
        info = attention_backend_info(part)
        missing = required - info.supported_types
        if missing:
            valid = [
                name
                for name in (
                    "fa", "fi", "trtllm", "triton", "dsa", "dsv4_sparse", "m3_sparse",
                    "qsa_sparse",
                )
                if required <= attention_backend_info(name).supported_types
            ]
            missing_names = "/".join(sorted(t.value for t in missing))
            raise ValueError(
                f"{getattr(model_config, 'model_type', 'model')} uses {missing_names} "
                f"attention, which backend {part!r} does not support; valid backends: "
                f"{', '.join(valid)} (or auto), got {config.attention_backend!r}."
            )
        if AttnType.SWA in required and not info.consumes_attn_spec:
            # SWA models drive window/sinks/sm_scale through the per-call AttentionSpec;
            # a backend that drops it would attend with the wrong window silently.
            raise ValueError(
                f"backend {part!r} does not consume the per-call AttentionSpec that "
                f"SWA models require, got {config.attention_backend!r}."
            )

    # An explicitly-selected backend may require a package that isn't installed. Auto
    # never resolves to one of these when its package is missing, so this only fires for
    # explicit --attention-backend choices.
    for part in backend_parts:
        info = attention_backend_info(part)
        if info.requires_flashinfer and not _flashinfer_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires flashinfer, which is "
                "not installed. Install it with `pip install 'freetoken[fi]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sgl_kernel and not _sgl_flash_attn_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires sgl_kernel, which is "
                "not installed. Install it with `pip install 'freetoken[sgl]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sm100 and not is_sm100_family():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires a compute capability "
                "10.x GPU: flashinfer's trtllm-gen kernels ship sm_100a/103a cubins only. "
                "Use --attention-backend fi (or triton) instead."
            )

    if required & {AttnType.MLA, AttnType.DSA} and config.page_size != 1:
        # The MLA backend's row addressing (latent scatter, DSA index keys, sparse
        # top-k page indices) assumes page_size == 1 throughout; reject explicitly
        # like the SWA models do rather than corrupting addressing silently.
        raise ValueError(
            f"latent-KV MLA models require --page-size 1, got {config.page_size}."
        )

    for part in backend_parts:
        info = attention_backend_info(part)
        if info.page_sizes is not None and config.page_size not in info.page_sizes:
            override("page_size", info.page_sizes[-1])
            logger.warning_rank0(
                f"Page size is overridden to {info.page_sizes[-1]} for the {part} backend"
            )


def _make_dummy_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    for key, param in model_state.items():
        if param.dtype in fp8_dtypes:
            # torch.randn is not implemented for fp8; fill via a uint8 view with small
            # codes (avoid NaN/inf fp8 encodings). Lets dummy-weight startup work for
            # block-fp8 models (the dense fp8 linears are fp8 regardless of moe_backend).
            t = torch.empty(param.shape, dtype=param.dtype, device=device)
            t.view(torch.uint8).random_(0, 16)
            state_dict[key] = t
        elif param.dtype.is_floating_point or param.dtype.is_complex:
            state_dict[key] = torch.randn(param.shape, dtype=param.dtype, device=device)
        elif param.dtype == torch.uint8 and key.endswith("weight_scale_inv"):
            # MXFP8 e8m0 exponent codes: 127 encodes scale 1.0; zeros would collapse
            # every scale to 2^-127 and zero the model. Scoped BY NAME: other uint8
            # buffers are packed payloads whose bytes mean something else entirely
            # (GGUF qweight blocks embed fp16 scales -- 0x7F7F is fp16 NaN), so they
            # keep the benign all-zeros fill below.
            state_dict[key] = torch.full(param.shape, 127, dtype=param.dtype, device=device)
        else:
            state_dict[key] = torch.zeros(param.shape, dtype=param.dtype, device=device)
    return state_dict


def _materialize_loaded_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for key, weight in weights:
        expected = model_state.get(key)
        if expected is None:
            state_dict[key] = weight.to(device=device)
        else:
            state_dict[key] = weight.to(device=device, dtype=expected.dtype)
    return state_dict


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _ensure_expandable_segments()  # before the first CUDA allocation below

        from freetoken.gpu_select import bind_assigned_gpu

        self.device = bind_assigned_gpu(config.tp_info.rank)
        _adjust_config(config)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config  # retained for runtime cache rebuild (rebuild_runtime_cache)
        # KV pool family fixed at construction from the model config: its classmethods own the
        # page-token geometry and cost arithmetic the engine needs BEFORE the pool exists
        # (num_pages sizing, --moe-cache-auto); the instance owns rebuild/validation after.
        self._pool_cls = resolve_pool_class(config.model_config)
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        free_min, free_max = self._sync_get_memory()
        init_free_memory = free_max  # startup KV sizing keeps cross-rank MAX (unchanged)
        self._baseline_free = free_min  # rebuild baseline: cross-rank MIN, deterministic across ranks
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        # ── #801 bullet 7: the draft head's three pools ──────────────────────────────────
        # A model that built a draft head answers `mtp_pool_model_config` with the config the
        # engine's POOLS must be sized from -- the head's layer id owned by an attention group,
        # one more index layer, and (nvfp4 only) one more MoE layer. Every other model has no such
        # method and nothing below changes. ⛔ It is applied HERE, after `create_model` and after
        # `_load_weight_state_dict`: both build/validate against `num_layers` and would try to
        # make a 49th BACKBONE layer out of a checkpoint that has 48.
        # ⛔ The backbone's own config is kept, because the expert-bank loader still reads the
        #   checkpoint's 48 MoE layers; the head's experts come from its own bank (below).
        # ⛔⛆ #801 round 5 bullet 7a: TWO deltas, not one, because they come apart. The head's
        #   ATTENTION layer exists under either head dtype (`_mtp_pool_layers`), but its MoE layer
        #   is an `OffloadMoELayer` only under the deployed nvfp4 dial (`_mtp_bank_layers`); under
        #   round 5 bullet 1's resident-bf16 research dial the head holds its experts as plain
        #   on-card tensors and there is no bank layer to attach, no cache slot to budget and no
        #   49th layer for `attach_offload_moe_cache`'s walk to find. Using the LAYER delta to gate
        #   the BANK work -- which is what this did before 7a -- appends an NVFP4 bank onto a bf16
        #   head and then dies on that walk's assertion, ~20 min into a load.
        self._backbone_model_config = config.model_config
        self._mtp_pool_layers = 0
        self._mtp_bank_layers = 0
        if hasattr(self.model, "mtp_pool_model_config"):
            pooled = self.model.mtp_pool_model_config(config.model_config)
            self._mtp_pool_layers = pooled.num_layers - config.model_config.num_layers
            self._mtp_bank_layers = pooled.num_moe_layers - config.model_config.num_moe_layers
            if self._mtp_pool_layers:
                logger.info_rank0(
                    f"#801: sizing the pools for {self._mtp_pool_layers} draft layer(s) "
                    f"({self._mtp_bank_layers} of them offload-MoE): "
                    f"num_layers {config.model_config.num_layers} -> {pooled.num_layers}, "
                    f"num_moe_layers {config.model_config.num_moe_layers} -> "
                    f"{pooled.num_moe_layers}"
                )
                object.__setattr__(config, "model_config", pooled)
        post_weights_free = self._sync_get_memory()[0]
        self._weights_bytes = self._baseline_free - post_weights_free
        # Pool-budget baseline for the desktop cache sliders: free VRAM after the weights are
        # resident but before ANY runtime cache pool (MoE expert cache below, KV pages, GDN
        # state) is allocated. This is the stable "if all free VRAM went to one pool" budget —
        # unlike a query-time mem_get_info it doesn't drift with allocator caching, CUDA
        # graphs, or other processes. Cross-rank MIN, deterministic across ranks.
        self._post_weights_free = post_weights_free
        self.moe_offload_cache = None
        self.cpu_moe_executor = None
        # Host-side auxiliary stores (qwen4_exp's pinned PLE table): after the weights so a
        # load failure is not masked, before the MoE offload cache so the bank residency
        # planning sees the pin quota the table already spent.
        self._host_tables_bytes = 0
        if hasattr(self.model, "load_host_tables"):
            self._host_tables_bytes = int(self.model.load_host_tables(config) or 0)
        if is_offload_moe_backend(config.moe_backend):
            self._init_offload_moe_cache(config)
        if hasattr(self.model, "prepare_for_runtime"):
            self.model.prepare_for_runtime()

        # ======================= KV cache initialization ========================
        new_free = self._sync_get_memory()[1]
        # The engine measures the budget and settles the sibling GDN state pool's bytes
        # off it; the KV pool family owns every geometry-specific formula behind the rest.
        available_memory = _startup_kv_budget(config.memory_ratio, init_free_memory, new_free)
        available_memory -= state_pool_bytes(config)
        self.num_pages = self._pool_cls.solve_num_pages(config, available_memory)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kv_pool(
            config, self.num_pages, device=self.device, dtype=self.dtype
        )

        # ======================= Linear (GatedDeltaNet) state initialization ========================
        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            self.linear_state_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_num_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_info.size,
                slot_states=config.model_config.slot_states,
            )
            self.ctx.linear_state_pool = self.linear_state_pool
        else:
            self.linear_state_pool = None

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # Pools routed by the shared table but deriving reads through their own mappings (DSV4)
        # re-point here (and again on any table realloc). The graph-input snapshot that reads
        # through them belongs to the attention backend, built later in init_capture_graph.
        self.kv_cache.attach_page_table(self.page_table)

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # padded/dummy rows index the GDN padding slot (0) so gather/scatter hits scratch.
        if self.linear_state_pool is not None:
            self.dummy_req.linear_slot_idx = self.linear_state_pool.padding_slot
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        if hasattr(self.model, "mtp_reserve_graph_buffers"):
            # ── #801 bullet 7: BEFORE any capture ───────────────────────────────────────────
            # A captured decode graph writes into the addresses it was captured with, and an
            # allocation made DURING `torch.cuda.graph(...)` comes out of the graph's private
            # pool -- right on the eager warm-up, wrong on every replay. `_capture_graphs` happens
            # to visit the LARGEST batch size first, which would make lazy allocation work by
            # accident; this is the line that says so instead. ⛔ The same resolver the runner
            # itself calls, on the same arguments, so the two cannot disagree -- and the assert
            # after construction is what proves they did not.
            _mtp_graph_bs = _determine_cuda_graph_bs(
                cuda_graph_bs=config.cuda_graph_bs,
                cuda_graph_max_bs=config.cuda_graph_max_bs,
                free_memory=init_free_memory,
            )
            # decode taps one row per PADDED request, prefill one per request (+ the dummy).
            _mtp_max_rows = max(max(_mtp_graph_bs, default=0), config.max_running_req + 1)
            # ⛔⛆ #801 r6 b9cf: A SECOND CURRENCY, AND 9cb SPENT THE FIRST ONE FOR IT. `_mtp_max_rows`
            #   is a REQUEST count -- the TAP's unit, because the tap takes one row per request.
            #   HIDDENCHECK's buffer is indexed by TOKEN ROW, and a prefill chunk is the
            #   scheduler's whole token budget (`prefill_budget = min(max_extend_tokens,
            #   prefill_chunk_budget)`, so `max_extend_tokens` is its upper bound): 4096 rows
            #   against a `_mtp_max_rows` of ~160. Sized by rows, layer 0's write raised on BOTH
            #   ranks at the first prefill, before anything served.
            # ⭐ The VERIFY width is NOT applied here -- `engine.py` is model-agnostic and the
            #   model knows its own `mtp_verify_width`. This passes the budget the engine owns.
            _mtp_max_tokens = int(getattr(config, "max_extend_tokens", 0) or 0)
            self.model.mtp_reserve_graph_buffers(_mtp_max_rows, _mtp_max_tokens)
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )
        if hasattr(self.model, "mtp_reserve_graph_buffers"):
            # #801 bullet 7: the reserve above must have covered what the runner actually captured.
            assert self.graph_runner.max_graph_bs <= _mtp_max_rows, (
                f"#801: graphs captured up to bs {self.graph_runner.max_graph_bs} but the draft "
                f"tap's buffer was reserved for {_mtp_max_rows} rows -- the buffer grew INSIDE a "
                f"capture and every replay writes into the graph's private pool"
            )
        if hasattr(self.model, "mtp_set_rank_group"):
            # ── #801 round 6 bullet 7: the cross-rank verify check's own group ───────────────
            # `tp_cpu_group` is GLOO on both branches of `_init_communication` and is the only
            # group here that is. ⛔ The model must not look one up for itself: on the deployed
            # arm (`--disable-pynccl`) `torch.distributed.group.WORLD` is an NCCL group, so a
            # default would be right on one configuration and silently wrong on the one #801
            # round 6 measures — and an agreement that rides the device path hangs exactly when
            # the ranks are out of step, which is the failure it exists to replace.
            # ⚠ A no-op unless `FREETOKEN_MTP801_SPECCHECK` is non-zero.
            self.model.mtp_set_rank_group(
                self.tp_cpu_group, config.tp_info.size, config.tp_info.rank
            )
        if hasattr(self.model, "mtp_capture_draft_graph"):
            # ── #801 round 5 bullet 6: `t_draft`, captured ───────────────────────────────────
            # AFTER the backbone's own decode graphs (just above) are captured and warmed --
            # capturing here, not lazily on the first real shadow step, keeps bullet 5's
            # `t_draft` EAGER timing clean (no first-call capture tax hiding inside a measured
            # step) and matches `reserve_multi_stream`'s own ordering rule: nothing may allocate
            # during a capture, and this hook's own graph is one more thing that must not.
            self.model.mtp_capture_draft_graph(
                self.attn_backend, self.stream, self.graph_runner.dummy_req
            )
        if config.attention_backend.split(",")[0] == "triton":
            # Prefill runs on the first comma part; warm its autotune cache.
            self._warmup_prefill()

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        model_state = self.model.state_dict()
        if config.use_dummy_weight:
            return _make_dummy_weight_state_dict(model_state, device=self.device)
        # _materialize casts each loaded tensor to its model-param dtype (model_state), so
        # models declaring per-tensor dtypes (e.g. DSV4's mixed fp8/fp32/bf16) are preserved;
        # offload models exclude experts (served from the offload cache, not dense weights).
        return _materialize_loaded_weight_state_dict(
            model_state,
            load_weight(
                config.model_path,
                self.device,
                include_moe_experts=not is_offload_moe_backend(config.moe_backend),
            ),
            device=self.device,
        )

    def _resolve_auto_moe_cache_size(self, config: EngineConfig, banks) -> tuple[int, int, bool]:
        """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

        Pure glue over the Phase-1 budget policy; isolated here so it is unit-testable
        without a GPU. Reused by the Phase-2 runtime rebuild.
        """
        from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)  # sibling GDN state pool, engine-summed
        num_experts = config.model_config.num_experts
        total_experts = config.model_config.num_moe_layers * num_experts
        return resolve_moe_cache_auto(
            baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes,
            memory_ratio=config.memory_ratio,
            cache_per_page=cache_per_page,
            fixed_cache_size=fixed_cache_size,
            per_expert_bytes=expert_bytes_per_slot(banks.sources),
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=config.moe_prefill_overlap,
            kv_reserve_tokens=max(config.kv_reserve_tokens, min_reserve),
            page_size=page_tokens,
            quant_format=banks.quant_format,
        )

    def _init_offload_moe_cache(self, config: EngineConfig) -> OffloadMoeCache:
        # A model may fully own cache construction via make_offload_moe_cache.
        # Otherwise load_expert_banks gives the model module a setup hook first, then
        # falls back to per-quant providers, and the engine wires the banks into cache.
        cache_factory = getattr(self.model, "make_offload_moe_cache", None)
        if cache_factory is not None and config.moe_cache_auto:
            raise ValueError(
                "--moe-cache-auto is not supported for models with a custom "
                "make_offload_moe_cache; pass --moe-cache-size explicitly."
            )
        # decode_target picks the bank layout + the per-decode mechanism:
        #   "hybrid" -> GPU-cache + CPU-overflow co-compute, every layer (--moe-backend hybrid);
        #   "cpu"    -> CPU executor for the cpu_layer_ids set (all layers under --moe-backend
        #               cpu, the --moe-cpu-layers subset under offload);
        #   "gpu"    -> plain GPU offload.
        # cpu/hybrid both read experts on the CPU, so banks load in the native (CPU-readable)
        # layout; the GPU slot-cache GEMM reads those same native rows. decode_target also
        # gates the CPU executor build below.
        cpu_layer_ids = _resolve_cpu_layers(config, config.model_config.num_moe_layers)
        if (
            not cpu_layer_ids
            and config.moe_cpu_layers is None
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes(self._host_tables_bytes) is not None
        ):
            cpu_layer_ids = _auto_cpu_layers(
                config, config.model_config.num_moe_layers, reserved=self._host_tables_bytes
            )
        if self._mtp_bank_layers:
            # #801 bullet 7: the head runs on EVERY step and its bank is one layer; routing it to
            # the CPU executor would be a different experiment (and `_auto_cpu_layers` picks TAIL
            # layers, which is exactly where the head's id lands). ⛔ Not a fix to the auto policy:
            # the backbone's own set is untouched.
            cpu_layer_ids = frozenset(
                i for i in cpu_layer_ids if i < self._backbone_model_config.num_moe_layers
            )
        if config.moe_backend == "hybrid":
            decode_target = "hybrid"
        elif cpu_layer_ids:
            decode_target = "cpu"
        else:
            decode_target = "gpu"
        # split residency: where pinning is quota-capped (_pin_budget_bytes), pin only the GPU layers' banks and mlock the CPU layers'
        # uncapped hosts keep every bank pinned (CPU decode reads them the same; overlap prefill stays on)
        # not applied to plain --moe-backend cpu; all-locked under a cap = --moe-backend offload --moe-cpu-layers 1.0
        split_residency = (
            bool(cpu_layer_ids)
            and config.moe_backend in ("offload", "hybrid")
            and _pin_budget_bytes(self._host_tables_bytes) is not None
        )
        if config.moe_backend == "cpu" and not split_residency:
            # cpu mode pins every bank for the prefill double buffer; over the pin cap that dies in cudaHostRegister, so lock everything instead
            from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

            budget = _pin_budget_bytes(self._host_tables_bytes)
            bank_bytes = None
            if budget is not None:
                bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config)
            if bank_bytes and bank_bytes > budget:
                split_residency = True
                logger.info_rank0(
                    f"--moe-backend cpu: banks {bank_bytes / 2**30:.2f} GiB exceed the "
                    f"pin budget; OS-locking all layers instead of pinning"
                )
        if split_residency and config.moe_prefill_overlap:
            # locked (unregistered) layers cannot feed the async pinned H2D double buffer; their prefill is a synchronous pageable copy via materialize
            logger.info_rank0(
                "--moe-cpu-layers split residency: disabling MoE prefill overlap "
                "(locked layers prefill via synchronous pageable copies)"
            )
            object.__setattr__(config, "moe_prefill_overlap", False)
        if cache_factory is None:
            # Fast path: an FTW checkpoint loads its repacked banks directly.
            # Slow path: load_expert_banks auto-picks parallel vs serial baseline by
            # expert-tensor granularity. Both pin-after-fill.
            # --expert-load: serial/parallel force the read; auto (None) lets load_expert_banks
            # pick (parallel for scattered experts, with a low-RAM fallback to serial).
            expert_parallel = {"serial": False, "parallel": True}.get(config.expert_load, None)
            requested_residency = None
            if split_residency:
                from freetoken.moe.host_banks import HostResidency

                requested_residency = [
                    HostResidency.LOCKED.value if i in cpu_layer_ids
                    else HostResidency.PINNED.value
                    for i in range(self._backbone_model_config.num_moe_layers)  # #801: backbone only
                ]
            banks = load_expert_banks(
                config.model_path,
                self._backbone_model_config,  # #801 bullet 7: the CHECKPOINT's MoE layers, not the head's
                device=self.device,
                dtype=self.dtype,
                dummy=config.use_dummy_weight,
                parallel=expert_parallel,
                decode_target=("cpu" if decode_target in ("cpu", "hybrid") else "gpu"),
                layer_residency=requested_residency,
            )
            if self._mtp_bank_layers:
                # ── #801 bullet 7: the head's experts, as extra LAYERS on the same banks ─────
                # Round 2 built them as an NVFP4 source bank in exactly the shape
                # `load_nvfp4_expert_sources` returns, so appending is the whole attachment: the
                # offload cache then sees a 49th layer with nothing special about it, and the
                # `--moe-cache-auto` budget below prices its 512 experts like any other layer's.
                # ⛔ Appended BEFORE the budget is solved, for that reason.
                extra = self.model.mtp_expert_source_banks(
                    self._backbone_model_config, banks.quant_format
                )
                for name, per_layer in extra.items():
                    if name not in banks.sources:
                        raise ValueError(
                            f"#801: the draft bank has {name!r}, which is not in the "
                            f"{banks.quant_format!r} schema {sorted(banks.sources)}"
                        )
                    if len(per_layer) != self._mtp_bank_layers:
                        raise ValueError(
                            f"#801: the draft bank holds {len(per_layer)} layer(s) of {name!r}, "
                            f"but the pools were sized for {self._mtp_bank_layers}"
                        )
                    banks.sources[name].extend(per_layer)
                if banks.layer_residency is not None:
                    from freetoken.moe.host_banks import HostResidency

                    banks.layer_residency.extend(
                        [HostResidency.PINNED.value] * self._mtp_bank_layers
                    )
                logger.info_rank0(
                    f"#801: attached the draft head's expert bank as layer(s) "
                    f"{list(range(self._backbone_model_config.num_moe_layers, config.model_config.num_moe_layers))}"
                )
            if config.moe_cache_auto:
                size, pages, overlap = self._resolve_auto_moe_cache_size(config, banks)
                object.__setattr__(config, "moe_cache_size", size)
                object.__setattr__(config, "moe_prefill_overlap", overlap)
                if config.num_page_override is None:
                    # Honor the plan's KV half too: MoE slots and KV pages were solved
                    # against ONE budget (ratio x baseline - weights), so both must come
                    # from it. Re-solving pages later from a fresh free-memory reading
                    # double-counts everything allocated since the weights measurement
                    # (this expert cache, the CPU-executor GPU buffers, allocator
                    # slack) and goes negative whenever the expert fill is exact --
                    # a greedy fill leaves no headroom for the measurement delta.
                    object.__setattr__(config, "num_page_override", pages)
                logger.info_rank0(
                    f"--moe-cache-auto resolved moe_cache_size={size} "
                    f"num_pages={pages} (prefill_overlap={overlap})"
                )
            _require_offload_cache_size(config.moe_cache_size, config.model_config.num_experts)
            cache = OffloadMoeCache(
                # Models with leading dense layers (GLM-4) only have experts on the MoE
                # layers; num_moe_layers == num_layers when first_k_dense_replace == 0.
                num_layers=config.model_config.num_moe_layers,
                num_experts=config.model_config.num_experts,
                cache_size=config.moe_cache_size,
                device=self.device,
                cache_policy=config.moe_cache_policy,
                prefill_overlap=config.moe_prefill_overlap,
                prefill_hit_d2d=config.moe_prefill_hit_d2d,
                quant_format=banks.quant_format,
                decode_target=decode_target,
                hybrid_max_fetch=config.moe_hybrid_max_fetch,
            )
            # before set_bank_sources: the residency validation and the copy plan's skip of non-pinned layers key on the CPU-layer set
            cache.cpu_layer_ids = cpu_layer_ids
            cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
            cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        else:
            cache = cache_factory(config, self.device)
            cache.decode_target = decode_target
            cache.hybrid_max_fetch = config.moe_hybrid_max_fetch
            cache.cpu_layer_ids = cpu_layer_ids
        if decode_target == "hybrid":
            self._resolve_hybrid_fetch(config, cache)
        # Must be set before CUDA graph capture so the (device-side) accumulation ops are
        # captured and re-run on every decode replay.
        cache.collect_stats = config.moe_collect_stats
        # attach_offload_moe_cache walks for OffloadMoELayers, or defers to a model's
        # _iter_offload_moe_layers() hook when its MoE blocks are bespoke nn.Modules (DSV4).
        layers = attach_offload_moe_cache(self.model, cache)
        assert len(layers) == config.model_config.num_moe_layers
        if cache.decode_target in ("cpu", "hybrid"):
            self._init_cpu_moe_executor(config, cache, layers)
        self.ctx.moe_offload_cache = cache
        self.moe_offload_cache = cache
        return cache

    def _resolve_hybrid_fetch(self, config: EngineConfig, cache) -> None:
        """Resolve --moe-hybrid-max-fetch -1 (auto) into a bandwidth-matched fetch fraction.

        Perfect fetch/compute overlap wants fetched : cpu-computed misses = pcie_bw :
        (cpu_bw - pcie_bw), i.e. fetching a pcie_bw / cpu_bw fraction of each decode
        step's misses -- both sides then finish together instead of one idling. The
        achieved bandwidths come from the cached `ft bench bw` profile (the same one the
        auto backend pick reads); without a usable profile the old fixed cap of 1 applies.
        """
        if config.moe_hybrid_max_fetch >= 0:
            return  # explicit fixed cap
        from freetoken.moe.bench_profile import load_hybrid_fetch_fraction

        gpu_name, gpu_uuid = _profile_gpu(self.device.index)
        fraction = load_hybrid_fetch_fraction(
            cache.quant_format, gpu_name=gpu_name, gpu_uuid=gpu_uuid
        )
        if fraction is None:
            cache.hybrid_max_fetch = 1
            logger.warning_rank0(
                "--moe-hybrid-max-fetch auto: no usable `ft bench bw` profile for "
                f"{cache.quant_format!r} experts; using a fixed fetch cap of 1"
            )
            return
        cache.hybrid_max_fetch = cache.num_experts  # inert: the fraction is the cap
        cache.hybrid_fetch_fraction = fraction
        logger.info_rank0(
            f"--moe-hybrid-max-fetch auto: fetching {fraction:.1%} of each decode step's "
            "expert misses over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU"
        )

    def _init_cpu_moe_executor(self, config: EngineConfig, cache, layers) -> None:
        """Build the persistent CPU MoE executor (decode-time expert compute).

        Must run before CUDA graph capture: the worker pool has to be live for the
        eager warmup forward, and the pinned IO buffers / host-func task pointers
        must be stable for the captured nodes. Buffers/tasks themselves are
        allocated lazily on the first (eager) forward at each batch size.
        """
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        sample = layers[0]
        required = ("top_k", "activation", "apply_router_weight_on_input")
        if not all(hasattr(sample, attr) for attr in required):
            raise NotImplementedError(
                "CPU MoE backend is not yet supported for this model architecture "
                f"(MoE layer {type(sample).__name__} is missing {required})."
            )
        # Decode batches never exceed max_running_req, but CUDA-graph padding can
        # round a batch up to the largest captured size; cover both.
        max_tokens = max(config.max_running_req, config.cuda_graph_max_bs or 0, 1)
        # gpt-oss mxfp4 carries clamped-swiglu scalars; other formats use the defaults.
        executor = CpuMoeExecutor(
            cache,
            top_k=sample.top_k,
            activation=sample.activation,
            apply_router_weight_on_input=sample.apply_router_weight_on_input,
            num_threads=config.moe_cpu_threads,
            max_tokens=max_tokens,
            device=self.device,
            swiglu_alpha=getattr(sample, "hidden_act_alpha", 1.702),
            swiglu_limit=getattr(sample, "swiglu_limit", None),
        )
        cache.set_cpu_executor(executor)
        self.cpu_moe_executor = executor

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def _target_moe_and_expert_bytes(self, moe_cache_size: int | None) -> tuple[int, int]:
        from freetoken.engine.cache_budget import expert_bytes_per_slot

        target_moe = (
            moe_cache_size
            if moe_cache_size is not None
            else (self.moe_offload_cache.cache_size if self.moe_offload_cache else 0)
        )
        per_expert_bytes = (
            expert_bytes_per_slot(self.moe_offload_cache.bank_sources)
            if self.moe_offload_cache is not None else 0
        )
        return target_moe, per_expert_bytes

    def _resize_kv_pool(self, config, num_pages: int, num_swa_pages: int | None) -> None:
        # IN-PLACE, identity-preserving: the CacheManager's swa_pool reference, ctx.kv_cache and
        # the model's per-access pool property all keep pointing at THIS pool, which frees its old
        # buffers before allocating the new ones. mark_for_rebind re-binds per-bind scratch on the
        # next forward (graph re-capture); the prefix tree + page bookkeeping reset is the
        # scheduler's generic cache_manager.rebuild.
        if self.kv_cache.needs_rebind_on_rebuild:
            self.model.mark_for_rebind()
        self.kv_cache.rebuild_from_config(config, num_pages, num_swa_pages=num_swa_pages)
        self.num_pages = num_pages

    def _refresh_seq_state(self, config) -> None:
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        if aligned_max_seq_len != self.page_table.shape[1]:
            # max_seq_len changed (e.g. KV grew past the startup token budget); the page table
            # columns must track it or new requests would index out of bounds. The scheduler
            # re-points its managers to engine.page_table on a num_pages rebuild.
            self.ctx.page_table = self.page_table = torch.zeros(
                (config.max_running_req + 1, aligned_max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        self.kv_cache.attach_page_table(self.page_table)

    @torch.inference_mode()
    def rebuild_runtime_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool,
        and/or the window pool (num_swa_pages: an absolute pinned window), followed by CUDA-graph
        re-capture. Does NOT reload weights or host expert banks. The caller (scheduler) must
        guarantee no in-flight prefill/decode.
        """
        config = self.config
        if (moe_cache_size is None and num_pages is None and num_mamba_slots is None
                and num_swa_pages is None):
            return

        # 0a. Geometry prevalidation BEFORE any destructive free. An invalid target (moe
        #     slots on a model with no offload cache, moe below num_experts / above the
        #     marlin cap, non-positive pages, or too few GDN slots to run) must reject
        #     recoverably with the old cache intact -- NOT after teardown, which would
        #     leave the server unable to serve. These checks are model-agnostic.
        if moe_cache_size is not None:
            if self.moe_offload_cache is None:
                raise CacheRebuildRejected(
                    "moe_cache_size requested but this model has no MoE offload cache"
                )
            try:
                self.moe_offload_cache.validate_rebuild(moe_cache_size)
            except ValueError as e:
                raise CacheRebuildRejected(str(e)) from e
        if num_pages is not None and num_pages <= 0:
            raise CacheRebuildRejected(f"num_pages must be positive, got {num_pages}")
        if num_mamba_slots is not None:
            if self.linear_state_pool is None:
                raise CacheRebuildRejected(
                    "num_mamba_slots requested but this model has no GDN state pool"
                )
            # num_mamba_slots is the USABLE slot count (what the user sets and the status bar
            # shows); the pool also reserves a padding sink (slot 0), so the physical pool is
            # num_mamba_slots + 1. _linear_pool_min_slots is the physical floor -> usable - 1.
            min_usable = _linear_pool_min_slots(config) - 1
            if num_mamba_slots < min_usable:
                raise CacheRebuildRejected(
                    f"num_mamba_slots {num_mamba_slots} is below the minimum {min_usable} "
                    f"(non-evictable working set for max_running_req={config.max_running_req}) "
                    f"needed to run; admission would deadlock"
                )
        if num_swa_pages is not None:
            # An absolute window pin for the radix-SWA window pool (Gemma) or the DSV4 window tier;
            # meaningless for dense/MHA models and the naive SWA path (concurrency x window).
            if not _supports_swa_ratio(config):
                raise CacheRebuildRejected(
                    "num_swa_pages requested but this model has no window pool "
                    "(needs DSV4 or a sliding-window model with --cache-type radix)"
                )
            if num_swa_pages <= 0:
                raise CacheRebuildRejected(
                    f"num_swa_pages must be positive, got {num_swa_pages}"
                )

        # 0b. Pool-family budget fit-check BEFORE any destructive free: an unfit geometry
        #     must reject (recoverable) so the old caches stay intact and serving continues,
        #     rather than freeing and then OOMing into permanent failure. The engine supplies
        #     the memory account; the pool answers whether its target geometry fits.
        target_moe, per_expert_bytes = self._target_moe_and_expert_bytes(moe_cache_size)
        # Price the sibling GDN state pool at ITS target (physical slots = usable + padding
        # sink) and hand the bytes in -- the KV pool only budgets its own tiers.
        target_mamba = (
            num_mamba_slots + 1
            if num_mamba_slots is not None
            else (self.linear_state_pool.num_slots if self.linear_state_pool is not None else None)
        )
        self.kv_cache.validate_rebuild(
            config, num_pages=num_pages,
            num_swa_pages=num_swa_pages, target_moe=target_moe,
            per_expert_bytes=per_expert_bytes, baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes, current_num_pages=self.num_pages,
            extra_fixed_bytes=(
                state_pool_bytes(config, target_mamba) if target_mamba is not None else 0
            ),
            extra_note=(
                f", mamba={target_mamba - 1} slots" if target_mamba is not None else ""
            ),
        )

        torch.cuda.synchronize(self.device)
        # Preserve the CUDA-graph batch-size set resolved at startup. The auto heuristic keys
        # off free memory, which is far smaller now that the caches are resident (post-cache
        # free << startup pre-load free), so re-deriving it here would silently drop large
        # batch sizes after the first rebuild. Reusing the already-resolved list keeps the
        # captured coverage identical (the fit-check above guarantees the graph headroom fits).
        prior_graph_bs = self.graph_runner.graph_bs_list
        # Point of no return for the scheduler's rollback logic: from here the live graphs and
        # pools start being freed. A failure BEFORE this flag flips leaves the engine serving
        # untouched (no rollback needed); after it, only a rebuild restores service.
        self.rebuild_teardown_started = True
        # 1. Tear down CUDA graphs + backend capture scratch (free-before-alloc).
        self.attn_backend.reset_capture()
        self.graph_runner.destroy_cuda_graphs()
        # 2. Resize caches in place (each frees its old GPU tensors before allocating).
        # Pin the new window first (validated above) so any KV-pool rebuild below sizes the window
        # to it (_dsv4_pool_sizes / _swa_paged_num_tokens read config.swa_num_pages_override).
        # frozen EngineConfig — mutate in place like the moe_cache_size path; `config.x = y` raises
        # FrozenInstanceError, which here aborts the rebuild after the CUDA graphs are gone (→ 503).
        if num_swa_pages is not None:
            object.__setattr__(config, "swa_num_pages_override", num_swa_pages)
        if moe_cache_size is not None:
            assert self.moe_offload_cache is not None, "no MoE offload cache to resize"
            self.moe_offload_cache.rebuild(moe_cache_size)
        if num_pages is not None:
            # sets self.num_pages (rebuilds KV + window)
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        elif num_swa_pages is not None:
            # Window-only change: no page-count change, but re-derive the window pool at the new
            # pin against the CURRENT page count. This re-allocs the same-size full pool and
            # the resized window, both inside the pool's own rebuild_from_config.
            self._resize_kv_pool(config, self.num_pages, num_swa_pages)
        if num_mamba_slots is not None:
            # Reallocate the GDN state pool (frees old tensors first). Must sit between graph
            # teardown and re-capture so the recaptured graphs bind the new state tensors.
            # +1 for the reserved padding sink: num_mamba_slots is the usable count.
            self.linear_state_pool.rebuild(num_mamba_slots + 1)
        # 3. Refresh max_seq_len (+ generic page table) for the new token budget.
        self._refresh_seq_state(config)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        # 4. Re-capture CUDA graphs against the new tensors (reset_capture above re-armed
        #    the backend; _sync_get_memory empties the cache so freed memory is reclaimed).
        gc.collect()
        free_min = self._sync_get_memory()[0]
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=prior_graph_bs,  # reuse the startup-resolved set (see above)
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=free_min,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )

    #: #801 r6 b9bw: fp → int bit view, by element width. ⛔ A dtype missing here is one
    #: hashed by VALUE, which for floats is the one thing this instrument may not do.
    _FT801_BITS = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }

    @staticmethod
    def _ft801_state_fingerprint(pool, slots) -> list:
        """Exact per-(surface, layer) integer fingerprints of each slot's linear state. #801 b9bw.

        ``pool`` is a :class:`LinearStatePool`; ``slots`` is one live GDN state slot per request
        (``Req.linear_slot_idx``, else ``Req.table_idx`` -- `attention/linear.py`'s own rule).
        Returns one ``{surface: [hash per layer]}`` dict per slot, in the order asked.

        ⛔ **MODEL-AGNOSTIC, like every other #801 hook in this file, and here it is load-bearing.**
        It reads only `LinearStatePool`'s public tensors and finds the declared SIBLING states by
        iterating ``pool.slot_states`` -- so PLE's two pools are covered without `ple` being named
        in the engine, and a state some other model's config declares is covered the day it is
        declared.

        ⛔⛆ **THE CURRENCY IS BITS, NOT CLOSENESS.** Bullet 9's gate is byte-identical served text,
        and 9bt already refused to tighten `test_ple_verify_801.py`'s ``assert_close`` to
        ``torch.equal`` because THAT goes red on correct code. This is the other side of that
        decision: it asserts nothing about the numbers, it reports only whether two arms hold the
        SAME BITS. A tolerance here would hide exactly the accumulation the round is hunting --
        and ``-0.0``, and a NaN's identity with itself, are gated so nobody "fixes" it into a
        float compare.

        ⛔ The surface order is SORTED, not insertion order: two arms are two container loads, and
        a reader that lined the columns up by whatever order a config happened to declare would
        compare the wrong state and call it a divergence.
        """
        surfaces = [("conv", pool.conv_states), ("recurrent", pool.recurrent_states)]
        surfaces += [(n, pool.slot_states[n]) for n in sorted(pool.slot_states)]
        index = torch.as_tensor(
            list(slots), dtype=torch.int64, device=pool.conv_states.device
        )

        rows, names, widths = [], [], []
        for name, states in surfaces:
            # [L, B, *rest] -- `index_select` COPIES, so the result is contiguous and the
            # `.view(dtype)` below is a bit reinterpretation rather than a raise on a strided
            # slice.
            picked = states.index_select(1, index)
            flat = picked.reshape(picked.shape[0] * picked.shape[1], -1)
            if flat.dtype in Engine._FT801_BITS:
                flat = flat.view(Engine._FT801_BITS[flat.dtype])
            bits = flat.to(torch.int64)
            # ⛔⛆ NOT A SUM. A plain sum is blind to a PERMUTATION -- precisely the shape a state
            # written by a chunked verify path against a per-token decode path can take. The
            # second moment is what makes a value's POSITION part of the fingerprint. Both
            # products wrap in int64: two's complement, deterministic, and a hash rather than an
            # arithmetic claim.
            weight = torch.arange(
                1, bits.shape[1] + 1, dtype=torch.int64, device=bits.device
            )
            mixed = bits.sum(-1) * 0x100000001B3 + (bits * weight).sum(-1)
            rows.append(mixed.reshape(picked.shape[0], picked.shape[1]))
            names.append(name)
            widths.append(picked.shape[0])

        # ⭐ ONE copy to the host for every surface, every layer and every request in the batch,
        # which is why the seam takes a LIST of slots rather than one. It runs on every forward of
        # an instrumented load, on BOTH arms; a `tolist` per surface per request would put four
        # syncs times the batch size inside the decode loop (#912's watch).
        table = torch.cat(rows, dim=0).tolist()
        out = []
        for j in range(len(index)):
            fp, at = {}, 0
            for name, width in zip(names, widths):
                fp[name] = [int(table[at + i][j]) for i in range(width)]
                at += width
            out.append(fp)
        return out

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        use_graph = self.graph_runner.can_use_cuda_graph(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, use_graph):
            logits = self.graph_runner.replay(batch) if use_graph else self.model.forward()
        if self.cpu_moe_executor is not None:
            # One pinned read: surfaces a fired flag-handshake watchdog (dead coordinator
            # -> stale expert outputs) as a loud error instead of silent corruption.
            self.cpu_moe_executor.raise_if_unhealthy()

        # ── #801 round 6 bullet 8: verify, or today's path exactly ────────────────────
        # ⛔⛆ `mtp_verify_step` REPLACES the three statements below for a staged verify step -- it
        # does not wrap them. `complete_one` advances by exactly ONE and would have to be undone;
        # `logits[: batch.size]` keeps the VERIFY row and DROPS the bonus row; and the sampler
        # draws one token per REQUEST where a verify step needs one per ROW. It returns `None` on
        # every batch that was not staged (every prefill, every flag-off row, every no-draft
        # decode step), and the `else` below is then the image's own code, byte for byte.
        #
        # ⛔⛆ IT CARRIES A HOST SYNC, AND THAT IS AN OPERATOR DECISION (2026-09-12), NOT AN
        # OVERSIGHT. `complete_one()` runs BEFORE the sampler today, which is legal only because
        # the advance is `+1` and needs no token values; a verify step's advance is `accepted_len`,
        # a device tensor off the sampler. `scheduler.py::overlap_loop` PREPARES batch N+1 before
        # it DRAINS batch N, so the advance cannot be deferred to the drain. Option (a) -- sync on
        # speculative steps only -- was chosen; `models/qwen4_exp/spec.py`'s bullet-8 header has
        # the three grounds. ⛔ The sync is a REAL COST OF SPECULATION and bullet 9 banks it as its
        # own line rather than absorbing it into the tok/s number.
        #
        # ⛔ MODEL-AGNOSTIC, like every other #801 hook here: a model without `mtp_verify_step`
        # never reaches the branch at all.
        # ⭐⭐⭐ #801 r6 bullet 9bj: COMMITCHECK -- `cached_len` READ ON BOTH SIDES OF THE COMMIT.
        # ⛔⛆ 9bi measured a reply whose KV held 63 generated tokens while the drains were handed
        # 62, and could not say which of two things happened: a commit advanced `cached_len` past
        # the `accepted_len` it was given, or a commit never reached a drain at all. The publish
        # rows CANNOT decide it -- they sample `cached_len` once per DRAIN, and by then
        # `overlap_loop` has launched the next forward, whose commit is already in the number
        # (9be measured that phase and was right to refuse to flag it per row). Both reads here
        # are inside ONE `forward_batch`, so no phase can get between them.
        # ⭐ ONE SITE FOR BOTH ADVANCE PATHS: `mtp_verify_step` -> `spec.commit_verify` for a
        # staged row, `req.complete_one()` for every prefill, flag-off row and no-draft step. A
        # dial that watched only `commit_verify` could not add up to the reply's KV total.
        # ⭐ MEASUREMENT-SAFE, like PUBCHECK and the retire ledger: python ints the request
        # already holds, and `staged.committed` is host integers by its own contract (filled by
        # `verify_step` AFTER the sync). No device read, no second forward, nothing that makes
        # `can_use_cuda_graph` decline the capture. ⇒ a decode figure off a COMMITCHECK load MAY
        # be banked as the arm's.
        # ⭐ OFF BY DEFAULT and the budget is read ONCE per Engine: a row that never sets the dial
        # pays one `getattr` per forward. `FREETOKEN_MTP801_COMMITCHECK=N` counts FORWARDS (not
        # drains, not retires), set by `arm_mtp_801.sh` from `FT801_COMMITCHECK`.
        # ⛔ MODEL-AGNOSTIC: it reads the step the model staged on the batch, never an import.
        # ⭐⭐⭐ #801 r6 b9cb: HIDDENCHECK's key, captured on the ENTRY side of the commit. See the
        # builder's edit 10 header for why it is taken here and printed below.
        # ⚠ `take()` SPENDS a unit and returns None cheaply when the dial is off, so a row that
        #   never sets it pays one `getattr` and one int compare per forward -- COMMITCHECK's
        #   standard. The peek/spend split `gdncheck_instrumented` needs does not arise: there is
        #   exactly one call site and it always goes on to print.
        _ft801_hc_take = getattr(self.model, "mtp_hiddencheck_take", None)
        _ft801_hc_buf = _ft801_hc_take() if _ft801_hc_take is not None else None
        _ft801_hc_entry = None
        _ft801_hc_rows = 0
        if _ft801_hc_buf is not None:
            _ft801_hc_entry = []
            for req in batch.reqs:
                _ft801_hc_entry.append(
                    {
                        "uid": getattr(req, "uid", None),
                        # ⛔⛆ **NO `prompt_fp` HERE, UNLIKE STATECHECK'S ROW, AND TWO REASONS
                        #   AGREE.** 9bz measured it DEGENERATE -- all five cells came back under
                        #   one hash, because under a chat template the leading 32 ids are the
                        #   system preamble -- so it keys nothing that `max_device_len` does not.
                        #   And computing it costs `input_ids[:32].sum()`, a DEVICE READ, inside
                        #   the region `test_the_dial_reads_nothing_off_the_device` slices to
                        #   prove COMMITCHECK is measurement-safe. ⇒ the entry capture is python
                        #   ints only, and the single sync of this dial is the buffer read below.
                        "max_device_len": int(req.max_device_len),
                        # ⛔ THE ENTRY POSITION. Read after the commit this is one row late.
                        "cached_len": int(req.cached_len),
                        # ⛔⛆ **ABSOLUTE, NOT A ROW COUNT** -- `core.py::Req` sets
                        #   `device_len = len(input_ids)` and asserts `cached_len < device_len`,
                        #   so at position 11,470 it reads 11,471. It is carried for the record;
                        #   the ROW COUNT is `extend_len` below. #801 r6 b9cf.
                        "device_len": int(req.device_len),
                        # ⭐⭐⭐ THE ROWS THIS FORWARD RUNS FOR THIS REQUEST -- `device_len -
                        #   cached_len`, which is 1 on a plain decode step, 2 on a verify step and
                        #   the CHUNK on a prefill. ⛔⛆ 9cb used `device_len` here and the FIRST
                        #   PREFILL CHUNK HID IT: at `cached_len == 0` the two are equal, so the
                        #   wrong formula is exactly right on the one forward every load runs
                        #   first, and wrong on every forward after it.
                        "extend_len": int(req.extend_len),
                        # ⭐ Where this request's rows START in the forward. The layer writes rows
                        #   in batch order, so a request owns `[at, at + extend_len)`.
                        "at": _ft801_hc_rows,
                    }
                )
                _ft801_hc_rows += int(req.extend_len)
        _ft801_cc_left = getattr(self, "_ft801_commitcheck_left", None)
        if _ft801_cc_left is None:
            _ft801_cc_left = int(os.getenv("FREETOKEN_MTP801_COMMITCHECK", "0") or 0)
        _ft801_cc_before = None
        if _ft801_cc_left > 0:
            self._ft801_commitcheck_left = _ft801_cc_left - 1
            _ft801_cc_before = [
                (getattr(req, "uid", None), int(req.cached_len), int(req.device_len))
                for req in batch.reqs
            ]
        next_tokens_gpu = None
        if hasattr(self.model, "mtp_verify_step"):
            next_tokens_gpu = self.model.mtp_verify_step(
                batch,
                logits,
                args,
                page_size=self.ctx.page_size,
                linear_pool=self.linear_state_pool,
            )
        if next_tokens_gpu is None:
            for req in batch.reqs:
                req.complete_one()

            batch_logits = logits[: batch.size]
            next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
        else:
            next_tokens_gpu = next_tokens_gpu.to(torch.int32)
            batch_logits = logits[: next_tokens_gpu.shape[0]]
        # ⭐⭐⭐ #801 r6 bullet 9bn: THE DRAINED FORWARD'S OWN POST-COMMIT `cached_len`, CARRIED.
        # ⛔⛆ `_process_last_data` runs ONE FORWARD LATE -- `overlap_loop` schedules, prepares and
        # FORWARDS batch N (whose commit lands on the host, right above) before it drains batch
        # N-1. So `req.cached_len` read in the drain is the NEXT forward's post-commit value on
        # every row -- 36 of 36 on load 25's banked ledger, 0 of 36 the drained forward's own --
        # and `publish_plan` judged the *length* verdict one position ahead. The verify arm then
        # finished one generated token early and the next drain hit `already_finished` and shipped
        # nothing: the `kv 63 / published 62` gap, in the act (README *Bullet 9bm*).
        # ⛔⛆ IT CANNOT BE RECOVERED DOWNSTREAM. After any commit `device_len == cached_len + 1`
        # ALWAYS, so the gap between the two carries nothing about the next forward's width. The
        # number exists only here, and only for the moment between the advance and the return.
        # ⭐⭐ ON THE BATCH, NOT ON THE REQUEST, and that is the design decision. `scheduler/
        # decode.py::schedule_next_batch` builds a FRESH `Batch` every forward, and the drain
        # already holds the batch it is draining (`last_data[0].batch`) -- the same carrier
        # `batch.ft801_verify_step` uses and the bullet-8 publish slice already trusts. A copy held
        # on the REQUEST would be the next forward's by drain time, and a two-deep rotation reads
        # two advances stale exactly when the request skipped the next forward (finished, aborted,
        # or simply not scheduled) -- a partial carrier whose residue reads as a FLAKY one-token
        # loss, which is the worst failure mode this round could ship.
        # ⭐ BOTH ADVANCE PATHS, ONE SITE: `mtp_verify_step` -> `spec.commit_verify` for a staged
        # row, `req.complete_one()` for every prefill, flag-off row and no-draft step. A
        # `VerifyStep`-only field would cover only the first, and the contamination is set by the
        # NEXT forward's width -- so an UNSTAGED drain on a speculating row is contaminated too.
        # ⛔ NOT A DIAL AND NOT GUARDED. This is the served path, not an instrument: a dial would
        # make the correct verdict conditional on a measurement flag being set. It costs one python
        # int per request per forward, off values the request already holds -- no device read,
        # nothing that makes `can_use_cuda_graph` decline the capture.
        # ⛔ MODEL-AGNOSTIC: plain ints on the batch, no import, and a drain that does not find the
        # attribute falls back to the image's own `not req.can_decode`.
        batch.ft801_post_commit = tuple(int(req.cached_len) for req in batch.reqs)
        if _ft801_cc_before is not None:
            # ⚠ `accepted` is recorded as -1 rather than raised when the staged step is short,
            #   for PUBCHECK's reason: an instrument may not kill the row it is measuring.
            #   A no-draft / prefill / flag-off forward advances by one through `complete_one`,
            #   which IS an `accepted_len` of 1 and is recorded as such.
            _ft801_cc_step = getattr(batch, "ft801_verify_step", None)
            _ft801_cc_acc = list(getattr(_ft801_cc_step, "committed", ()) or ())
            print(
                "[#801] commitcheck: "
                + _ft801_json.dumps(
                    {
                        "left": _ft801_cc_left,
                        "staged": _ft801_cc_step is not None,
                        "is_prefill": bool(getattr(batch, "is_prefill", False)),
                        "reqs": [
                            {
                                "uid": uid,
                                "accepted": (
                                    1
                                    if _ft801_cc_step is None
                                    else (
                                        int(_ft801_cc_acc[i])
                                        if i < len(_ft801_cc_acc)
                                        else -1
                                    )
                                ),
                                "cached_len_before": cached_before,
                                "cached_len_after": int(batch.reqs[i].cached_len),
                                "device_len_before": device_before,
                                "device_len_after": int(batch.reqs[i].device_len),
                                "delta": int(batch.reqs[i].cached_len) - cached_before,
                            }
                            for i, (uid, cached_before, device_before) in enumerate(
                                _ft801_cc_before
                            )
                        ],
                    }
                ),
                file=_ft801_sys.stderr,
                flush=True,
            )
        # ⭐⭐⭐ #801 r6 bullet 9bx: STATECHECK -- the COMMITTED linear state, fingerprinted, on
        # BOTH arms. See the builder's edit 9 header for why it is read here and nowhere else.
        _ft801_sc_left = getattr(self, "_ft801_statecheck_left", None)
        if _ft801_sc_left is None:
            _ft801_sc_left = int(os.getenv("FREETOKEN_MTP801_STATECHECK", "0") or 0)
        if _ft801_sc_left > 0 and self.linear_state_pool is not None:
            self._ft801_statecheck_left = _ft801_sc_left - 1
            # ⛔ `attention/linear.py`'s OWN rule: the hybrid-radix live slot when allocated, else
            #   `table_idx`. Reading the other one is a different request's state on every load
            #   this round has run.
            _ft801_sc_slots = [
                (req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx)
                for req in batch.reqs
            ]
            _ft801_sc_fp = Engine._ft801_state_fingerprint(
                self.linear_state_pool, _ft801_sc_slots
            )
            print(
                "[#801] statecheck: "
                + _ft801_json.dumps(
                    {
                        "left": _ft801_sc_left,
                        "staged": getattr(batch, "ft801_verify_step", None) is not None,
                        "is_prefill": bool(getattr(batch, "is_prefill", False)),
                        "reqs": [
                            {
                                "uid": getattr(req, "uid", None),
                                # ⛔ The cross-ARM key. A uid is whatever order the scheduler
                                #   happened to admit requests in; these two are the same on
                                #   either arm for the same cell, and constant for the life of
                                #   the request. ⚠ `prompt_fp` reads the LEADING ids, so it is
                                #   stable only while the prompt is longer than the window --
                                #   which every cell this round serves is, and the reader says so.
                                "prompt_fp": (
                                    int(req.input_ids[:32].to(torch.int64).sum())
                                    * 0x100000001B3
                                    + int(req.input_ids[:32].numel())
                                ),
                                "max_device_len": int(req.max_device_len),
                                "cached_len": int(req.cached_len),
                                "device_len": int(req.device_len),
                                "slot": int(_ft801_sc_slots[i]),
                                "fp": _ft801_sc_fp[i],
                            }
                            for i, req in enumerate(batch.reqs)
                        ],
                    }
                ),
                file=_ft801_sys.stderr,
                flush=True,
            )
        # ⭐⭐⭐ #801 r6 bullet 9cb: HIDDENCHECK -- layer 0's seven block tensors, per row, on BOTH
        # arms. Keyed by the ENTRY values captured above, never by `req.cached_len` as it reads
        # here. See the builder's edit 10 header.
        if _ft801_hc_entry is not None:
            # ⭐ ONE host copy, sliced to the rows this forward actually wrote. The buffer is
            #   sized for the largest batch the capture resolver reserved, and copying all of it
            #   would put thousands of stale integers in the log on every decode step.
            _ft801_hc_table = _ft801_hc_buf[:, :_ft801_hc_rows].tolist()
            for _ft801_hc_req in _ft801_hc_entry:
                _ft801_hc_at = _ft801_hc_req.pop("at")
                # ⛔ NUMBERS, NO NAMES: `engine.py` may not import `spec.py`. The order is
                #   `spec.FT801_HIDDENCHECK_NAMES` and the reader is what names it.
                _ft801_hc_req["fp"] = [
                    [_ft801_hc_t[_ft801_hc_at + _ft801_hc_i] for _ft801_hc_t in _ft801_hc_table]
                    for _ft801_hc_i in range(_ft801_hc_req["extend_len"])
                ]
            print(
                "[#801] hiddencheck: "
                + _ft801_json.dumps(
                    {
                        "staged": getattr(batch, "ft801_verify_step", None) is not None,
                        "is_prefill": bool(getattr(batch, "is_prefill", False)),
                        "tensors": len(_ft801_hc_table),
                        "reqs": _ft801_hc_entry,
                    }
                ),
                file=_ft801_sys.stderr,
                flush=True,
            )
        if hasattr(self.model, "mtp_shadow_step"):
            # ── #801 round 5 bullet 4: the real draft step, after the real sampler ──────────
            # Read-only: never feeds back into `next_tokens_gpu` or anything downstream of it.
            # Duck-typed like every other #801 hook, so this file stays model-agnostic
            # (`models/qwen4_exp/model.py`'s own docstring: "imports nothing qwen4exp-specific").
            # ⛔⛆ RE-ENTER `self.ctx.forward_batch(batch)` -- the `with` above (line 921) has
            # already EXITED by this point, so `Context._batch` is back to `None`. The head's own
            # forward runs the SAME MoE layer type the backbone does, and `moe.py::forward` reads
            # `get_global_ctx().batch.is_prefill` unconditionally; outside any `forward_batch`
            # scope that raises `AssertionError: No active batch in context` (found on this
            # round's own GPU box load, `check_shadow_load_801.sh`, 2026-09-12 -- the CPU-only
            # wiring suite spies on `draft_next_token_ids` and so never runs the real MoE forward
            # that needed this). `forward_host_ctx` is deliberately NOT re-entered alongside it:
            # it is the disk-PLE prefetch hook and the head "ships no PLE tensors" (`mtp.py`'s own
            # assertion), so the head's forward never touches what it guards.
            with self.ctx.forward_batch(batch):
                # ⛔ #801 round 5 bullet 7b: `batch_logits` is the TARGET's own distribution `p`,
                # which the acceptance mass (`draft.py::acceptance_mass`) needs alongside the
                # head's `q`. Passed, not recomputed: the sampler above already has it, and a
                # second forward would be both a cost and a second chance to differ.
                self.model.mtp_shadow_step(next_tokens_gpu, batch, args, batch_logits)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    @torch.inference_mode()
    def _warmup_prefill(self) -> None:
        """Compile the Triton prefill path before the first real request.

        Decode CUDA graph capture warms the decode path, but the first prefill
        can still pay Triton/cublas setup costs. Use the dummy request row and
        restore it afterwards so padded decode graph replay keeps using the
        dedicated dummy KV slot.
        """
        if self.max_seq_len < 2:
            return

        warmup_lens = [min(80, self.max_seq_len)]
        if self.max_seq_len >= 128:
            warmup_lens.append(128)
        warmup_lens = sorted({length for length in warmup_lens if length >= 2})
        if not warmup_lens:
            return

        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            for length in warmup_lens:
                dummy_row[:length] = torch.arange(
                    length, dtype=torch.int32, device=self.device
                )
                warm_req = Req(
                    input_ids=torch.zeros(length, dtype=torch.int32, device="cpu"),
                    table_idx=self.dummy_req.table_idx,
                    cached_len=0,
                    output_len=1,
                    uid=-1,
                    sampling_params=None,  # type: ignore[arg-type]
                    cache_handle=None,  # type: ignore[arg-type]
                )
                batch = Batch(reqs=[warm_req], phase="prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = torch.zeros(length, dtype=torch.int32, device=self.device)
                batch.positions = torch.arange(length, dtype=torch.int32, device=self.device)
                batch.out_loc = dummy_row[:length]
                self.attn_backend.prepare_metadata(batch)
                with self.ctx.forward_batch(batch):
                    self.model.forward()
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        logger.info_rank0(
            f"Prefill warmup complete for lengths {warmup_lens} "
            f"in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _profile_gpu(index: "int | None" = None) -> Tuple[str | None, str | None]:
    """(name, uuid) of visible device ``index`` (default: the current, i.e. bound, device); (None, None) without CUDA."""
    if not torch.cuda.is_available():
        return None, None
    ident = gpu_identity(torch.cuda.current_device() if index is None else index)
    return ident["name"], ident["uuid"]


def _ensure_expandable_segments() -> None:
    """Default the CUDA allocator to expandable segments.

    The motivating case is the offload prefill, which repeatedly dequantizes
    variable-sized NVFP4 expert blocks to BF16 (a different size per layer as the
    active-expert count varies). Under that alloc/free churn the default caching
    allocator fragments badly -- reserved memory can balloon far past the actual peak
    allocation (observed ~78GiB reserved for a <30GiB working set).
    ``expandable_segments`` lets freed regions of any size be reused, keeping
    reserved ~= allocated, so it is applied to every run, not just offload ones.

    Env vars are parsed once at import and ignored if set afterwards, so we apply the
    setting via the runtime API instead. Must run before the first CUDA allocation (the
    caller guarantees CUDA is not yet initialized). Any user-provided allocator config
    is respected and left untouched.
    """
    if os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception as exc:  # pragma: no cover - depends on torch build
        logger.info_rank0(f"Could not enable expandable_segments ({exc}); continuing")
        return
    logger.info_rank0("Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)")


def _resolve_cache_type(has_linear_attention: bool, requested: str) -> str:
    # Hybrid GDN models default to the HybridRadixCache (snapshots GDN state at chunk
    # boundaries -> cross-request prefix reuse). An explicit ``--cache-type naive`` opts out
    # to the old no-reuse path (debugging / parity baseline / lower GDN-state memory).
    if has_linear_attention:
        return "naive" if requested == "naive" else "hybrid_radix"
    return requested


def _adjust_dsv4_config(config: EngineConfig, override) -> None:
    """DSV4 engine-config reconciliation at config-resolution time (before the pool exists).
    Syncs the resolved runtime config into the opaque ``dsv4_args`` payload, sets
    page_size to the window page P, forces single-chunk prefill, and clamps cuda_graph_bs/max_bs to
    the DSV4 decode batch size.
    """
    model_config = config.model_config
    model_config.dsv4_args.max_seq_len = config.max_seq_len
    model_config.dsv4_args.max_batch_size = config.max_running_req + 1  # +1 dummy
    # config.swa_full_tokens_ratio is the DSV4 window/full ratio directly (default sizing);
    # a runtime rebuild pins an absolute window via swa_num_pages_override instead.
    # DSV4's KV page IS the P-token window page (window == radix reuse granularity == lcm of
    # the compress ratios), so max_num_tokens = num_pages * page_size holds like every model.
    P = model_config.dsv4_args.window_size
    override("page_size", P)
    logger.info_rank0(f"DSV4 KV pages are {P}-token window pages; page_size set to {P}")
    # The generic CacheManager materializes DSV4 'radix' as the shared SWARadixCache (is_swa);
    # 'naive' stays naive with the pool's swa currency riding swa_paged.
    if getattr(config, "cache_type", "radix") != "naive":
        override("cache_type", "swa_radix")
    # 'radix' (SWARadixCache on the full-loc currency, carry-aware re-prefill) is the default and is
    # honored, as is an explicit 'naive'. Don't let max_extend_tokens force a second chunk within
    # one prompt (the pool's prefill_chunk_budget still chunks prompts larger than the window
    # pool); prefill batches ragged (bs>=1), each segment resuming from its own cached_len.
    if getattr(config, "max_extend_tokens", 0) < config.max_seq_len:
        override("max_extend_tokens", config.max_seq_len)

    # DSV4 decode batches at most max_running_req rows; its full-loc snapshot is sized to that,
    # so a graph bs above it would exceed the backend's captured snapshot rows. Clamp any
    # oversized explicit list / max_bs here (before GraphRunner ever sees it).
    mr = config.max_running_req
    if config.cuda_graph_max_bs is not None and config.cuda_graph_max_bs > mr:
        logger.warning_rank0(
            f"cuda_graph_max_bs {config.cuda_graph_max_bs} exceeds DSV4 max_running_req {mr}; "
            "clamping to max_running_req (larger decode batches never occur)."
        )
        override("cuda_graph_max_bs", mr)
    if config.cuda_graph_bs is not None:
        kept = [bs for bs in config.cuda_graph_bs if bs <= mr]
        if kept != list(config.cuda_graph_bs):
            dropped = [bs for bs in config.cuda_graph_bs if bs > mr]
            logger.warning_rank0(
                f"dropping cuda_graph_bs entries {dropped} above DSV4 max_running_req {mr} "
                "(larger decode batches never occur)."
            )
            override("cuda_graph_bs", kept)


def _parse_cpu_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-cpu-layers``: an explicit MoE-layer id list (``"3,7,11"``), a count
    (``"8"`` -> 8 layers evenly strided across depth), or a fraction (``"0.5"``). Ids are
    indices into the MoE layers, ``[0, num_moe_layers)``."""
    s = spec.strip()
    if not s:
        return frozenset()
    if "," in s:
        ids = {int(x) for x in s.split(",") if x.strip()}
        for i in ids:
            if not 0 <= i < num_moe_layers:
                raise ValueError(
                    f"--moe-cpu-layers id {i} out of range [0, {num_moe_layers})"
                )
        return frozenset(ids)
    if "." in s:
        frac = float(s)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"--moe-cpu-layers fraction {frac} must be in [0, 1]")
        k = round(frac * num_moe_layers)
    else:
        k = int(s)
        if not 0 <= k <= num_moe_layers:
            raise ValueError(f"--moe-cpu-layers count {k} must be in [0, {num_moe_layers}]")
    # k layers spread evenly across depth (frozenset dedups any rounding collisions;
    # k == 0 yields an empty range, hence an empty set).
    return frozenset(round(i * num_moe_layers / k) for i in range(k))


def _resolve_cpu_layers(config: EngineConfig, num_moe_layers: int) -> frozenset[int]:
    """MoE layer ids whose decode runs on the CPU executor.

    ``--moe-backend cpu`` -> every layer. ``--moe-backend offload`` + ``--moe-cpu-layers``
    -> the parsed subset (the rest stay on the GPU offload/PCIe path). Otherwise none.
    """
    if config.moe_backend == "cpu":
        return frozenset(range(num_moe_layers))
    spec = config.moe_cpu_layers
    if not spec or not is_offload_moe_backend(config.moe_backend):
        return frozenset()
    return _parse_cpu_layers_spec(spec, num_moe_layers)


# expert activations the CPU MoE executor supports (csrc ActKind)
_CPU_MOE_ACTS = (
    "silu", "swish", "gelu", "gelu_tanh", "gelu_pytorch_tanh", "swigluoai",
)


def _cpu_moe_executor_viable(model_config) -> bool:
    """Whether an automatic CPU-decode decision may target the CPU MoE executor.

    A default boot must degrade to GPU offload instead of crashing in CpuMoeExecutor after the whole load; explicit cpu/hybrid/--moe-cpu-layers picks still fail loudly."""
    from freetoken.moe.cpu_executor import _WFMT_IDS, compiled_extension_supports

    try:
        from freetoken.kernel import _cpu_moe  # noqa: F401
    except ImportError:
        return False
    act = getattr(model_config, "hidden_act", "silu")
    moe_wfmt = getattr(model_config, "moe_weight_format", None)
    if act not in _CPU_MOE_ACTS and moe_wfmt != "mxfp4":
        return False
    if moe_wfmt != "mxfp4" and not compiled_extension_supports(act):
        return False
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
    return fmt == "mxfp4" or fmt in _WFMT_IDS


def _pin_budget_bytes(reserved: int = 0) -> int | None:
    """Bytes this process can still safely cudaHostRegister, or None when the platform does not cap pinning (plain Linux).

    WSL's WDDM-backed CUDA caps pinning near half of RAM, shared across processes -- budget 40%. FREETOKEN_PIN_BUDGET_GB overrides anywhere. ``reserved`` subtracts host bytes already pinned outside the expert banks (qwen4_exp's PLE table)."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        cap = int(float(env) * 2**30)
    elif not hasattr(os, "uname") or "microsoft" not in os.uname().release.lower():  # WSL kernel tag
        return None
    else:
        cap = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.4)
    return max(0, cap - reserved)


def _auto_cpu_layers(config: EngineConfig, num_moe_layers: int, reserved: int = 0) -> frozenset[int]:
    """Pick CPU (locked) MoE layers automatically when the banks exceed the pin budget.

    Locks just enough head+tail layers: per-layer decode miss rates are U-shaped, so the ends are the cheapest to move off the slot cache."""
    from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

    bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config)
    if not bank_bytes:
        return frozenset()
    budget = _pin_budget_bytes(reserved)
    if budget is None or bank_bytes <= budget:
        return frozenset()
    if not _cpu_moe_executor_viable(config.model_config):
        logger.info_rank0(
            f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB exceed the "
            f"pin budget {budget / 2**30:.2f} GiB, but the CPU MoE executor cannot "
            f"serve this model; keeping every layer pinned on the GPU offload path"
        )
        return frozenset()
    n = min(num_moe_layers, math.ceil(num_moe_layers * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = frozenset(range(head)) | frozenset(range(num_moe_layers - (n - head), num_moe_layers))
    logger.info_rank0(
        f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB > pin budget "
        f"{budget / 2**30:.2f} GiB; locking {n} head+tail MoE layers for CPU decode "
        f"({sorted(ids)})"
    )
    return ids


# MoE-only knobs and the value each resolves to on a dense model. moe_backend is handled
# separately (its dense value is 'fused', but 'auto' resolves there without a warning).
_DENSE_MOE_SETTINGS = {
    "moe_cache_size": 0,
    "moe_cache_rate": None,
    "moe_cache_auto": False,
    "moe_cpu_layers": None,
    "moe_cpu_threads": 0,
    "moe_hybrid_max_fetch": -1,
    "moe_prefill_overlap": True,
    "moe_prefill_hit_d2d": False,
    "expert_load": "auto",
}


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    model_config = config.model_config
    single_stream_only = getattr(model_config, "single_stream_only", False)
    is_dsv4 = getattr(model_config, "dsv4_args", None) is not None
    has_swa_attention = getattr(model_config, "has_swa_attention", False)
    has_linear_attention = getattr(model_config, "has_linear_attention", False)
    is_moe = getattr(model_config, "is_moe", False)
    expert_quant = getattr(model_config, "expert_quant", "none")

    if not is_moe:
        # A dense model has no routed experts: the MoE knobs are inert, and the offload family
        # is worse than inert -- engine init would build an expert cache for a model that has
        # none and abort startup (weights already resident) on an unrelated expert-source
        # error. Drop them at this one choke point, which the CLI and the programmatic
        # LLM(...) path both pass through. 'auto'/'fused' is the silent dense resolution;
        # anything else was asked for explicitly, so report what is being ignored.
        dropped = [
            f"{name}={getattr(config, name)!r}"
            for name, dense_value in _DENSE_MOE_SETTINGS.items()
            if getattr(config, name, dense_value) != dense_value
        ]
        if config.moe_backend not in ("auto", "fused"):
            dropped.insert(0, f"moe_backend={config.moe_backend!r}")
        override("moe_backend", "fused")
        for name, dense_value in _DENSE_MOE_SETTINGS.items():
            override(name, dense_value)
        if dropped:
            logger.warning_rank0(
                f"{getattr(model_config, 'model_type', 'model')} is a dense model (no routed "
                f"experts); ignoring MoE settings: {', '.join(dropped)}"
            )

    if single_stream_only:
        # The model runs one sequence at a time: it collapses the batch to one row and the
        # decode CUDA graph is captured at bs=1. Force the runtime knobs so the KV pool, page
        # table and graph capture all stay bs=1.
        if config.max_running_req != 1:
            override("max_running_req", 1)
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs >= 1:
            override("cuda_graph_bs", [1])
            override("cuda_graph_max_bs", 1)

    if config.cuda_graph_max_bs is None:
        override("cuda_graph_max_bs", config.max_running_req)

    if is_dsv4:
        _adjust_dsv4_config(config, override)

    if has_swa_attention:
        # Both SWA cache paths use the global-paged swa pool (page_size==1 only for now).
        if config.page_size != 1:
            raise ValueError(
                f"SWA models currently support only page_size=1, got {config.page_size}."
            )
        # naive keeps cache_type='naive' (NaivePrefixCache, no reuse) on the paged pool (==
        # sglang SWAChunkCache); radix materializes as swa_radix (SWARadixCache, cross-request
        # reuse == sglang SWARadixCache). Both allocate from the same swa pool + free out-of-window.
        if getattr(config, "cache_type", "radix") != "naive":
            if not 0.0 < config.swa_full_tokens_ratio <= 1.0:
                raise ValueError(
                    f"swa_full_tokens_ratio must be in (0, 1], got {config.swa_full_tokens_ratio}"
                )
            override("cache_type", "swa_radix")

    if has_linear_attention:
        override(
            "cache_type",
            _resolve_cache_type(True, getattr(config, "cache_type", "radix")),
        )

    # Type x backend capability matrix: resolve auto from the per-type priority
    # lists, then validate whatever is now selected (explicit or auto) -- every
    # comma part must serve every required type, with packages/arch available.
    required_attn_types = _required_attn_types(model_config)
    _dtype = getattr(config, "dtype", None)  # duck-typed test configs omit it
    if (
        required_attn_types & {AttnType.BSA, AttnType.QSA}
        and _dtype is not None
        and _dtype.itemsize != 2
    ):
        # Reject at config time: the BSA/QSA pool's own assert only fires after the
        # model is resident (and not at all under `python -O`).
        raise ValueError(
            f"--dtype {config.dtype}: block-sparse attention serves 16-bit "
            "compute only (the index slab budgets 2 bytes/token); use bfloat16 "
            "or float16."
        )
    if _dtype == torch.float16 and "mxfp8" in (
        getattr(model_config, "attn_quant", "none"),
        getattr(model_config, "dense_quant", "none"),
    ):
        # The MXFP8 GEMV folds the pow2-descaled fp8 weight into the activation
        # dtype; fp16's narrow exponent can overflow/flush what bf16 represents
        # exactly, and the combination was never numerically validated.
        raise ValueError(
            "--dtype float16 with MXFP8 resident weights is unsupported (the "
            "W8A16 fold is only validated exact in bfloat16); use bfloat16."
        )
    if config.attention_backend == "auto":
        override(
            "attention_backend",
            _resolve_auto_attention_backend(required_attn_types),
        )
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
    _validate_attention_backend_choice(config, override, required_attn_types)

    if config.moe_cache_rate is not None:
        total_experts = config.model_config.num_moe_layers * config.model_config.num_experts
        override("moe_cache_size", math.ceil(total_experts * config.moe_cache_rate))

    # The CPU MoE executor supports the silu/gelu family plus the clamped
    # swigluoai (csrc ActKind; "gpt_oss_swiglu" rides inside the mxfp4 kernel and
    # swigluoai the generic GEMV epilogue). A model with any other expert
    # activation cannot decode on the CPU: reject an explicit cpu/hybrid pick at
    # config time, and keep auto from upgrading offload -> hybrid off the profile.
    # hidden_act (the dense activation) stands proxy for the expert activation --
    # true for every in-tree model. mxfp4 experts pass regardless: their act runs
    # inside the mxfp4 kernel, not the generic epilogue.
    _cpu_moe_act_ok = getattr(model_config, "hidden_act", "silu") in _CPU_MOE_ACTS or (
        getattr(model_config, "moe_weight_format", None) == "mxfp4"
    )
    if (
        is_moe
        and not _cpu_moe_act_ok
        and (config.moe_backend in ("cpu", "hybrid") or config.moe_cpu_layers)
    ):
        asked = (
            f"--moe-cpu-layers={config.moe_cpu_layers!r}"
            if config.moe_backend not in ("cpu", "hybrid")
            else f"--moe-backend {config.moe_backend!r}"
        )
        raise ValueError(
            f"{asked}: the CPU MoE executor does not support this model's expert "
            f"activation {getattr(model_config, 'hidden_act', None)!r}; drop the flag "
            "and let every layer decode on the GPU offload path instead."
        )

    if is_moe and config.moe_backend == "auto":
        # A MoE model always defaults to the offload family: experts stream from pinned host
        # banks into an auto-sized GPU slot cache, which is the only default that serves a model
        # bigger than the GPU. The resident 'fused' path (bf16 / block-fp8 experts, the two
        # formats MoELayer can allocate) is still reachable, but only when asked for explicitly
        # -- auto never picks it, because nothing here knows whether the experts would fit in
        # HBM and a wrong guess is a weight-load OOM rather than a slower-but-working run.
        default_backend = "offload"
        # Hardware-adaptive config: a cached `ft bench bw` profile can upgrade
        # the offload default to hybrid when this machine's CPU MoE bandwidth clears its PCIe
        # gather bandwidth by the bench threshold (default 2x). hybrid is VRAM-equivalent to
        # offload -- same auto-sized GPU slot cache (_resolve_auto_moe_cache_size), plus a
        # host-RAM CPU executor -- so this never raises the OOM risk; with no profile (or one
        # from different hardware) it stays offload. offload remains the always-safe fallback.
        # Key the lookup on the real expert format: mxfp4/q4_0 live in moe_weight_format when
        # expert_quant is "none", and "none" with no weight format means plain bf16 experts.
        moe_wfmt = getattr(model_config, "moe_weight_format", None)
        bench_fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
        from freetoken.moe.bench_profile import load_backend_recommendation

        gpu_name, gpu_uuid = _profile_gpu()
        if load_backend_recommendation(bench_fmt, gpu_name=gpu_name, gpu_uuid=gpu_uuid) == "hybrid":
            from freetoken.moe.cpu_executor import compiled_extension_supports

            _act = getattr(model_config, "hidden_act", "silu")
            if not _cpu_moe_act_ok:
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the CPU MoE executor does not "
                    f"support this model's expert activation "
                    f"{getattr(model_config, 'hidden_act', None)!r}; staying on offload"
                )
            elif moe_wfmt != "mxfp4" and not compiled_extension_supports(_act):
                # Stale prebuilt _cpu_moe.so: an explicit cpu/hybrid pick still
                # hard-fails in the executor, but a default must not turn into a
                # post-load crash -- degrade to offload.
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the compiled _cpu_moe "
                    f"extension predates activation {_act!r} (rebuild with "
                    f"`python setup.py build_ext --inplace`); staying on offload"
                )
            else:
                default_backend = "hybrid"
                logger.info_rank0(
                    f"benchbw profile recommends hybrid for {bench_fmt!r} experts on this GPU"
                )
        override("moe_backend", default_backend)
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

        if (
            is_offload_moe_backend(config.moe_backend)
            and config.moe_cache_size <= 0
            and config.moe_cache_rate is None
            and not getattr(config, "moe_cache_auto", False)
        ):
            # args.py's "no sizing flag -> default --moe-cache-auto" only fires when the
            # backend is already offload-family at *parse* time. A bare `ft serve <FTW MoE
            # checkpoint>` (no --moe-backend, no cache flags) still has moe_backend=="auto" at
            # parse time -- the auto -> offload/cpu/hybrid resolution above is the first point
            # the concrete backend is known, so mirror the same default here: no sizing flag
            # was given, so let the scheduler resolve the cache size from free VRAM instead of
            # failing the _require_offload_cache_size guard with size=0.
            override("moe_cache_auto", True)
            logger.info_rank0(
                "No MoE cache sizing flag given; defaulting to --moe-cache-auto for "
                f"auto-selected backend {config.moe_backend!r}"
            )

    if is_moe and config.moe_backend == "fused":
        # An explicit 'fused' keeps the experts resident, so there is no slot cache to size. The
        # sizing flags no longer redirect the backend, so ignore them here and say so -- the
        # geometry the user asked for is what runs. Report the flag actually passed: --moe-cache-
        # rate was already folded into moe_cache_size above, and the three are mutually exclusive.
        if config.moe_cache_rate:
            inert = f"--moe-cache-rate={config.moe_cache_rate}"
        elif config.moe_cache_size:
            inert = f"--moe-cache-size={config.moe_cache_size}"
        elif getattr(config, "moe_cache_auto", False):
            inert = "--moe-cache-auto"
        else:
            inert = None
        if inert:
            logger.warning_rank0(
                f"MoE backend 'fused' keeps its experts resident; ignoring {inert} "
                "(use --moe-backend offload to serve experts from a slot cache)"
            )
            override("moe_cache_size", 0)
            override("moe_cache_rate", None)
            override("moe_cache_auto", False)

    if is_moe and config.moe_backend == "cpu":
        # CPU-compute decode keeps experts in host RAM and computes them on the CPU;
        # the GPU only holds the two-layer prefill double buffer. So the slot cache is
        # fixed at exactly two expert layers (prefill overlap requires >= 2*num_experts)
        # and --moe-cache-size / --moe-cache-auto / --moe-cache-rate do not apply.
        num_experts = config.model_config.num_experts
        if getattr(config, "moe_cache_auto", False):
            override("moe_cache_auto", False)
        override("moe_cache_size", 2 * num_experts)
        override("moe_prefill_overlap", True)
        logger.info_rank0(
            f"MoE backend 'cpu': decode computes experts on CPU; GPU keeps a "
            f"two-layer prefill buffer (moe_cache_size={2 * num_experts})"
        )

    if (
        is_moe
        and expert_quant not in ("none", "fp8_block")
        and not is_offload_moe_backend(config.moe_backend)
    ):
        raise ValueError(
            f"{expert_quant} experts require --moe-backend offload or cpu, "
            f"got {config.moe_backend!r}"
        )

    if is_moe and config.moe_cpu_layers and config.moe_backend not in ("offload", "hybrid"):
        # the layer split needs the offload host banks + slot cache; 'cpu' already runs every layer on CPU, 'fused' keeps experts resident on the GPU (no host banks)
        raise ValueError(
            "--moe-cpu-layers requires --moe-backend offload or hybrid (got "
            f"{config.moe_backend!r}); use --moe-backend cpu to run all layers on CPU"
        )

    if is_moe:
        object.__setattr__(model_config, "moe_backend", config.moe_backend)
    object.__setattr__(model_config, "nvfp4_backend", config.nvfp4_backend)

    # Must stay LAST: page_size is only final here (_adjust_dsv4_config sets P=128, the
    # TRTLLM block sets 64). Also covers the programmatic LLM(...) path that bypasses parse_args.
    if config.num_token_override is not None:
        if config.num_page_override is not None:
            raise ValueError("--num-tokens and --num-pages are mutually exclusive")
        if config.num_token_override % config.page_size != 0:
            raise ValueError(
                f"--num-tokens {config.num_token_override} is not a multiple of the resolved "
                f"page size {config.page_size}; nearest valid values: "
                f"{config.num_token_override // config.page_size * config.page_size} or "
                f"{(config.num_token_override // config.page_size + 1) * config.page_size}"
            )
        override("num_page_override", config.num_token_override // config.page_size)

    # The rope cos/sin table is baked to rotary_config.max_position, and neither rope kernel
    # bounds-checks the position it gathers with -- a longer ceiling reads past the table.
    # DSV4 is exempt: it sizes its own table from the resolved max_seq_len (_adjust_dsv4_config).
    rotary = getattr(model_config, "rotary_config", None)
    seq_override = getattr(config, "max_seq_len_override", None)
    if seq_override is not None and rotary is not None and not is_dsv4:
        if seq_override > rotary.max_position:
            raise ValueError(
                f"--max-seq-len-override {seq_override} exceeds the model's "
                f"rope table ({rotary.max_position} positions). Serving past it would read "
                "out of bounds; extend the checkpoint's rope_scaling / "
                "max_position_embeddings in config.json instead."
            )

    # The startup ServerArgs dump is the *requested* config, printed in the frontend process
    # before any of the resolution above ran -- so "moe_backend='auto'" is all it can say. This
    # is the one line that reports what actually runs, for every path (explicit backends never
    # hit an "Auto-selected ..." log at all).
    resolved = [
        f"attention_backend={config.attention_backend!r}",
        f"cache_type={getattr(config, 'cache_type', 'radix')!r}",
        f"page_size={config.page_size}",
    ]
    if is_moe:
        resolved.insert(0, f"moe_backend={config.moe_backend!r}")
    logger.info_rank0(f"Resolved config: {', '.join(resolved)}")
