# FreeToken on AMD RDNA4 — branch `rocm-gfx1201`

This branch makes **FreeToken** build and serve **Qwen3.8-Flash-Next-NVFP4** on **AMD RDNA4**
(`gfx1201`, Radeon AI PRO R9700) under ROCm 7.14 — on one card, and across two cards with
`--tp-size 2`. It is **19 commits** on top of upstream
[`4b94bdc3`](https://github.com/FlashML-org/FreeToken/commit/4b94bdc38a46a4dfe534e8793126160d56904c44),
and it is not a product: it is the tree that serves this model on one particular box, published so
that somebody else with the same hardware does not have to rediscover it.

It has **known defects**, listed below, one of which silently degrades every request that contains
an image. Read §2 before you send a picture to it.

- **Build:** `Dockerfile.gfx1201`, §5.
- **Run:** §6 — and §1 first, because three settings are mandatory and nothing tells you when one
  is missing.
- **Upstream's own README** is unchanged, below this one, in `README.md`.

---

## 1. ⛔ The three settings without which it does not work

Not one of these produces a clear error when it is wrong. Two of them produce a *plausible* wrong
answer instead — a machine that looks slow, or a crash that reads like a hardware fault.

```sh
--expert-load parallel                      # a command-line flag to `ft serve`
-e PYTORCH_ALLOC_CONF=expandable_segments:False
-e HIP_VISIBLE_DEVICES=0                    # or 0,1 for --tp-size 2 — always explicit
```

**`--expert-load parallel`.** The default is `auto`, and on this hardware `auto` always chooses
`serial`. Serial is the memory-mapped reader, and here it is roughly a **100× stall** on weight
load — the server appears to hang rather than to fail. ⚠ The flag exists partly to **override a
guard**: `moe/expert_banks.py`'s `_host_ram_fits_parallel` will only pick the parallel reader on its
own when `MemAvailable` exceeds **Σ(shards) + largest shard** — for this checkpoint that is
125.91 + 9.99 = **135.9 GiB**, which a 128 GiB box can never satisfy. Passing the flag explicitly is
how you say *"I know, load it anyway"*, and the source says so itself (*"override with
`--expert-load parallel`"*). ⚠ The log line to look for is:

```
expert banks: low free RAM -> serial build (avoids parallel-reader OOM; override with --expert-load parallel)
expert banks: slow path (serial build)
```

⛔ If you see `serial build`, you are about to wait a very long time. Without this note a replicator
reasonably concludes the parallel loader is unsupported on their machine; it is not.

**`PYTORCH_ALLOC_CONF=expandable_segments:False`.** The engine **forces**
`expandable_segments:True` — `engine/engine.py:1028`, called before the first CUDA allocation, and
it logs *"Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)"*, which is the override
this uses. On `gfx1201` that setting produces a **GPU memory fault**,
not an out-of-memory error — a 1.25 GiB allocation dies with 12.25 GiB free, and the message names
the wrong cause. ⛔⛆ Worse: PyTorch's own OOM text *recommends* `expandable_segments:True`. If you
hit an allocation failure here, the advice printed on your screen is the thing that breaks it.
Set it to `False` in the environment; the engine's own default is overridden by the env var.

**`HIP_VISIBLE_DEVICES`.** Always set it, always explicitly, even for one card. The engine's
bandwidth profile (`ft bench bw`) is keyed by **GPU UUID**, and on a second card an unset value is
silently ignored rather than reported. For `--tp-size 2`, `0,1`.

**TP=2 adds three more:**

```sh
--disable-pynccl                            # or the load looks like a TIMEOUT, with no error
-e OMP_WAIT_POLICY=PASSIVE                  # or a cold load takes ~19 minutes instead of ~60 s
--ple-backend disk                          # see §2.4 — on TWO cards this is the right choice
```

⛔⛆ The absence of `--disable-pynccl` does not report a collectives failure. It reports **nothing**,
and the load times out. `OMP_WAIT_POLICY=PASSIVE` is an OpenMP spin-wait problem: without it the
ranks burn every core spinning during weight load and a cold start takes ~19 minutes.

**TP=1 needs one flag that looks like a default and is not:**

```sh
--ple-backend pinned                        # the upstream DEFAULT is `disk` and costs ~27 % of decode here
```

---

## 2. ⛔ Known defects — read this before you trust an answer

### 2.1 ⛔⛆ Image token positions are wrong, and so is everything after them

The served checkpoint declares **M-RoPE** — `text_config.rope_parameters.mrope_section =
[11, 11, 10]`, `mrope_interleaved: true` — and the publisher's own text backbone applies three
position streams to a request that contains an image. **This branch implements none of it** and
feeds ordinary 1-D positions. `models/qwen4_exp/vision.py` even carries a comment asserting the
checkpoint has no `mrope_section`; that comment is **false** for the checkpoint we serve. The value
is dropped earlier, in `models/config.py`, because `rope_type == "default"` short-circuits
`rope_scaling` to `None` — `mrope_section` does not live under `rope_scaling` at all.

**What that means in practice:**

- ⛔ An image request is **silently mispositioned** — not only the image tokens, but **every token
  after them**, because the position delta the model expects to carry forward is never computed.
  Nothing errors. The answer is fluent. We have **no fidelity instrument** on this box and
  therefore **make no claim about how wrong it is** — only that it is wrong by construction.
- ⭐ **Text-only requests are provably exact.** With no image grids the model's own code collapses
  to `t = h = w = sequence position`, which is what this branch feeds. This is the model's
  degenerate case, not a coincidence, and it is read from the publisher's source.

⭐⭐ **Upstream has since fixed this properly.** `FlashML-org/FreeToken#454` (merged 2026-09-13)
implements M-RoPE end to end for `qwen4_exp`: it reads `mrope_section` from `rope_parameters` —
the place it actually lives — handles the interleaved layout, builds three position streams in the
scheduler, and carries `mrope_delta` across cached and device lengths, which is the
"everything after the image" half. ⇒ **for image quality, upstream's vision path is better than
this branch's, and ours is superseded.** What upstream does *not* have is tensor parallelism for
this model (§3) or the RDNA kernel fix (§3), so the two are not yet interchangeable.

### 2.2 ⛔ Nothing bounds the number of images in one request

One HTTP request can take **the host** down — not the GPU, the host. The multimodal preprocessor's
padded batch costs roughly `n_images × p_max` in host RAM at about 57 KiB per soft token, and:

- `--max-multimodal-prompt-tokens` bounds the **whole prompt**.
- `--max-image-soft-tokens` bounds the **sum** over the request's images.
- ⛔⛆ **Neither bounds `n_images`, and neither bounds the largest single image.**

Because the cost is `n × p_max` while the cap is on `Σ`, a request that is well **inside** the
configured cap can still ask for tens of GiB: we killed a row at **51 %** of its own configured cap.
The failure is mid-request — the client hangs, the worker dies, and the row is gone.

**Mitigations that work, in order of how much they buy:**

1. ⭐ **Cap what a single image expands to, in the model's `preprocessor_config.json`**, by lowering
   `size.longest_edge`. This is config-only, needs no code, and it turned the request that killed
   the row above into one that served. A 2048²-equivalent ceiling (`longest_edge: 4194304`) is the
   value this box runs.
   ⛔⛆ If you bind that file into a container, the host file must **exist** before `docker run`, or
   Docker silently creates a *directory* at that path and the model fails to load.
2. `--max-image-soft-tokens` as a **wire ceiling**. ⚠ Read it on the right axis: it is a host-RAM
   instrument and refuses rather than resizes; it is the wrong control for VRAM, which scales with
   the **largest single** image. VRAM's controls are `--memory-ratio` and `FREETOKEN_VIT_GROUP`.
3. Bound it in front of the engine. If this is exposed to anything you do not control, cap the
   number of image parts at your proxy. There is no engine-side flag that does it.

### 2.3 ⚠ TP=2 vision carries a rank desync

At `--tp-size 2`, an encode failure on one rank could branch before all ranks agreed on it. The last
commit on this branch (`scheduler: agree an encode failure across all TP ranks before it branches`)
addresses the agreement path, and upstream's own scheduler comments call the general machinery
*"deferred"*. ⚠ The limitation is recorded rather than closed: **TP=2 + images is the least-tested
combination on this branch.** TP=1 + images, and TP=2 text-only, have many more hours on them.

### 2.4 ⛔ `--ple-backend` defaults to `disk`, and on one card that is −27 % decode

The Per-Layer Embedding table can be pinned in host memory or streamed from disk. Upstream's default
is `disk`. On **one** card on this hardware that costs about **27 %** of decode — not because of the
NVMe, but because of a ROCm launch-gating sync path (the stream-memop fast path `dlopen`s
`libcuda`, which does not exist on ROCm; the expected log line is `sync=launch-gating`).

- **TP=1 ⇒ pass `--ple-backend pinned` explicitly.** The default will cost you a quarter of decode
  and nothing will say so.
- **TP=2 ⇒ `--ple-backend disk`** is correct: it is what brings the pin from ~111 GiB down to
  ~64 GiB, which is what makes two ranks fit on this box at all.

---

## 3. What is actually this branch's, and what is not

⛔ This branch is **not** "vision and TP=2 on RDNA4" any more. Upstream merged vision for this model
on 2026-09-13 (#454) — a whole `mm/` subsystem, better factored than ours. Honest accounting, read
from upstream `main` at `68a81ffd` on 2026-09-15:

| | |
|---|---|
| ⭐ **The only tree we know of that serves this model on RDNA4** | the ROCm stack + the router fix + the kernel clamp, together |
| ⭐ **The RDNA LDS clamp** | Without it, `qwen4_exp` on RDNA serves **prompts of at most 15 tokens** and then the worker dies — a chat template alone is longer than that. The QSA prefill kernel asks for more LDS than an RDNA workgroup has. Upstream `kernel/triton/qsa/attend.py` still has no RDNA handling. Filed as upstream **#349**. |
| ⭐ **Tensor parallelism for `qwen4_exp`** | Upstream `models/qwen4_exp/weight.py:207` **still** raises `NotImplementedError("qwen4_exp weight loading supports TP=1 only")`. Nobody's TP=2 for this model is upstream. |
| ⭐ **The TP>1 relay handshake** | upstream's `sync_all_ranks` is a general barrier, not the slow-joiner fix. Filed as upstream **#364**. |
| ⭐ **All-rank agreement on an encode failure** | upstream's scheduler comments call this machinery *"deferred"*. |
| ⛔ **Vision** | superseded by upstream **#454**, which is also *more correct* than ours (§2.1). |
| ⛔ **Chunked multimodal prefill, image-keyed prefix cache, batch packing** | all have upstream counterparts now. |

⭐ **Credit where it is due.** [@gdevenyi](https://github.com/gdevenyi) carried the same
intermediate-axis shard as **PR #385** and the vision half as **PR #386**, on CUDA (2 × RTX 6000
Ada), independently and at the same time. Both were closed unmerged on 2026-09-14. The TP=2 design
here was arrived at separately; the overlap is real and worth saying out loud.

⭐ Upstream findings from this work that are filed and open: **#348** (the two request surfaces
disagree on image input), **#349** (the LDS ceiling), **#350** (`CpuMoeExecutor` silently wrong under
graph replay — ⭐ being fixed upstream by #378, validated on this same card), **#364** (the relay
handshake), **#371** (a cold two-rank load is OpenMP spin-wait, not thread count — independently
reproduced by another user on 2026-09-11).

⛔ **No quality or fidelity claim is made anywhere in this README.** This box has no fidelity
instrument. Every number in this document is throughput or latency.

---

## 4. Hardware — what actually gates this

⛔ **Host RAM is the gate, not VRAM.** The expert banks stay **host-resident** and are streamed every
step: **~31.65 GiB per rank**, at TP=1 *and* at TP=2. Two ranks do not halve it; they duplicate it.

| | this box |
|---|---|
| GPUs | 2 × Radeon AI PRO R9700, 32 GiB each (`gfx1201`) |
| host RAM | 128 GiB (122 GiB usable) — ⚠ this is the tight resource |
| model on disk | ~126 GB (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) |
| ROCm | 7.14, via the pinned `rocm/pytorch` base (see `Dockerfile.gfx1201`) |

- **`gfx1201` is tested.** `gfx1200` is *untested here* — same generation, probably needs no code
  change, but nobody has run it.
- **RDNA3 and older are not this branch's target.** The LDS clamp is written for the RDNA LDS
  ceiling generally, but nothing else here has been exercised on RDNA3.
- **A 2 × 48 GiB box crosses a threshold this one does not** — at that size the experts become
  resident, the PCIe gather disappears, and the TP=2 numbers below understate what you would see.

### What to expect if your build works

⚠ **Every number is state-bound**: one box, the configuration in §6, measured 2026-09. They are here
so you can tell a working build from a broken one, not as a benchmark.

| shape | decode @ 10k ctx | decode @ 100k | TTFT @ 10k |
|---|---:|---:|---:|
| **TP=1**, one card, `--ple-backend pinned` | ~26 tok/s | ~26 tok/s | ~9.4 s |
| **TP=2**, two cards, `--ple-backend disk` | **35.0 tok/s** | **34.6 tok/s** | **5.7 s** |

⚠ TTFT is a prefill number and carries the condition *gfx1201, `BLOCK_N` clamped to 32* — the clamp
in §3 is what makes prefill run at all here, and it also makes it slower than an unclamped kernel
would be on hardware that can take one. ⭐ Decode is **flat in context** on this engine: **×1.0089 across 25.7×** of
context. A llama.cpp row serving the same model on the same box decays **×1.665** over 10×.

---

## 5. Build

```sh
git clone --branch rocm-gfx1201 https://github.com/lukascechovic/FreeToken.git
cd FreeToken
docker build -f Dockerfile.gfx1201 -t freetoken-gfx1201:local .
```

That is the whole input. Nothing else is needed — no token, no patch series, no second repository.
The build **copies the tree you checked out**; it does not clone or patch anything.

**What the build asserts, and why it is worth the minute it costs.** The base image is pinned **by
digest**, because `rocm/pytorch:rocm7.14_…` is a mutable tag on someone else's registry and we have
watched a moved tag swap the engine under a running server with nothing erroring. Then the build
checks the **port is present, by content** — the vision tower, the TP>1 shard axes, the disk PLE,
the scheduler work, the kernel clamp (pinned by md5) — and it checks that two things are **absent**:
upstream's TP=1-only refusal, and a live MoE-cache-resizing experiment that cost this row 24.8 % of
decode, silently, and was dropped.

⇒ if you build the wrong branch by mistake, you find out **at build time**, not at load time on a
box with 126 GB of weights already mounted.

⚠ Requirements: Docker, ~55 GB of disk (the image is **50.1 GB**, most of it the ROCm base), and
network for the base image and PyPI. **No GPU is needed to build**, and no GitHub token: nothing is
cloned during the build.
⚠ Build from a plain `git clone`. A source tarball, a GitHub zip, or a `git worktree` checkout is
refused on purpose — without history the image cannot record which commit produced it, and it
prints that record at the end of the build.

---

## 6. Run

Weights are **never** baked into the image. Mount them.

### TP=1 — one card

```sh
docker run --rm --name freetoken \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=<video-gid> --group-add=<render-gid> \
  --ipc=host --shm-size=8g \
  --memory=118g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:False \
  -e FREETOKEN_LOAD_VISION=1 \
  -e FREETOKEN_VIT_GROUP=1 \
  -v /your/models:/models:ro \
  -p 127.0.0.1:8080:1919 \
  freetoken-gfx1201:local \
  ft serve \
    --model /models/Qwen3.8-Flash-Next-NVFP4 \
    --served-model-name qwen3.8-flash-next \
    --moe-backend offload \
    --moe-cache-auto \
    --expert-load parallel \
    --nvfp4-backend triton \
    --ple-backend pinned \
    --max-running-requests 1 \
    --kv-reserve-tokens 262144 \
    --max-prefill-length 4096 \
    --memory-ratio 0.85 \
    --host 0.0.0.0 --port 1919
```

**The three things you must substitute:**

| | |
|---|---|
| `/your/models` | the directory holding `Qwen3.8-Flash-Next-NVFP4`. `--model` is the path **inside** the container. |
| `<video-gid>` / `<render-gid>` | your host's `video` and `render` group ids: `getent group video render \| cut -d: -f3`. They are **not** the same on every distribution — 44 and 991 here. Without them `/dev/kfd` is not usable and the load fails with a permission error that does not name the group. |
| `HIP_VISIBLE_DEVICES` | the card to serve on. §1. |

**Flags worth understanding before you copy them:**

- `--memory-ratio 0.85` — the fraction of VRAM the engine plans for. ⛔ Do not raise it because
  there "looks like" headroom: at the engine default 0.9 a legal seven-image request drove decode
  25.2 → 19.2 tok/s on this box and it never recovered, with 851 MiB free on card 0. The plan is
  reached **at boot**, so a too-high ratio fails at load, not later.
- `--max-running-requests 1` — single-user admission. Raising it raises concurrency **and** divides
  one expert-cache budget among the slots, so it is not a free knob.
- `--kv-reserve-tokens` — the **admission budget**, not a per-prompt limit. `Σ(prompt + max_tokens)`
  across live requests must fit inside it, which means a client that sends a large `max_tokens`
  books all of it up front even if it generates ten tokens.
- `FREETOKEN_VIT_GROUP=1` — bounds the ViT batch. **Inert unless set**; `0`/unset keeps the
  historical all-at-once path, which strands VRAM per distinct image count.
- `--memory=118g` on the container — a host-RAM ceiling, so a runaway request hits a container
  limit instead of the host OOM killer. ⛔ Read §2.2 first: this bounds the blast radius, it does
  not prevent the request.

### TP=2 — two cards

Same as above, plus:

```sh
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e OMP_WAIT_POLICY=PASSIVE \
  ...
    --tp-size 2 \
    --disable-pynccl \
    --ple-backend disk \
    --memory-ratio 0.75 \
```

⛔ `--ple-backend disk` at TP=2 is not optional on a 2 × 32 GiB box — `pinned` does not fit.
⛔ Drop `--memory=118g` for the TP=2 shape; its host-RAM profile is different.
⚠ A cold TP=2 load is ~60 s **with** `OMP_WAIT_POLICY=PASSIVE` and ~19 minutes without it.

### Kernel caches

```sh
  -e XDG_CACHE_HOME=/root/.cache/freetoken-<this-image-id> \
  -e TRITON_CACHE_DIR=/root/.cache/freetoken-<this-image-id>/triton \
  -e FREETOKEN_KERNEL_CACHE_DIR=/root/.cache/freetoken-<this-image-id>/freetoken/kernels \
  -v /host/cache/freetoken-<this-image-id>:/root/.cache/freetoken-<this-image-id> \
```

⛔⛆ Triton and AITER key their caches on the **directory**, not on the image. A directory shared
between two images lets one image execute the other's compiled kernels, and nothing errors. Give
every image its own, and **wipe it when the image id changes**.
⚠ `FREETOKEN_CACHE_DIR` — which you will find in older run scripts, including ours — is read by
**nothing**. The variables that matter are the three above.

---

## 7. Where this branch comes from, and how it is maintained

The commits here are **derived**. They are generated from a patch series and a build recipe kept in
a separate operations repository, and each commit's tree is asserted equal to the corresponding rung
of that series — the branch is checked *against* the thing that builds, never hand-edited into
agreement with it.

Four rules, so that anyone reading a commit here knows what it is:

1. ⭐ **Append-only.** GitHub anchors line-level review comments to commit SHAs; rebuilding the
   branch orphans every one of them onto commits no branch reaches. New work is appended. A rewrite
   is a deliberate, announced act, never a side effect of a re-sync.
2. **Versions are tags, not rewrites.** `rocm-gfx1201` grows forward; a tree that served gets a tag.
3. ⛔ **A rebase onto newer upstream gets a NEW branch.** Upstream `477c8601` (#418) is 172 files,
   +5,575/−5,412, collides with 11 of the applied rungs and *deletes* the file the TP=2 mixer work
   patches. That move is a rewrite against a new abstraction, not a rebase, and it will land on
   `rocm-gfx1201-v2` off the new base. This branch and its tags stay as the record of what served.
4. **Upstream is upstream.** This is a fork branch, not a competing distribution. Where upstream has
   landed the same capability (§3), upstream's is the one to use.

⚠ **What is NOT on this branch:** a multi-token-prediction draft head for this model. It is worth
**+24.2 % decode at TP=1** and a **1.0 % loss at TP=2**, and it lives on **`rocm-gfx1201-mtp`** —
this branch plus one commit, documented in `README.mtp.md` there. ⛔ It is a separate branch on
purpose: this one's whole claim is that its head tree equals the tree of the image the deployed rows
serve, and the MTP row runs *that same image* plus run-time file mounts, so no image anywhere
contains that code. A commit here would break the claim by construction. Upstream **#421** is the
open thread asking for the capability.

---

## 8. Licence and upstream

Everything here inherits FreeToken's licence (`LICENSE`), unchanged. Upstream is
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken); its README is preserved below
this section in `README.md`, and `CONTRIBUTING.md` is upstream's — including its AI policy, which
this work's authors read as: a human must have run the code on real hardware and be able to explain
it to a reviewer unaided.

**Issues about this branch** belong on this fork, not upstream. Issues about FreeToken belong
upstream.
