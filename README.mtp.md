# The MTP draft head — branch `rocm-gfx1201-mtp`

This branch is [`rocm-gfx1201`](README.gfx1201.md) **plus one commit**: a multi-token-prediction
draft head for **Qwen3.8-Flash-Next-NVFP4**, 24 files, 16 modified and 8 added. Everything the
parent branch says still applies — ⛔ **read [`README.gfx1201.md`](README.gfx1201.md) first**, it
carries the three mandatory settings and the defects that can hurt a reader, and none of that is
repeated here.

What this file adds is the three things that are only true on **this** branch: what the head is
worth, the seven environment variables that turn it on, and the 1.42 GB file you have to build
before it will load.

- **Build:** the parent's `Dockerfile.gfx1201`, unchanged — §4.
- **Run:** §2 for the environment, §3 for the expert bank. The bank has **no default** and a row
  without it does not fall back.
- ⛔ **Honest status:** §1 — ⛔⛆ and it was **corrected on 2026-09-15**: read **§1.0 before quoting
  any number from this file**. Short version: at **TP=2 it is a ~+12 % decode win** on served
  traffic, and at **TP=1 the headline +24.2 % is a like-for-like figure the deployment does not
  get**. Either way it emits different text from the non-MTP rows.

---

## 1. ⛔ What it is worth, and what it costs — measured, both directions

### ⛔⛆ 1.0 CORRECTION, 2026-09-15 — both headline numbers are LIKE-FOR-LIKE, and the deployment reading differs

⭐ Nothing below is withdrawn. §1.1 and §1.2 are arithmetically intact and each is stated against a
control differing in **one** environment variable. What they are **not** is the answer to *"should I
turn this on"* — and until this correction the file read as though they were.

| | §1.1 / §1.2 say | the deployment reading |
|---|---|---|
| **TP=1** | +24.2 % | ⛔ **−9.3 %.** Both arms ran `--ple-backend disk`, because `pinned` + MTP **does not fit** at TP=1 — the bank pins **full width in one process** (each TP=2 rank pins half). So the real choice is `disk`+MTP at **21.82 tok/s** against `pinned` **without** MTP at **~24**. Break-even needs the disk tax under **19.53 %**; it measures **27 %**. |
| **TP=2** | −1.0 % | ⭐ **+12.2 %.** The −1.0 % is **α 0.6062's** number, measured on two prose cells. A census of the live row — 6 requests, **21,303** checked draft steps — reads α **0.6363 … 0.9363**, pooled by token **0.8153**, every request clearing break-even **0.618**. At the measured step cost that is **+12.2 %**, and the traffic this row serves sits **at or above** the pooled figure ⇒ read it as a **floor** (~+20 % at the top of the census range). |

⛔ **What kind of number +12.2 % is:** step cost × acceptance, both measured on matched arms in one
session — **not** an A/B of served traffic. The only direct decode A/B at TP=2 *is* the −1.0 %, and
the same arithmetic reproduces it at that acceptance. ⇒ the model and the measurement agree; what
moves between them is **α**, and a prose corpus understates what real traffic accepts.

⛔⛆ **The rule this file now carries:** a decode number means nothing without **the α it was taken
at** and **the PLE backend both arms ran**. Quote §1.1 or §1.2 **with §1.0, or not at all**.

### 1.1 ⭐ TP=1, one card: **+24.2 % decode** — ⛔ `disk`-vs-`disk`, not the deployment choice (§1.0)

Four loads, MTP on versus MTP off, same row, same box, 2026-09-14:

| | ms/step | tok/s | tok/forward |
|---|---:|---:|---:|
| **MTP on** | 74.77 / 75.27 | **21.93 / 21.57** | 1.6398 / 1.6235 |
| **MTP off** | 56.95 / 56.93 | 17.56 / 17.57 | 1.0000 / 1.0000 |

**+24.2 %** overall, +13.1 … +30.8 % per paired cell. The step-cost ratio is **`S` = 1.3152**
(1.2392 … 1.3986 over 48 pairings) ⇒ **break-even acceptance is 0.3152**, and the row serves
**α ≈ 0.63** — roughly twice what it needs.

⚠ Conditions, because the number does not travel without them:

- It is a **~12k-context reading** (11,431–12,985 prompt tokens, 512 out, the served sampler).
  Decode decays with context on this engine, so `S` = 1.3152 is **not** a whole-range constant.
- It is the **deployed-shape** step cost, not a pure compute ratio: loading the head also moves the
  MoE cache (3,627 slots off → 3,367 on) and the KV pool (6.19 → 6.70 GiB). Those belong inside
  MTP's cost here.
- ⭐ The control is provably a control: the **same 24 files stay in the tree** with only
  `FREETOKEN_LOAD_MTP=0`, so nothing the port changes is charged to MTP by accident. The off arm
  measured exactly **1.0000 tok/forward over 11,280 forwards**.

### 1.2 ⛔ TP=2, two cards: a **1.0 % loss** at α 0.6062 — ⛔⛆ and the served row runs higher (§1.0)

Measured separately, with the served sampler: **0.9901×**, on a probe corpus whose two scored cells
ran α **0.6036** and **0.6088** — below TP=2's break-even of **0.618**. ⛔ The two shapes have
**different** step-cost ratios and you may not carry one onto the other: TP=2's `S` = 1.6222 applied
to §1.1's acceptance rate predicts **+0.6 %**, i.e. it would report the +24 % TP=1 row as break-even.

⛔⛆ **What this figure is NOT: the served row's gain.** It is what the row does *at α 0.6062*, and
the live row does not run there — see **§1.0**. A census of its own `[#801] alpha` lines over ~2 h of
ordinary use (6 requests, 21,303 checked draft steps) reads **0.6363 … 0.9363**, pooled by token
**0.8153**, with **every** request above break-even ⇒ **+12.2 %**, worst single real request
**+1.1 %**. ⚠ The third cell of that probe — a coding task, α **0.6641**, the only one of the three
clearing break-even — was dropped by a zlib content filter for missing its threshold by **0.0093**.
⇒ the −1.0 % was measured on the two cells **least** like served traffic.

### 1.3 ⛔ No concurrency claim at all

Every number here is **batch size 1**. MTP at bs>1 is **unmeasured** on this branch, and the row it
was measured on refuses concurrency by design. ⛔ Do not read §1.1 as a throughput figure for a
loaded server.

### 1.4 ⛔⛆ It is not lossless — **1 flip in 602**, and here is the other half

Against the non-MTP arm, greedy, the draft-and-verify path produced a different token **once in 602
generations**. Both halves of that finding travel together or neither does:

- ⛔ **the flip is real** and it means this row's text is *different* from the non-MTP rows'. Not
  *wrong* — *different*. If you need bit-identical continuations, do not enable the head.
- ⭐ **it is not a defect in the port.** It was chased to closure: the divergence is **bf16 rounding
  in the verify path**, and the Per-Layer Embedding table — the component under suspicion — is
  **bit-exact at T = 2 on 58 of 58** probes. There is no known correctness bug behind it.

⛔ **No quality or fidelity claim is made anywhere in this file**, exactly as on the parent branch:
this work has no fidelity instrument. Every number above is throughput or an exact-match count.

---

## 2. ⛔ The seven environment variables the row sets

The head is **off** unless you turn it on, and three of these seven are load-bearing in a way that
fails silently rather than loudly. This is the whole supported surface:

```sh
-e FREETOKEN_LOAD_MTP=1
-e FREETOKEN_MTP801_SHADOW=1
-e FREETOKEN_MTP801_VERIFY=1
-e FREETOKEN_MTP801_RUN_HEAD=0          # ⛔⛆ ZERO. Not a typo. See below.
-e FREETOKEN_MTP801_SPECCHECK=8
-e FREETOKEN_MTP801_HEAD_DTYPE=nvfp4
-e FREETOKEN_MTP_BANK=/path/in/container/mtp-experts-nvfp4.safetensors
```

| | |
|---|---|
| `FREETOKEN_LOAD_MTP=1` | **The switch.** Loads the head as a 49th offload layer: 29 dense tensors plus 512 routed experts, read from the bank in §3. ⚠ There is no `--spec-type` flag on this engine — this variable *is* the speculation switch, and a tool that looks for a command-line flag will report MTP as off on a row where it is on. |
| `FREETOKEN_MTP801_SHADOW=1` | The real draft step and the cross-step tracker. It is what **produces** a draft. |
| `FREETOKEN_MTP801_VERIFY=1` | The real draft → verify → commit path: what **consumes** the draft. ⛔ `SHADOW=1` is mandatory alongside it. ⛔⛆ Without `VERIFY` the head still loads and is still forwarded, but **no draft is ever consumed** — the row pays every cost of MTP and collects none of its benefit, and nothing says so. |
| `FREETOKEN_MTP801_RUN_HEAD=0` | ⛔⛆ **Zero is correct and is not a typo.** `RUN_HEAD=1` is an earlier round's stand-in forward whose output is discarded. With `SHADOW=1` the head is already forwarded for real, so leaving `RUN_HEAD=1` runs the head's forward **twice per step** and nothing errors — you simply pay double for the draft. |
| `FREETOKEN_MTP801_SPECCHECK=8` | Cross-rank agreement on the first 8 speculating steps of each load. ⚠ **TP=2 only, and keep it:** if the ranks ever disagree on `accepted_len` the row wedges on NCCL's 60 s watchdog with the cause long scrolled out of the log, and this is the only instrument that would see it. It costs no decode figure. ⛔ At **TP=1 it declines by design** — `{"ran": false, "declined": "tp_size 1: one rank cannot disagree with itself"}` is the **healthy** reading, and a check asserting `"ran": true` fails a working one-card row. |
| `FREETOKEN_MTP801_HEAD_DTYPE=nvfp4` | The head's expert dtype. `nvfp4` is already the default; it is set explicitly so the row records which bank it expects. `bf16` bypasses the bank and the offload cache entirely and is a research path, not a served one. |
| `FREETOKEN_MTP_BANK` | Absolute path, **inside the container**, to the file §3 builds. ⛔ No default — see §3. |

⚠ **Everything else is a research dial, not a supported setting.** The 24 files read **sixteen
further `FREETOKEN_MTP801_*` names**. Most are instruments that default to off and cost at least a
second forward per step; ⚠ **two are not on/off at all** — `VERIFY_GRAPH_MAX_BS` defaults to **4**
and sizes the verify graph set, and `CHECK_DIR` is a directory path. They exist because this port
was developed by measuring it. ⛔ Do not set any of them on a serving row.

### ⭐ Confirm it actually speculated — two log witnesses

⛔⛆ **A row that quietly never speculates passes every text check perfectly.** That happened twice
during development, so check the log rather than the output:

```sh
docker logs <container> 2>&1 >/dev/null | grep '\[#801\]'
```

- `[#801] alpha {…"checked": 35, "accepted": 26, "rate": 0.7429…}` — **the draft is being
  consumed.** `checked` > 0 is the thing to look for. This witness survives at **both** TP.
- `[#801] speccheck:` with `"ran": true, "agreed": true` and **zero**
  `[#801] speccheck DISAGREEMENT:` lines — ranks in lockstep. ⛔ **TP=2 only**; at TP=1 see the
  `SPECCHECK` row above.

⛔⛆ **Both witnesses are on `stderr` while the geometry lines are on `stdout`.** Grepping one
stream reads the other's witnesses as *absent*, which is exactly how a non-speculating row passes
inspection. The redirect above is not decoration.

⭐ One more structural tell, on stdout: a load that has the head captures graphs **twice** —
`Capturing graphs (decode)` *and* `Capturing graphs (verify T=2)`. ⛔ A load with only the first
pass has not loaded the head, whatever else it printed.

---

## 3. ⛔ The expert bank — 1.42 GB, you build it, it is not published

The head needs one derived file that is in neither the checkpoint nor the image:

```
mtp-experts-nvfp4.safetensors   1,419,510,312 B
```

⛔ **It is deliberately not published as bytes.** It is derived from the checkpoint's own `mtp.*`
tensors — the roughly one third of the weights that the engine's weight loader drops on load — so
shipping it would raise a redistribution question in exchange for nothing the branch cannot rebuild
for you. `tools/make_mtp_bank.py` rebuilds it in pure python and numpy: **no torch, no safetensors
library**, about **152 s of one CPU core**.

```sh
python3 tools/make_mtp_bank.py /path/to/Qwen3.8-Flash-Next-NVFP4 --plan
python3 tools/make_mtp_bank.py /path/to/Qwen3.8-Flash-Next-NVFP4 --out /path/to/mtp-bank
```

- ⭐ **Run `--plan` first.** It reads nothing but the checkpoint's index and writes nothing, and it
  predicts the artefact **to the byte** — 1,419,510,312 — because it takes that number from the
  writer's own header code rather than from arithmetic beside it. If `--plan` disagrees with the
  size above, stop: your checkpoint is not the one this was built against.
- ⛔⛆ **The model path is required and there is no default.** That is deliberate, and it is the one
  place this recipe differs from the research CLI it wraps: a default here would go looking for one
  particular machine's filesystem and report a missing file, which reads as *your* setup being
  broken rather than as the tool assuming something it had no business assuming.
- The default granularity is **`row`**, which is what the deployed bank was built at and what every
  number in §1 was measured with. ⚠ `--granularity` exists; changing it gives you a bank no
  measurement in this repository was taken at.

Then mount it read-only and point the row at it:

```sh
  -v /path/to/mtp-bank/mtp-experts-nvfp4.safetensors:/ft801-bank/mtp-experts-nvfp4.safetensors:ro \
  -e FREETOKEN_MTP_BANK=/ft801-bank/mtp-experts-nvfp4.safetensors \
```

⛔⛆ **A missing bank is not a fallback.** The weight loader refuses to guess a path, and it refuses
for a reason: the checkpoint's own `mtp.layers.N.mlp.experts.*` tensors are stacked bf16 and are
dropped on every load path, so a head without its bank is not a head running slowly — it is a head
whose MoE reads whatever the expert cache slab happened to hold.

⚠ `tools/head_801.py` and `tools/nvfp4_801.py` beside it are **derived copies** of llm-server's
research tree. Regenerate them there; never edit them on this branch.

⛔⚆ **Run `make_mtp_bank.py`, never those two directly.** They are research CLIs and their model
argument is **optional**, defaulting to one particular machine's directories
(`/home/luka/models/…`, `/home/luka/freetoken-mtp801/mtp-bank`). On your machine that does not
refuse — it goes looking for somebody else's filesystem and reports a missing file, which reads as
*your* setup being broken. The defaults stay because these files are byte-identical copies and a
divergence here would be worse; the recipe above is the supported entry point, and it has no
defaults at all. ⚠ `nvfp4_801.py --write` also sweeps all **three** granularities — three 4.8 GiB
reads — and writes only the last.

---

## 4. Build and run

The build file is the parent branch's and is **unchanged** — this branch simply has 24 more files
in the tree it copies:

```sh
git clone --branch rocm-gfx1201-mtp https://github.com/lukascechovic/FreeToken.git
cd FreeToken
docker build -f Dockerfile.gfx1201 -t freetoken-gfx1201-mtp:local .
```

The run command is [`README.gfx1201.md`](README.gfx1201.md) §6's, plus §2's seven variables and
§3's bank mount. The one-card shape the +24.2 % was measured on:

```sh
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:False \
  -e OMP_WAIT_POLICY=PASSIVE \
  …
  ft serve \
    --model /models/Qwen3.8-Flash-Next-NVFP4 \
    --moe-backend offload --moe-cache-auto --expert-load parallel \
    --nvfp4-backend triton \
    --max-running-requests 1 \
    --kv-reserve-tokens 262144 \
    --ple-backend disk \
    --max-prefill-length 4096 \
    --memory-ratio 0.85 \
    --host 0.0.0.0 --port 1919
```

⚠ Worth knowing before you copy it:

- `--max-running-requests 1` is **not** conservatism here. At `mr=1` the captured graph set is
  `[1]`, so a batch can never transit size 3 — the size that pads to 4 and kills the row inside the
  draft head's KV store. ⛔⛆ Raising the server's admission limit without raising this one is not a
  mismatch, it is a crash.
- `--memory-ratio 0.85` on this shape left **4.05 GiB free after graph capture**. ⛔ Do not raise it
  looking for headroom: the plan is reached **at boot**, and a ratio of 0.90 on a sibling row took a
  live vision-encode failure 30 minutes into a serving load. ⛔ And do not act on torch's own
  out-of-memory advice — it recommends `expandable_segments:True`, which **GPU-faults** on
  `gfx1201`. See the parent README §1.
- ⚠ `--ple-backend disk` is what the +24.2 % was measured against. The parent's §2.4 note that
  `disk` costs ~27 % of decode at TP=1 is a **different** comparison and the two axes were never
  separated in one experiment — so ⛔ do not add the two numbers together in either direction.
- ⭐ A cold load reaches health in about **61 s** at TP=1.
- ⚠ **No image request has been run on this shape.** The head is text-path work; the vision defects
  in the parent README §2 are unchanged by it, and untested alongside it.

---

## 5. Where this branch comes from

⛔ **This branch is `rocm-gfx1201` plus one commit, and that commit is derived output, not
hand-written.** It is generated from the same patch series and build recipe as the rest of the
ladder, kept in a separate operations repository, and its tree is asserted equal to what the tooling
derives from the 24 source files — never edited into agreement with it.

Two of the 24 files are themselves generated by anchor-rewriting scripts whose anchors must each
occur exactly once, so a drift in the base raises rather than landing somewhere plausible. ⇒ ⛔ **do
not hand-edit this commit.** Regenerate it.

**Why a separate branch rather than a rung on `rocm-gfx1201`:** the parent branch's single claim is
that its head tree equals the tree of the image the deployed rows actually serve. The MTP row runs
**that same image** plus run-time file mounts, so no image anywhere contains this code — an MTP
commit on the parent would break its claim by construction, not by preference. Keeping it here
leaves both claims clean: `rocm-gfx1201` reproduces what the vision rows serve, and
`rocm-gfx1201-mtp` reproduces what the MTP row serves.

The parent branch's four maintenance rules (append-only, versions are tags, a rebase onto newer
upstream gets a new branch, upstream is upstream) apply here unchanged — see
[`README.gfx1201.md`](README.gfx1201.md) §7.

⭐ **Upstream has an open thread asking for exactly this capability**: FreeToken **#421**. Nothing
here is upstream, and if it lands there, upstream's is the one to use.

---

## 6. Licence and upstream

Unchanged from the parent branch — FreeToken's licence (`LICENSE`), upstream is
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), and its README is preserved
verbatim below the banner in `README.md`.

**Issues about this branch** belong on this fork, not upstream.
