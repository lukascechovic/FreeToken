> ## ⚠ You are on branch `rocm-gfx1201-mtp` — `rocm-gfx1201` plus a multi-token-prediction draft head
>
> One commit on top of [`rocm-gfx1201`](../../tree/rocm-gfx1201): a **speculative draft head for
> Qwen3.8-Flash-Next-NVFP4**, 24 files. Everything the parent branch says still applies.
>
> ### 📖 **[README.mtp.md](README.mtp.md) is the documentation for this branch** — read it *after*
> ### **[README.gfx1201.md](README.gfx1201.md)**, which carries the three mandatory settings.
>
> **It will not load without two things the parent branch does not need:**
>
> ```
> -e FREETOKEN_LOAD_MTP=1          # plus six more — README.mtp.md §2
> -e FREETOKEN_MTP_BANK=/…/mtp-experts-nvfp4.safetensors
> ```
>
> ⛔ That 1.42 GB bank is **not published**. You build it from your own checkpoint with
> `tools/make_mtp_bank.py` — README.mtp.md §3. A row without it does **not** fall back.
>
> ### ⛔ What it is worth, both directions
>
> - **+24.2 % decode at TP=1** (one card, ~12k context, batch size 1). ⛔ **A 1.0 % loss at TP=2**,
>   and the two shapes' numbers may not be carried onto each other.
> - **Not lossless: 1 flip in 602 generations.** Chased to closure — it is bf16 rounding in the
>   verify path, not a correctness bug — but this branch **emits different text** from the
>   non-MTP rows. Not wrong; different.
> - ⛔ **No concurrency claim at all.** Every number is batch size 1; bs>1 is unmeasured.
>
> ⚠ No quality or fidelity claim is made anywhere: this work has no fidelity instrument.
>
> ---
>
> *The `rocm-gfx1201` banner follows — it describes the branch this one is cut from, and all of it
> applies here. Upstream's own README follows that, unchanged.*

> ## ⚠ You are on branch `rocm-gfx1201` — a fork branch, not upstream FreeToken
>
> This branch adds **AMD RDNA4 (`gfx1201`, Radeon AI PRO R9700) support for
> Qwen3.8-Flash-Next-NVFP4**: the ROCm kernel work that makes it serve at all, tensor parallelism
> across two cards, and image input. It is 19 commits on upstream
> [`4b94bdc3`](https://github.com/FlashML-org/FreeToken/commit/4b94bdc38a46a4dfe534e8793126160d56904c44).
>
> ### 📖 **[README.gfx1201.md](README.gfx1201.md) is the documentation for this branch.** Read it before running anything.
>
> **Build:** `docker build -f Dockerfile.gfx1201 -t freetoken-gfx1201:local .`
>
> **Three settings are mandatory and nothing tells you when one is missing:**
>
> ```
> --expert-load parallel                        # `auto` picks serial here => ~100x slower load
> -e PYTORCH_ALLOC_CONF=expandable_segments:False   # :True GPU-FAULTS on gfx1201 — and torch's
>                                                   #   own OOM message advises :True
> -e HIP_VISIBLE_DEVICES=0                      # or 0,1 for --tp-size 2 — always explicit
> ```
>
> ### ⛔ Known defects — the two that can hurt you
>
> - **Image token positions are wrong on this branch**, and so is everything after them. The
>   checkpoint declares M-RoPE; this branch implements none of it and feeds 1-D positions. Nothing
>   errors and the answer reads fine. **Text-only requests are provably exact.**
>   ⭐ Upstream merged a correct M-RoPE implementation in
>   [#454](https://github.com/FlashML-org/FreeToken/pull/454) on 2026-09-13 — for image *quality*,
>   upstream's vision path is better than this one.
> - **Nothing bounds the number of images in one request.** A single legal request can exhaust
>   **host** RAM and take the machine down mid-request. The existing caps bound the *sum* of soft
>   tokens, not the image count. See README.gfx1201.md §2.2 for the mitigations.
>
> ⚠ TP=2 + images is the least-tested combination here. No quality or fidelity claim is made
> anywhere: this work has no fidelity instrument.
>
> ---
>
> *Upstream's own README follows, unchanged.*

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/xzwSnMdsX"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
