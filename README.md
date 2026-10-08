<h1 align="center">Strata-GLM</h1>

<p align="center"><b>GLM-5.3-Flash (320B MoE, 3-bit) on one gaming PC</b><br>
A GLM-5.3-Flash engine built on <a href="https://github.com/Niko1221/Strata">Strata</a> and
<a href="https://github.com/Indras-Mirror/Strata-DS4">Strata-DS4</a> · Linux · NVIDIA · MIT</p>

> **Status: early, experimental research code.** It runs the real model end to end and the numbers below are measured,
> but it has been run on exactly one PC, long context (lightning indexer, landed 2026-10-09; a 524K context
> allocates) is only lightly tested, there is no server or chat UI, and quality has only been checked with perplexity. The DS4 README this
> branch grew from is [README.ds4.md](README.ds4.md); upstream Strata's is [README.strata.md](README.strata.md).

## What this is

GLM-5.3-Flash = DeepSeek-V4-style hyper-connections (4 streams, Sinkhorn mixing) and MoE (42 layers x 288 experts,
top-8, sigmoid router), **Kimi Delta Attention** (linear attention, 34 of 45 layers) and nope-MLA with a lightning
indexer (11 layers). The engine (`tools/glm/`) runs the dense half as a ggml graph on the GPU and the routed experts
through Strata-DS4's expert tier (VRAM cache + pinned-RAM arena + CPU pool + PCIe split + NVMe file tier).

Model: `neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF` (113.6 GB; ~107 GB of experts, Q2_K/Q3_K/Q4_K).

## The finding that shapes everything

GLM's routing is **near-flat**: the top 10% of a layer's experts take only ~18% of its routes, and a popularity-ranked
VRAM cache hits no more than its share of experts. Hot-expert caching (what makes Strata fast on Qwen/DeepSeek) cannot
help; what matters is how many experts are **resident at all**. 107 GB of experts do not fit 90 GB RAM + 24 GB VRAM, so:

1. **Exclusive, byte-sized residency**: VRAM slots never duplicate arena experts, and the VRAM cache is filled by bytes
   with the smallest (Q2_K) experts first.
2. **REAP expert pruning** (Router-weighted Expert Activation Pruning, Cerebras 2025): score every expert by router weight
   x output norm on calibration text (`--saliency`), drop the least salient (`tools/glm/reap_prune.py`, `--prune`).
   Pruned experts leave routing, the arena and VRAM, so the rest is fully resident and the NVMe is never read.
3. **Chunked prefill**: with flat routing a prompt chunk reads almost every expert once, so throughput grows with chunk
   size.

## Measured (2026-10-09)

RTX 4090 24 GB · Ryzen 7 5700X (8 cores, AVX2) · 90 GB DDR4 · exFAT NVMe. 2000-token held-out prompts, greedy.

| configuration | prefill tok/s | decode tok/s | quality |
|---|---|---|---|
| first run (cold cache, decode-loop prefill) | 1.7 | 1.75 | - |
| exclusive 72 GiB arena + byte-sized VRAM cache | 86-90 (chunk 1024) | 3.7-4.0 | baseline |
| + REAP prune 25% (fully resident) | **175** | **10.4** | code ppl 3.60 vs 3.68 base |
| + `--skip-miss 0.10` | 175 | **16.4** | +4.0% ppl |

Caveat measured: a static prune list holds only for what it was calibrated on - German text (not in calibration) lost
18-29% perplexity. Calibrate on the languages/domains you use. Details: [docs/glm/FINDINGS.md](docs/glm/FINDINGS.md).

## Build and run

```bash
cmake -B build-glm-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DSTRATA_MMQ_KQUANTS=ON && cmake --build build-glm-gpu --target glm_generate
# third_party/llama.cpp: the ggml copy from Strata-DS4 (gitignored)

M=GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
# calibrate (once): saliency from a few 2000-token texts you care about, then a prune list
build-glm-gpu/glm_generate -m $M --ids-file calib.i32 -n 1 --slots auto --vram-margin 5.5 --arena-gib 72 \
    --arena-skip-resident --prefill-chunk 1024 --chunk-mmq --saliency sal.bin
python3 tools/glm/reap_prune.py sal*.bin --frac 0.25 -o prune.txt
# run
build-glm-gpu/glm_generate -m $M --ids-file prompt.i32 -n 256 --slots auto --arena-gib 72 --arena-skip-resident \
    --prefill-chunk 1024 --chunk-mmq --prune prune.txt [--skip-miss 0.10]
```

Token ids in, token ids out (HF `tokenizers` on the upstream `tokenizer.json`). Load the real model through
`tools/ds4/memguard.sh` (cgroup RAM cap).

## Roadmap

Long-context testing (the lightning indexer is in; KDA layers keep constant state, so 200K-500K is the goal), soft
pruning + on-demand reads (keeps uncalibrated languages), larger prefill chunks,
re-seeding the VRAM cache after prefill, abliteration via an `attn_output` transplant, MTP, a server.

## Credits

Strata (Niko1221), llama.cpp / ggml, neurall/llama.cpp (the glm5-next reference graph), neuralll + pfeifferj (the
GSQ-RCO quant), Z.ai (GLM-5.3-Flash, MIT), REAP (Cerebras).
