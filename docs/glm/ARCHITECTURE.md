# GLM-5.3-Flash architecture deep dive: what to reuse, what to build, what to skip

2026-10-08. Sources: `zai-org/GLM-5.3-Flash/config.json` + `model.safetensors.index.json` (copies in
`bench/glm-2026-10-08/upstream/`), the reference implementation `neurall/llama.cpp` `src/models/glm5-next.cpp`
(clone at `~/AI/llama.cpp-glm53`, commit `2e0435a`, read in full), and the target GGUF's `tensor-types-q4kattn.txt`.
Numbers marked *estimate* are arithmetic, not measurements.

## Headline
GLM-5.3-Flash = **DeepSeek V4's mHC and MoE** + **Kimi's KDA linear attention** (34 layers) + **DeepSeek V3.2-style
MLA with a lightning indexer** (11 layers). Strata's ggml copy, which DS4 builds against
(`third_party/llama.cpp/ggml`, ops from llama.cpp #25585 etc.), **already has every op the model needs**:
`ggml_dsv4_hc_{pre,post,comb}` (fused mHC), `ggml_gated_delta_net` with the KDA path (per-channel decay; CUDA
`gated_delta_net.cu` has the `kda` branch), `ggml_ssm_conv`, `ggml_lightning_indexer`, `ggml_flash_attn_ext`, top-k,
`set_rows`/`get_rows`, `ggml_swiglu_split`. So the dense half is a new ggml graph builder (`GlmDense`, modeled on
`Ds4Dense`) with almost no new kernels. The experts are `Ds4MoeTier` unchanged.

## Layer by layer (formulas from the fork; tensor names = GGUF)

### Streams (mHC), all 45 layers - IDENTICAL to DS V4
- Embedding repeated into 4 streams: `x[4][4096]`.
- Before attention and before FFN: `mixes = hc_fn · rmsnorm(flatten(x))` (24 = (2+4)*4 numbers), then
  `pre = sigmoid(mixes[0:4]*scale0 + base[0:4]) + eps`, `post = 2*sigmoid(mixes[4:8]*scale1 + base[4:8])`,
  `comb = sinkhorn(mixes[8:24]*scale2 + base[8:24])` (row softmax, +eps, 20 alternating column/row normalizations).
  Input to the block = `sum_s pre[s]*x[s]`; after the block `x'[d] = post[d]*out + sum_s comb[d][s]*x[s]`.
- Same functions, same fused ops as `Ds4Dense` (`ds4_dense.cpp` ~L515-566). **Reuse verbatim.**
- **Difference from DS V4: the head.** GLM collapses the streams with a plain **mean of the 4 streams**, then
  `output_norm`, then `output`. No `hc_head` tensors (DS V4 has `hc_head_*`). Simpler.

### KDA (Kimi Delta Attention), layers 0-2, 4-6, ... (34 layers, every layer not ≡ 3 mod 4)
Per token, d_inner = 64 heads x 128 = 8192:
1. `q,k,v = silu(causal_conv1d_k4(W{q,k,v} · x))`, a separate depthwise conv per q/k/v (`ssm_conv1d_{q,k,v}`,
   kernel 4: 3 tokens of conv state per channel).
2. Per-channel log-decay: `g = -5 * sigmoid( exp(A_log)[h] * (f_b · (f_a · x) + dt_bias) )` - f_a 4096->128,
   f_b 128->8192 (low rank), `ssm_a` holds `-exp(A_log)` per head, `gate_lower_bound -5` -> g in (-5, 0).
3. `beta = sigmoid(ssm_beta · x)` per head (64).
4. `q,k` L2-normalized (eps 1e-6), then the gated delta rule on a 128x128 state per head:
   `S = diag(exp(g)) S;  S += beta * k ⊗ (v - S^T k);  o = S^T q` (scale per FLA convention - check against
   `ggml_gated_delta_net`).
5. Output gate: `o = rmsnorm_per_head(o) * ssm_norm * sigmoid(g_b · (g_a · x))`, then `attn_output` 8192->4096.
- State: 64 x 128 x 128 f32 = 4 MiB per layer, 136 MiB for all 34, **constant in context length**. Plus conv state
  3 x 3 x 8192 floats per layer.
- Code: Strata has GDN kernels of its own (`src/kernels/cuda/{gdn,fused_gdn,native_gdn}.cu`, Qwen-Next, scalar
  decay); the ggml op already does KDA. **Decision: use `ggml_gated_delta_net` (KDA path) inside the GlmDense
  graph**, exactly as the fork does; consider a fused Strata kernel only if nsys shows KDA matters (decode: 34 x
  4 MiB state read+write ~ 0.3 ms at 1 TB/s *estimate* - it won't).
- Decode cost of the projections per KDA layer: q,k,v,o 4 x 4096x8192 Q4_K (~76 MB) + small low-rank/bf16 pieces.
  34 layers ~2.6 GB of weight reads per token -> **~3 ms at ~900 GB/s** *estimate*. This is the bulk of the dense
  half.

### DSA / MLA, layers 3, 7, ..., 43 (11 layers)
1. `qr = rmsnorm(q_a · x)` (1536), `q = q_b · qr` -> 64 heads x 256, **no RoPE** (`qk_rope_head_dim 0`,
   `mla_use_nope`). No positional encoding at all in these layers (the KDA convs/recurrence carry order).
2. `kv = rmsnorm(kv_a_mqa · x)` (512 latent) -> cached (K-only cache: the latent is both K and V).
3. Absorb: `q' = k_b · q` per head (256 -> 512); scores `q' · kv_latent / sqrt(256)`; softmax; `o = v_b · (p · kv)`
   per head (512 -> 256); `attn_output` 16384 -> 4096.
4. **Lightning indexer** (all 11 layers are "full" - each has its own): index key per token
   `ik = layernorm(attn_k · x)` (128, with bias), gate `ig = kpool_gate · x`; keys are **pooled in groups of 4
   consecutive tokens** (softmax over the 4 of `ig + ape`, weighted sum of the 4 keys); index query
   `iq = attn_q_b · qr` (32 heads x 128); pool score = `sum_h w_h * relu(iq_h · pooled_k)` with
   `w = proj · x / sqrt(128*32)`; keep the top 512 pools (= 2048 tokens) + the incomplete tail (<= 3 tokens);
   attention runs only over those.
- **Exact simplification for short contexts:** with <= 2048 + 3 tokens of context every pool is selected, so
  attention = dense MLA. Phase 1 builds dense MLA (gate vs the fork on prompts < 2K); the indexer comes in
  phase 2 (needed for long context, and for speed beyond ~8K).
- Cache per token per layer: 512 latent (f16: 1 KiB) + indexer row 3 x 128 f32 (1.5 KiB). 11 layers: ~28 KiB/token
  -> 32K context ~0.9 GiB, 128K ~3.5 GiB of VRAM *estimate*. (DS V4's compressed cache is smaller per token; GLM's
  KDA layers keep nothing per token.)

### FFN
- Layers 0-2: dense SwiGLU, 12288 wide, Q4_K (~85 MB each, VRAM).
- Layers 3-44 (42 layers): router `ffn_gate_inp` f32 4096x288, **sigmoid** scores, `+ exp_probs_b` for selection
  only (noaux_tc, n_group 1), top-8, weights normalized then x **2.5**; **clamped SwiGLU, limit 10** (experts and
  shared expert); 1 shared expert 2048 wide (Q4_K). Same router/clamp code paths as DS V4 (DS V4 uses sqrt-softplus
  scoring, so the scoring function is a switch - check `ds4_dense.cpp` router).
- Expert blobs: 288 x 42 = 12,096 experts, gate/up/down 4096<->2048. Q2_K in 36 layers, Q3_K in 3, Q4_K in 3 (by the
  types file; RCO chose them). ~8.3 MB per Q2_K expert, ~100 GB in all *estimate* (exact from the GGUF once it lands).

### MTP
Config has 1 NextN layer (a DSA layer + MoE). Our GGUF has none; `neuralll/GLM-5.3-Flash-MTP-GGUF` (4.6 GB, Q4_K,
`-md` in the fork) is the block alone. neuralll measured **depth-1 acceptance 70-75% but MTP SLOWER** (20.2 -> 17.6-18.4
t/s) on 2x3090 with 30% of expert work on the CPU. That matches our speculation economics (MiMo s16: one draft
needs ~71% acceptance just to break even when expert misses dominate). **Skip MTP/DFlash2 until the single-pass is
fast**; revisit only if most experts end up VRAM/RAM-resident.

## Memory plan (4090 24 GB, 90 GB RAM) *estimate*
| what | where | size |
|---|---|---|
| attention + dense FFN + shared experts + routers + indexers (Q4_K / bf16 / f32) | VRAM | ~5-6 GB |
| output head (Q4_K, 154880 x 4096) | VRAM | ~0.36 GB |
| token embedding (Q8_0) | RAM (get_rows on CPU, upload one row) | ~0.67 GB |
| KDA state + conv | VRAM | ~0.14 GB |
| MLA latent + indexer cache, 32K ctx | VRAM | ~0.9 GB |
| compute buffers, CUDA graphs | VRAM | ~1-2 GB |
| **expert slots** | VRAM | **~14-15 GB = ~1,750 Q2_K experts (14%)** |
| expert arena | pinned RAM | 66-70 GiB = ~8,500 experts (70%) |
| rest (~16%) | NVMe file tier (exFAT NVME1TB) | |
So ~86% of experts in VRAM or RAM - a better ratio than MiMo (115.7 GB file) and DS4. File-tier reads should be rare
once the cache is seeded.

## Decode budget per token *estimate*
- Dense half: KDA projections ~3 ms + MLA ~1 ms + dense FFN/shared ~1.5 ms + head ~0.5 ms + launch overhead (with
  CUDA graphs, DS4 got attention+router to 17.9 ms for 43 layers - kernel count, not bandwidth, dominated). Budget
  15-20 ms.
- Experts: 42 x 8 = 336 x ~8.3 MB = **~2.8 GB/token**. At a ~50% VRAM hit rate (DS4 got 72% with LRU), ~1.4 GB
  goes through the CPU pool + PCIe at ~30 GB/s -> ~45 ms; at 72% ~0.8 GB -> ~26 ms.
- Total ~45-65 ms -> **~15-22 tok/s**. neuralll's fork: 21.5 t/s on 2x3090 (48 GB VRAM). The first real number
  to get is the fork on THIS box (expected below 21.5 with half the VRAM).

## Decisions
1. **Branch from DS4, new `tools/glm/`** (`glm_dense.{cpp,hpp}`, `glm_generate.cpp`, `glm_ref` later) that links
   `tools/ds4/ds4_moe.*` unchanged. Copy the mHC/router/shared-expert/graph-reuse pieces of `Ds4Dense` rather
   than templating Ds4Dense (it is DS-V4-specific: compressed windows, sinks, RoPE, hc_head).
2. **ggml graph for everything dense**, ops 1:1 with the fork's graph so the fork's intermediate tensors
   (`cb()` names like `kda_scan_out`, `hc_attn_post`, `l_out`) can be dumped and compared layer by layer.
3. **Oracle = the fork itself.** Dump per-layer `l_out` (all 4 streams) and logits with `GGML_SCHED` eval callback
   (`llama-eval-callback` / `tools/ds4/golden_dump.cpp` pattern) on 3-4 prompts. Mini-fixture CPU gates as for DS4
   (`make_mini_gguf.py` for glm5-next: few layers, few experts) so most iteration needs no full-model load.
4. **Phase 1 = dense MLA (no indexer)**, exact for contexts <= 2051 tokens; phase 2 = indexer.
5. **LoRA at run time, rank 1**: `down_proj` of routed experts -> add `B·(A·h)` after the expert (CPU and GPU paths
   in Ds4MoeTier, A/B per (layer, expert) from the adapter GGUF), shared `down` and `attn_output` in the ggml graph
   (`ggml_mul_mat` with A then B). Check the fork's LoRA results for the gate (KDA `o_proj` LoRA needs the
   heretic-gguf one-line patch in the fork).
6. **Speed later, in this order:** CUDA graphs + single readback per layer (DS4 lessons), `--vram-lru`, chunked
   MMQ prefill (Q2_K/Q3_K/Q4_K already supported), route-bias, then indexer for long context. MTP/DFlash2 last.
7. Alternative if decode disappoints: `patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF` (50% of experts pruned, Q3_K_M
   78.8 GB - would sit entirely in RAM+VRAM) - quality cost unmeasured; OpenMOSE's REAP-250B (~25% pruned) publishes
   KL/top-1 tables. Not now: Mal picked GLM for quality.

## Open questions (answer from the fork/HF code before coding)
- `ggml_gated_delta_net` KDA scale convention (q scaled by 1/sqrt(128)?) and state layout (k-major?).
- `exp_probs_b` / `expert_weights_scale 2.5` / `norm_topk_prob` order in `build_moe_ffn` for sigmoid gating.
- Chat template (`upstream/chat_template.jinja`) and stop tokens 154820/154827/154829.
- Expert blob layout in this GGUF: gate/up separate (yes: `ffn_{gate,up,down}_exps`), same type per layer?
- Vision (`model.visual.*`, 24 blocks): upstream only; an mmproj exists (`orcarouter`/REAP repos, 1.1 GB F16) if we
  ever want GLM image input.
