# Strata-GLM: port plan for GLM-5.3-Flash

Written 2026-10-08 at branch `glm` = `deepseek4` `a693545` (no GLM code yet). Facts below come from
`zai-org/GLM-5.3-Flash/config.json` and the target GGUF's `tensor-types-q4kattn.txt`; nothing here is measured on
the real model yet. Re-check every architectural claim against the reference implementation before coding it.

## Target files (on `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/`)
| file | what |
|---|---|
| `GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf` | 113.6 GB (105.8 GiB), `neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF`, sha256 `5f03b74f...138f9d6d`. Experts = pfeifferj GSQ-RCO 3.0-bit; non-expert Q8_0 -> Q4_K (+0.95% ppl, 7-16% faster decode in their fork). |
| `tensor-types-q4kattn.txt` | every tensor name and its type (the inventory below comes from it) |
| `lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf` | **default uncensor adapter.** Rank-1 F16 LoRA (Abliterix SRA): routed-expert `down_proj` layers 3-28 (all 288 experts), shared-expert `down_proj` 18-44, attn `o_proj` 15-44. StrongREJECT refusal 1.67%, lower KL than v1. |
| `lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered-LoRA-v1.gguf` | v1: same modules, routed experts in layers 18-44. Refusal 3.33% (StrongREJECT), 1% (SimpleSafety). |
| `lora/heretic/glm-5.3-heretic-lora.gguf` | MorinoNushi heretic-gguf trial 8 (28 MB): refusal 95% -> 26%, KL 0.068. Weaker, early. |
| `download.log` | the download's log (exFAT volume, `FORCE_EXFAT=1` by Mal's choice) |

Download: `FORCE_EXFAT=1 bash ~/AI/download_glm53_flash.sh /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO`
(resumable; re-run the same command after an interruption; sha256 check at the end).

No uncensored GSQ-RCO quant fits (GCSA-AiLab's is 137 GB), hence the runtime LoRA.

## Architecture (config.json, `glm5_next_text`)
- 45 layers, hidden 4096, vocab 154880, untied output, rms_norm_eps 1e-5, max positions 1M.
- **mHC** (hyper-connections): `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps 1e-6`; tensors `hc_{attn,ffn}_{base,fn,scale}`
  per layer. Same family as DeepSeek V4's mHC: reuse `tools/ds4/ds4_dense.cpp` and the fused `ggml_dsv4_hc_*` ops,
  after diffing the formulas. The tensor list shows NO `hc_head`/`output_hc` tensors: find out how GLM collapses the
  4 streams before the head.
- **Attention, two kinds:**
  - 34 **KDA** (Kimi Delta Attention, linear) layers = every layer except 3,7,11,...,43. 64 heads x head_dim 128,
    `short_conv_kernel_size 4` (separate q/k/v conv1d), `gate_lower_bound -5.0`. Tensors: `attn_{q,k,v,output}`
    (Q4_K), `ssm_{a, dt.bias, beta, f_a, f_b, g_a, g_b, conv1d_{q,k,v}, norm}`. f_a/f_b = low-rank per-channel decay
    gate, g_a/g_b = low-rank output gate. Recurrent state per layer = 64 x 128 x 128 f32 = 4 MiB (34 layers: 136 MiB),
    constant in context length.
    Reuse: Strata's gated-delta-net kernels for Qwen-Next (`src/kernels/cuda/{gdn,fused_gdn,native_gdn}.cu`,
    `src/prefill/gdn_rec_parity.cu`). KDA differs from GDN in the decay: per-channel (a vector per head) instead of
    one scalar per head - the recurrence and chunked prefill kernels need that change.
  - 11 **DSA** (DeepSeek sparse attention, MLA) layers 3,7,...,43: `q_lora_rank 1536`, `kv_lora_rank 512`,
    qk_nope 256, **qk_rope 0** (`mla_use_nope`: no RoPE in these layers), v 256, 64 heads. Lightning indexer:
    32 heads x 128, top-2048 keys, `index_kpool 4` + compress. Tensors: `attn_q_a(+norm)`, `attn_q_b`,
    `attn_kv_a_mqa(+norm)`, `attn_k_b`, `attn_v_b` (bf16), `attn_output`. The GGUF list shows no indexer tensors -
    check whether the fork drops the indexer (dense attention) or stores it elsewhere. Below 2048 tokens of context the
    top-k selects every key, so dense attention is exact there.
- **FFN:** layers 0-2 dense (`ffn_{gate,up,down}`, 12288 wide, Q4_K). Layers 3-44 = 42 MoE layers: 288 routed experts,
  top-8, 2048 wide, sigmoid scores + `exp_probs_b` bias (noaux_tc), `norm_topk_prob`, `routed_scaling_factor 2.5`,
  1 shared expert (`*_shexp`, Q4_K), **`swiglu_limit 10`** (clamped SwiGLU, as in DS V4 - the expert kernels already
  take a limit). Router `ffn_gate_inp` f32.
- Experts: Q2_K in 36 layers, Q3_K in 3, Q4_K in 3 (which ones: see the types file). ~8.3 MB per Q2_K expert
  (gate+up+down), ~100 GB of experts in all. Per token: 42 x 8 = 336 expert reads (~2.8 GB at Q2_K).
- MTP: config has `num_nextn_predict_layers 1`, but this GGUF has no layer 45 (no MTP). Vision: `vision_config` in the
  model, not in this GGUF (an `mmproj` exists in other repos).

## Memory budget on this box (4090 24 GB + 90 GB DDR4)
Experts ~100 GB vs a ~66-70 GiB pinned arena + VRAM slots: most experts can sit in RAM/VRAM (a better ratio than
MiMo's 115.7 GB GGUF). Non-expert weights (~4.4 GB at Q4_K + bf16 MLA pieces) live in VRAM. KDA state is small and
fixed; MLA cache per token is small (latent 512). Leaves VRAM for expert slots - VRAM slots were the scarcest
resource on DS4 (~120 fewer slots = -6 points hit rate).

## Reference oracle
`neurall/llama.cpp` (GLM-5.3-Flash support from PRs #27773/#27917 by timkhronos + a VRAM-filling expert cache built
on PR #27861). Their number: 2x RTX 3090 + 3700X, decode 13.8 -> 21.5 t/s with the cache, prompt 217 -> 181 t/s,
wikitext-2 ppl 3.5871 (40 x 512). Build it into `~/AI/llama.cpp-glm53` (not inside this repo), then:
(1) baseline tok/s + ppl on this box, (2) golden logits / hidden states for the port's gates (as for DS4:
`tools/ds4/golden_dump.cpp`, `compare_logits.py`). Note PR #27754's `ggml_mul_mat` on KDA o_proj ignores LoRA
(heretic-gguf ships a one-line patch) - matters only when comparing LoRA runs.

## Port order
1. Reference: build the fork, baseline on the real model (memguard, GPU lock), golden dumps for 3-4 prompts.
2. GGUF loading: tensor map for glm5next (`tools/glm/` mirroring `tools/ds4/`), tokenizer + chat template check.
3. Dense side (`GlmDense`, from `Ds4Dense`): mHC, KDA (CPU reference first, then CUDA from the GDN kernels),
   MLA/DSA (dense attention first; indexer later), dense FFN layers 0-2, shared expert, router. CPU gate vs the
   oracle's hidden states on a mini fixture (`make_mini_gguf.py` style), then the real model.
4. Experts: `Ds4MoeTier` as is (Q2_K/Q3_K/Q4_K; MMQ prompt chunks; `--vram-lru`; split reads) + a routing profile
   (`route_probe`) for the seed.
5. Runtime rank-1 LoRA: for an expert's `down_proj`, `y += B (A . h)` - one 2048-dot + one 4096-axpy per active
   expert, applied after the expert kernel (CPU and GPU paths) so the quantized blobs are untouched; same for shared
   down and `attn_output`. Gate: logits with the LoRA vs llama.cpp `--lora` on the fork.
6. Speed work from DS4/MiMo: CUDA graphs, chunked MMQ prefill, VRAM LRU, route-bias, then MTP if an MTP head GGUF
   shows up.
7. Quality: `tools/mimo/quality/run_quality.py` pattern (7 agentic/coding tasks) with GLM's chat template.
