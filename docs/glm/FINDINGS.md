# Strata-GLM findings

## s1 (2026-10-08): why GLM-5.3-Flash, what to run, what fits

**Choice (Mal, 2026-10-08):** order is now GLM-5.3-Flash -> (DS4 in its own session) -> MiMo later. MiMo is kept for
its omnimodality (audio in, video, speech out), not coding.

**Quality research** (benchmarks + Reddit, 2026-10-08):
- Artificial Analysis Intelligence Index: GLM-5.3-Flash 42, DeepSeek V4.1 Flash 39, MiMo-V2.6-Flash 38,
  DeepSeek V4 Flash 0731 34.
- Vendor-reported, GLM vs DS 0731: Terminal Bench 2.1 84.3 vs 82.7, DeepSWE 63.4 vs 54.4, Toolathlon 78.4 vs 70.3.
- User reports: GLM writes the cleanest code and fixes root causes, but is slow and loops on long unattended runs.
  DeepSeek is fast and needs little babysitting. MiMo-V2.6-Flash is widely panned for coding agents (thinking and
  tool loops).

**DeepSeek V4.1-Flash is not viable here:** it's a new architecture (encoder-decoder + engram, 552B + 196B). Read
from the Q2_K GGUF headers (vcruz305, needs a llama.cpp fork): 264 GB = 197 GB experts + 65 GB engram (can stay
on NVMe) + 3 GB other. That's about 2x RAM+VRAM even with the engram on disk.

**GLM files:** target is neuralll's 3.0-bit Q4_K-attention GSQ-RCO (113.6 GB). Uncensoring is a rank-1 LoRA
applied at run time (GCSA Abliterix v2 by default); details in PLAN.md.

**Architecture** (PLAN.md): it overlaps with DS V4 (mHC with 4 streams, MLA layers, clamped SwiGLU, sigmoid router
with bias) and adds KDA linear attention in 34 of 45 layers, a relative of Strata's Qwen-Next gated delta net.

NOT measured yet: anything on the real model.

## s2 (2026-10-08): reference read, ops verified, phase-1 code written (not built)
- neurall/llama.cpp glm5-next.cpp read in full; its graph is the spec (ARCHITECTURE.md). mHC identical to DS V4
  (same fused ops); head = plain mean of the 4 streams (no hc_head). Dense FFN layers 0-2 ALSO use the SwiGLU clamp
  (swiglu_clamp_shexp[il] = 10 on every layer).
- Strata's ggml (DS4's third_party/llama.cpp) already has gated_delta_net (KDA), lightning_indexer, ssm_conv,
  dsv4_hc_*, swiglu_clamp: no new kernels for phase 1.
- The expert tier needs no changes (per-layer Q4_K/Q3_K/Q2_K + non-routed dense layers work since MiMo).
- Online (2026-10-08): no GLM-5.3-Flash-specific kernel projects worth reusing - neurall's fork (expert cache from
  llama.cpp PR #27861, MTP-only GGUF loading), LayerStoRm (expert streaming, multi-GPU SM120, PCIe-bound, 24.5 tok/s on
  2x5090+2x5080), DGPP (GB10-only engine). The KDA/MLA kernels everyone uses are llama.cpp's.
- MTP head GGUF (neuralll, 4.6 GB): depth-1 acceptance 70-75% but slower in the stock fork with 30% CPU experts; ours
  is planned Strata-fied (ARCHITECTURE.md). NOT measured here.

## s3 (2026-10-08 evening): CPU build clean; the mini fixture runs the whole dense path
- `build-glm` (CPU tree) configures and builds clean on the first try: `tools/glm/glm_dense.cpp` (881 lines) +
  `glm_generate.cpp` compile and link with no errors, `glm_generate_cpu` runs. First compile of the phase-1 code.
- New `tools/glm/make_mini_glm.py`: a 4-layer `glm5-next` GGUF (tiny dims, same structure as the real model) -
  blk.0 KDA+dense, blk.1 MLA+MoE, blk.2 KDA+MoE, blk.3 MLA+MoE - so all four attention x FFN combinations are covered.
  `glm_generate_cpu -m /tmp/mini-glm.gguf --backend cpu --experts cpu --ids 1,2,3 -n 3` loads and runs end to end (mHC,
  KDA conv + recurrence, dense MLA, dense FFN, sigmoid router + shared expert + `Ds4MoeTier`, mean head): finite logits,
  valid token ids, exit 0. No GPU, no real model needed.
- `--experts cpu` note: the tier reports `arena 0.0 GiB (0 experts)` and reads experts from the GGUF per token (file
  tier) - output is correct, but the CPU pool/arena is not populated; look at the tier's cpu-only path if CPU speed
  ever matters.
- `build-glm-gpu` (CUDA, arch 89, MMQ_KQUANTS) also builds clean, and `glm_generate --backend cuda --experts cpu` runs
  the same fixture with the same token stream as the CPU backend (44 5 9 5 44) - so on this fixture the CUDA path for
  mHC / `gated_delta_net` (KDA) / `flash_attn_ext` (MLA) / head agrees with CPU. GPU left at 8% / 0.6 GB; ComfyUI down.
- Reference fork: a **CPU-only** build exists (`~/AI/llama.cpp-glm53/build/bin/{llama-cli,llama-perplexity,
  llama-tokenize}`); no CUDA fork build yet (DS4 was running). Making the fixture fork-loadable surfaced two findings:
  (1) **FIXED** - MLA layers must have `glm5-next.attention.head_count_kv` = 1, not n_head: the DSA layer writes the
  shared latent (kv_lora), so a nonzero kv-head count sizes the kv cache `n_embd_k_gqa = head_k*head_kv` too wide
  (llama-kv-cache.cpp:209/233) -> `ggml_set_rows` `a->ne[0]==b->ne[0]` assert via `build_dsa_layer`. The fixture now
  writes `HEAD_COUNT_KV = [0, 1, 0, 1]` and that assert is gone. (Our engine is unaffected: `is_kda` is `head_count_kv==0`.)
  (2) **REMAINING** - the CPU-only fork then hits `binary_op: dst f32, src0 f32, src1 f16` in the fused lightning-indexer
  path (`cparams.fused_lid`, llama-context.cpp:249) - the fork's GLM indexer is CUDA-oriented, so a CUDA fork build (or
  the real model) is needed. Gate plan: fork `llama-perplexity` vs our `--ppl` on the same ids.
- Build rule learned this session: do NOT build while the `strata-ds4` session is running (compiles skew its timings).

## s4 (2026-10-08 night): FIRST real-model run (GLM-5.3-Flash 3.0-bit Q4K-attn) - runs, slow cold
- Blocker fixed: `native_expert_supported` (src/kernels/cuda/iq_kernels.cu) requires the down-proj type to be in
  `STRATA_D_FMTS`; Q4_K (12) was missing, so GLM's Q4_K expert layers (3-5) were refused at tier init
  ("native_expert_grouped has no kernel for layer 3's types 12/12/12"). Added `X(12)` - `Fmt<12>`, `vec_dot_q4_K_q8_1`
  and the row-bytes `case 12` already existed, so it is a one-token, backward-compatible fix.
- `glm_generate --backend cuda --experts gpu --slots auto --arena-gib 60 --vram-lru --ctx 2048 -n 16` (memguard 84 70):
  45 layers (34 KDA, 11 MLA), 288 experts top-8, vocab 154880; dense half on CUDA in **1.4 s**; 1269 resident slots;
  arena 60 GiB = 6917 experts + file tier 5179; tier load 55.9 s. Prefill (5 tok, decode-loop) 1.6 tok/s;
  **decode 1.75 tok/s** (15 passes / 8.55 s). experts **549 ms/token** - hit **18.6%** (gpu 6%, cpu 71.8%, pcie 9.6%,
  file-tier 2160), file reads 460 ms/pass: COLD cache, seeded in index order with **no routing profile**.
  attention+router 14.7 ms, finish 2.7 ms, head 0.75 ms. VRAM free at the end 1.67 GiB (tight).
- NOT a quality result: output was a degenerate repeat (token 500 x16) on arbitrary prompt ids and with **no
  abliteration LoRA** - runtime LoRA (ARCHITECTURE Decision 5) is **not implemented**. Next: (a) routing profile
  (`--dump-routes` -> `--profile`) to warm the expert cache, (b) implement the runtime rank-1 LoRA (Abliterix v2 as the
  baseline), (c) a real tokenized prompt + compare vs the fork.

## s5 (2026-10-08 night): abliteration adapter LOADER done + verified (NOT wired yet)
- `tools/glm/glm_lora.hpp` (header-only) parses a rank-1 GGUF LoRA into host floats: `exps(layer,mod,expert)` for the
  routed `ffn_{gate,up,down}_exps`, `solo(layer,mod)` for `ffn_down_shexp` / `attn_output`; `Lora1{a,b}` is one
  `y += b*(a.x)`. Verified against `lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf`:
  gate 26 layers a=4096 b=2048, up 26 (4096/2048), down 26 (2048/4096), down_shexp 27 (2048/4096),
  attn_output 30 (in per layer 8192 KDA / 16384 MLA -> 4096); spot values sane; shapes match the GGUF.
- Adapter truth (from the GGUF, corrects PLAN's older note): `alpha 1.0`, F16, 135 pairs, and it touches gate/up as
  well as down.
- NOT wired: no `--lora` flag; neither the `GlmDense` graph (`attn_output`, `ffn_down_shexp`) nor `Ds4MoeTier`
  (routed exps) applies it yet. That is the next slice (see RESUME_PROMPT.md Next).

## s6 (2026-10-08 late): abliteration LoRA wired into the DENSE half; cold baseline reproduced on a real prompt
- **`--lora <adapter.gguf>` is in** (`glm_generate.cpp`); `GlmDense` applies the adapter's whole-module rank-1 pairs in
  `build_attn`: `attn_output` on both the KDA and the MLA attention output, `ffn_down_shexp` on the shared expert's
  SwiGLU output, each as `y += mul_mat(B, mul_mat(A, x))` (A `[in,1]`, B `[1,out]`, F32, in their own backend buffer
  `Impl::lctx`/`lbuf`, uploaded in `init`). Without `--lora` every delta is NULL and the graph is bit-identical. Both
  trees built clean; `glm_dense.cpp` + `glm_generate.cpp` pass `-fsyntax-only`.
- **NOT wired: the routed experts** (`ffn_{gate,up,down}_exps`, layers 3-28) - where the adapter's abliteration mostly
  sits. The tier's expert math has no single hook: CPU = `pool->run_split_multi_native` (ds4_moe.cpp:1123, from
  `cpu_run` :1103), duplicated across `gpu_run`/`gpu_run_n`/`gpu_run_chunk` + the CUDA grouped/MMQ kernels. Needs new
  kernels + a tier interface - design in RESUME_PROMPT.md. **So the model is not yet abliterated.**
- **Real prompt, not arbitrary ids**: `bench/glm-2026-10-08/prompts/` (690-token `neutral6x`) tokenized with the HF
  `tokenizers` lib on `bench/glm-2026-10-08/upstream/tokenizer.json`. The fork's `llama-tokenize` REFUSES on
  `glm5-next` (`llama_init_from_model: glm5-next requires ctx_other`) - the HF lib is the working tokenizer.
- **Cold baseline reproduced** (690-token prompt, `--dump-routes routes.bin`, 512x42x8 u16 = 344064 B): prefill
  **1.74 tok/s**, decode **1.75 tok/s**, hit 19.0%, cpu 71.4%, pcie 9.6%, file tier 2152, file reads 461 ms/pass;
  experts 552 ms/token of which **cpu pool 525 ms** - and 461 of that 525 is FILE READS. The tier is I/O-bound, not
  compute-bound: the warm arena (no file reads) should cut the CPU pool toward ~65 ms and lift decode several-fold.
- **LoRA verified ACTIVE** (bench/glm-2026-10-08/run_lora.sh, real 21-token chat prompt, same ids both runs): the
  adapter changes generation (token 3 `29` -> `16343`) and `logits[0]` max|d| = **2.14** (mean 0.318), same argmax.
  So the loader -> graph path works end to end. (It only moves the dense-half modules; not an abliterated model yet.)
- **Warm cache: the routing profile BARELY helped** (`--profile routes.bin --vram-lru`): prefill 1.74 -> **1.91 tok/s**,
  decode 1.75 -> **1.88 tok/s** (+7%), hit 19.0 -> **29.7%** - but **file reads stayed 472 ms/pass** (cold 461) and the
  cpu pool stayed 476 ms/token. Why: the arena holds only ~55-57% of the 12096 (layer,expert) pairs (the 60 GiB
  budget), so ordering barely matters; worse, the 512-token profile block saw only **6551 distinct** experts, so the
  ranked arena stopped at 54 GiB instead of the 60 GiB budget. The `--dump-routes` sample (512 tokens = 1 block) is too
  small to rank the whole table.
- **Next speed levers (measured, not guessed)**: (a) a BIGGER arena - all ~12096 experts is ~72 GiB and would put file
  tier at 0 (MemAvailable was 85 GiB); (b) raise `--pcie` (0.25 default -> 0.55, DS4's value) so more misses compute on
  the GPU instead of the CPU pool; (c) lower `--pf-b` (1.43 -> ~0.7): 60 prefetch issued/token vs 22.9 useful = wasted
  DMAs; (d) **chunked MMQ prefill** - prefill is still the 1-token decode loop (1.9 tok/s), the single biggest number
  (DS4 392 tok/s chunked).


## s7 (2026-10-09): routing is FLAT -> chunked prefill, exclusive byte-sized cache, REAP pruning (bench/glm-2026-10-09/)
- **Why the cache could not help:** GLM's aux-free routing is near-uniform. Over 512 tokens the top 10% of experts per
  layer take 18% of routes; a 1265-slot VRAM cache ranked by frequency hits 10.6% on held-out tokens (= its share of
  experts). Locality is weak too: consecutive tokens share 0.83 of 8 experts/layer (random 0.22), a 64-token window
  is no better than random, and a 2-token verify touches 15.2 distinct experts -> MTP ~0.9x here (memory-bound).
  Decode speed = which fraction of the 12096 experts (107.2 GB) is resident at all, not which ones.
- **Lossless fixes (measured):** `--arena-skip-resident` (VRAM slots no longer duplicate arena experts) + `--arena-gib
  72`: decode 1.75 -> 3.95 tok/s. VRAM cache sized in BYTES (`slot_gib`, smallest blobs first; was a Q4_K-sized
  count): 951 -> 1851 slots at margin 3.5 GiB.
- **Chunked prefill** (`--prefill-chunk N [--chunk-mmq]`; GlmDense passes up to 4096 tokens: per-layer hand-off
  tensors and router blocks are now ONE shared set, chunk graphs share one re-planning allocator, `logits_rows` reads
  any 16 rows): 2000 tokens 1.7 -> **80-90 tok/s** (1024-token chunks, file tier present). Needs `--vram-margin 5.5`
  at chunk 1024 + MMQ (3.5 OOMs in `m_xg`).
- **Open gate:** chunked vs decode-loop ppl on `neutral` (115 tok): loop 20.4601 (bit-identical over 2 runs), chunked
  19.3327 (-0.057 NLL); first-token logits max|d| 1.51, same argmax. Needs the fork oracle (or MMQ off) to say which
  path drifts.
- **REAP pruning** (`--saliency F` in chunked prefill: tier reads back every entry's expert output, accumulates
  w*||f(x)||; `tools/glm/reap_prune.py`; `--prune F` masks experts out of routing AND out of arena/VRAM). Calibration
  4 x 2000 tokens (code, prose, zh/ja, chat). Held-out 2000-token evals, chunk 1024, margin 5.5:

  | config | eval_code ppl | eval_chat ppl | prefill tok/s | decode tok/s | file tier |
  |---|---|---|---|---|---|
  | base (no prune) | 3.676 | 5.633 | 86-90 | 3.72 | 1335 |
  | REAP 14.2% (fit92) | **3.576** | **6.672 (+18%)** | 150 | 7.2 | 145 |
  | random 14.2% (control) | 3.909 (+6.3%) | - | 151 | 7.0 | 145 |
  | REAP 25% | 3.597 | - | **175** | **9.3** | 0 |

  REAP's choice matters (random is 9% worse than REAP on code), and code survives even 25%. **But the chat eval
  (mostly German README text, a language NOT in calibration) lost 18% at 14%**: a static prune list holds only for
  calibrated domains. Next: soft pruning (penalty, not mask) + broader calibration (more languages/domains).
- Process lesson: `pkill/pgrep -f <pattern>` matched my own shell twice (exit 144). Kill exact PIDs from a pid file.
- **Chain 3 (2026-10-09 05:25-05:33):** broader calibration (+romance READMEs, python, json/tool-call; still NO
  German) made the German-heavy chat eval WORSE: 7.28 (4-domain list 6.67, base 5.63). German relies on its own
  experts that a broad non-German calibration ranks low -> pruning is domain-specific; build the prune list from the
  mix actually used (Mal's idea: English + Chinese + code + tool calls) and keep several lists.
- **skip_miss (decode, REAP 25% prune, fully resident, `--vram-margin 1`, decode-loop ppl over 300 eval_code
  tokens):**

  | --skip-miss | decode tok/s | ppl | skipped/token |
  |---|---|---|---|
  | 0 | 10.38 | 11.159 | 0 |
  | 0.05 | 11.37 | 11.465 (+2.7%) | 31 |
  | 0.10 | **16.38** | 11.604 (+4.0%) | 115 |

  Decode 1.75 (yesterday) -> 10.4 lossless-ish (prune 25%) -> 16.4 with skip 0.10. The margin-1 VRAM cache (hit
  35.8% vs 29% at margin 5.5) is worth ~1 tok/s: re-seeding the cache bigger after a chunked prefill is a TODO.

## s8 (2026-10-09 morning): lightning indexer IN (long context), shared-allocator bug fixed
- **Lightning indexer** (`GlmDense`, MLA layers; glm5-next.cpp build_kpool_select semantics): pooled-key cache
  `ipool` F16 [128, ctx/4+1] (pool p = tokens 4p..4p+3: LayerNorm(indexer.attn_k x) mixed by a per-channel softmax over
  the 4 of compressor_gate x + ape), carry `itail` [256, 3] across passes; score = ggml_lightning_indexer (128 x 32 heads,
  wmma), pool visible iff 4p+3 <= t (step mask on GPU), top-512 pools (CUB top_k) + incomplete tail, dead slots to
  per-slot dump rows, F16 additive mask via set_rows -> flash_attn. Below idx_top_k positions the dense causal path
  runs (exact and cheaper); the pool cache is updated on every pass either way. `--allow-long-ctx` = force dense.
  Context is no longer capped at 2051: `--ctx N` sizes the latent cache (N=524288 -> ~6 GB VRAM).
- **CPU gate** (`bench/glm-2026-10-09/gate_indexer.sh`, mini fixture, idx_top_k 8): G1 indexer == dense at 11 tokens
  (max|d| 0), G2 differs at 30, G3/G4 chunk 7 / chunk 5 == one-token loop (0), G5 dense chunk == dense loop (0); clean
  under AddressSanitizer (`build-glm-asan`). Mutation (tail off by one) -> G1 4.49 + G4 0.354 FAIL. Indexer SCORING is
  not gated against the fork yet (`~/AI/llama.cpp-glm53` patched: `GLM_NO_FUSED_LID=1` disables its fused indexer so
  its CPU build can run the fixture - rebuild + compare = TODO).
- **Real model, 2300-token prompt crossing 2048** (REAP 25%, chunk 1024): indexer ppl 4.3780 vs forced dense 4.3765
  (+0.03%; ~250 of 2300 positions dropped per token at the end), prefill 156 tok/s, decode 8.5 tok/s (margin 5.5).
- **BUG FIXED - shared gallocr stale pointers:** gallocr treats a tensor with ->data set as externally allocated, so
  after the shared chunk allocator's buffer grew, earlier chunk graphs kept pointers into freed memory (CPU: heap
  corruption with chunks > 4 tokens; GPU: silent). `Impl::alloc` now nulls data/buffer of every graph tensor not in a
  persistent buffer before re-planning (pointer compare only - the stale buffer must not be dereferenced). The
  chunked-prefill numbers of s7 ran with this bug: neutral ppl moved 19.33 -> 18.47 after the fix (loop 20.46) - the
  remaining chunk-vs-loop gap is GPU-only (CPU fixture is bit-exact) -> MMQ vs CPU-pool expert numerics or batched FA;
  **open, needs the fork oracle**. Prune-sweep conclusions (relative) stand; absolute chunked ppl should be re-taken.
- **BUG FIXED - flash_attn mask stride:** CUDA FA with 512-wide heads needs every mask stride % 16 bytes; the indexer
  mask rows are cap + nsel halfs -> padded to a multiple of 8.
- **524288-token context allocates** (11.15 GiB free after the dense half -> 734 expert slots at margin 5.5, REAP 25%
  fully resident, arena 69.2 GiB). 50K-token prefill + decode at that depth: see s9 / bench/glm-2026-10-09/long50k.log.
- From strata-ds4-gpu (DS4 measurements, to port): soft prune `--prune-penalty 0.5` + `--arena-adapt` (pruned experts
  read from NVMe on demand into an LRU arena slot) keeps languages: ppl +0.3% overall, German +3.9% (hard +72%), decode
  +10%. GLM has both flags; next GPU block tests them with an en+zh+code+tools calibration list.

## s9 (2026-10-09 06:08): 524288-token context, 50K-token prefill + decode at 50K depth
- REAP 25% prune, chunk 1024, margin 5.5 (734 VRAM slots): **prefill 50000 tokens in 217.8 s = 229.6 tok/s**
  (attention+router 43 s, experts 169 s); **decode at 50K depth 8.23 tok/s** (attention+router 20.6 ms/token vs
  ~22 at 2K) - long context costs almost nothing in decode. Log: bench/glm-2026-10-09/long50k.log.

## s10 (2026-10-09 16:54-17:03): soft prune on GLM - penalty must be scaled to the router (bench/glm-2026-10-09/chain5.sh)
- **Penalties don't transfer from DS4.** GLM selects on sigmoid(logit) + exp_probs_b: scores in [0,1], and the bias
  spread across experts is only p10-p90 0.03-0.13 (blk.3 0.035 ... blk.44 0.131). DS4's 0.5 would be ~a hard prune
  here, so the sweep is 0.02 / 0.05 / 0.10.
- List: REAP 25% (72/layer) from en+zh+code+tools calibration (`sal-cal_{code,prose,multi,chat,python,json}`,
  12000 tokens; romance + German held out) -> `prune-ezct-0.25.txt`; control `prune-rand-0.25.txt` (72/layer, seed 41).
  Soft runs add `--arena-adapt`. Held-out 2000-token evals, chunk 1024, margin 5.5:

  | config | eval_chat ppl (German, uncalibrated) | eval_code ppl | prefill tok/s |
  |---|---|---|---|
  | base, no prune (s7, ran WITH the gallocr bug) | 5.633 | 3.676 | 86-90 |
  | hard REAP 25% | 8.949 (+59%) | **3.564** | 161 (code) |
  | hard random 25% (control) | - | 4.289 (+20% vs REAP) | 170 |
  | soft 0.02 + adapt | **5.457** | - | 89 |
  | soft 0.05 + adapt | 5.516 | 3.583 | 97-102 |
  | soft 0.10 + adapt | 5.528 | - | 114 |

  Soft prune gets back all of the uncalibrated-language loss (8.95 -> 5.46-5.53) while code stays within 0.5% of
  hard. The REAP choice matters (random costs +20% on code). **Caveat:** the base row predates the allocator fix
  (s8 moved chunked ppl by ~4%), so "soft < base" isn't established. Re-take base on the next block.
- **Decode** (eval_code300, -n 32, margin 1, soft 0.05 + adapt): **7.57 tok/s** vs 10.38 for the hard 4-domain 25%
  list (sm0). Experts 114 vs 78 ms/token: 282 file-tier reads, 211 arena swaps, 43 ms/pass file reads, cpu-pool wall
  98 vs 73 ms. Over 32 tokens that is adapt warming up / thrashing. Prefill also drops 161 -> ~100 tok/s.
- Next: re-take the no-prune base; longer decode (256+ tokens) to see whether adapt settles; an admission gate (DS4
  research, Infernix: promote a file-read expert only if its decayed hit count beats the coldest arena expert, not
  on first read); try soft 0.02 on code + decode.
- Process: two copies of chain5 overlapped for ~1 min (a queued job still in its `sleep 120` was missed by the
  process check). memguard.sh already takes ds4-gpu.lock for each run, so an outer `flock` on the same lock
  deadlocks: never wrap memguard runs in it. c5-hard-chat.log was truncated by the duplicate's start; its ppl
  (8.9493) came from the monitor (`c5-hard-chat.note`).

## s11 (2026-10-09 17:12-17:21): clean base; winning back soft-prune decode (bench/glm-2026-10-09/chain6.sh)
- **Base measured again after the allocator fix** (no prune, chunk 1024, margin 5.5): eval_chat **5.547** (s7 said 5.633),
  eval_code **3.668** (3.676). Against it: soft 0.02 chat 5.457 (-1.6%), soft 0.05 chat 5.516 (-0.6%) / code 3.583
  (-2.3%), hard 25% code 3.564 (-2.8%) but chat 8.95 (+61%). **Soft prune 25% costs no quality vs unpruned** on
  either the calibrated domain or the held-out language, and is slightly better on these evals.
- New tier options (tools/ds4/ds4_moe.{hpp,cpp}, glm_generate flags; both default off):
  - `--skip-file T`: like skip_miss, but only for experts the arena does not hold (NVMe file tier = the soft-pruned
    ones); dropped if their weight is < T x the token's weight sum.
  - `--arena-admit H`: arena_adapt admission gate. Per-(layer, expert) heat with a half-life of H tokens; a file
    read is promoted only if its heat is >= 2 and beats the coldest arena expert (which becomes the victim).
- Decode (eval_code300: 300-token decode-loop prompt = ppl, then 64 tokens timed; margin 1; ezct 25% list):

  | config | decode tok/s | ppl (299 tok) | file reads | file ms/pass | experts ms/token |
  |---|---|---|---|---|---|
  | hard 25% | **9.52** | 11.974 | 0 | 0 | 86.2 |
  | soft 0.05 + adapt (s10, -n 32) | 7.57 | 11.764 | 282 | 43.4 | 114.1 |
  | soft 0.05 + adapt + admit 64 | 7.20 | 11.860 | 709 | 51.8 | 121.0 |
  | **soft 0.05 + adapt + skip-file 0.10** | **8.46** | **11.554** | 249 (494 skipped) | 21.3 | 98.8 |
  | soft 0.05 + adapt + admit 64 + skip-file 0.10 | 8.23 | 11.676 | 304 (580 skipped) | 26.3 | 102.4 |

  The gate is a loss: GLM's routing is flat, so the soft-pruned experts that do win come back often enough that
  refusing to cache them means re-reading them from NVMe (709 reads vs 282). Promote-on-first-read stays the default.
  skip-file 0.10 halves the NVMe time and gets back 0.9 of the 1.95 tok/s soft prune costs, with no ppl loss (the
  299-token decode-loop ppl spread here is ~3%, i.e. noise level; it is not a gain).
- Remaining gap to hard: ~21 ms/pass of file reads + a slightly busier CPU pool. Next: skip-file 0.15/0.20, penalty
  0.10 (fewer soft-pruned wins), and skip-file + skip-miss 0.05 to buy back speed elsewhere; chunked prefill
  (161 -> ~100 tok/s under soft prune) has no skip yet; run_chunk is the place to add it.

## s12 (2026-10-09 17:25-17:37): soft prune at hard-prune speed - skip-file sweep (chain7) + prefill skip (chain8)
- New: `--skip-file-prefill T` (Ds4MoeConfig::skip_file_chunk): in a prompt chunk, a file-tier entry below T x its
  token's weight sum is redirected to the token's heaviest expert with weight 0 (contributes nothing; an expert left
  without entries is never read). Kernels untouched.
- **Decode** (chain7; eval_code300, 64 tokens timed, margin 1, ezct 25% list, soft = penalty 0.05 + adapt):

  | config | decode tok/s | file ms/pass | file-tier skipped |
  |---|---|---|---|
  | hard 25% (s11) | 9.52 | 0 | - |
  | hard 25% + skip-miss 0.05 | 10.77 | 0 | - |
  | soft + skip-file 0.10 (s11) | 8.46 | 21.3 | 494 |
  | **soft + skip-file 0.15** | **9.82** | 7.9 | 1023 |
  | soft + skip-file 0.20 | 10.10 | 4.1 | 1087 |
  | soft penalty 0.10 + skip-file 0.10 | 8.89 | 17.1 | 224 |
  | soft + skip-file 0.10 + skip-miss 0.05 | 8.91 | 21.6 | 568 |

  The 299-token decode-loop ppls (10.9-11.8) swing ~7% run to run: too small to rank quality. Quality is judged
  at full size below.
- **Quality at full size + prefill** (chain8; 2000-token held-out evals, chunk 1024, soft 0.05 + adapt; the prefill
  skip applies the same rule, so it is also the quality proxy for decode skip-file):

  | config | eval_chat (German) | eval_code | prefill tok/s (chat / code) |
  |---|---|---|---|
  | base, no prune (s11) | 5.547 | 3.668 | ~88 |
  | hard 25% | 8.949 (+61%) | 3.564 (-2.8%) | 161 (code) |
  | soft, no skip (s10) | 5.516 (-0.6%) | 3.583 (-2.3%) | 97 / 102 |
  | **soft + skip-file-prefill 0.15** | **5.665 (+2.1%)** | **3.605 (-1.7%)** | **130 / 140** |
  | soft + skip-file-prefill 0.20 | 5.860 (+5.6%) | 3.625 (-1.2%) | 143 / 157 |

- **Recommended config:** `--prune prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-file 0.15
  --skip-file-prefill 0.15` -> decode ~9.8 tok/s (= hard prune), prefill 130-140 (hard 161), German +2% vs
  unpruned instead of +61%, code still better than unpruned. 0.20 buys ~10% more prefill speed for another
  +3.5% on German. To win back the rest: skip-miss on VRAM misses (hard + 0.05 -> 10.77), bigger chunks, VRAM
  re-seed after prefill.

## s13 (2026-10-09 17:39-17:55): PCIe share, skip-miss on top, decode-path quality gate, DwarfStar (chains 9-10)
- **--pcie sweep** (recommended soft config, eval_code300, 64 tokens timed): 0.25 9.82 | 0.35 **9.97** | 0.45 9.12 |
  0.55 9.86 | 0.65 9.14 tok/s. Flat within noise from 0.25 to 0.55. The PCIe DMA reads the same host RAM the CPU pool
  reads, so moving work to PCIe slows the pool (0.45: cpu pool 77 ms for fewer experts). At 0.65 PCIe is the long pole
  (65 ms). Keep 0.25-0.35.
- **+ --skip-miss 0.05: 10.82 tok/s** (+10%; tier wall 74.8 vs 81.5 ms).
- **Decode-path quality gate** (new): eval_chat (German, 2000 tokens) as a decode-loop ppl, so every token goes through
  run() and skip-file/skip-miss act exactly as in decode (~3.5 min/run at ~9 tok/s):
  recommended (`--skip-file 0.15 --pcie 0.35`) **5.940**, + skip-miss 0.05 **5.807**. Skip-miss costs no measurable
  quality. Decode-loop ppl is not comparable with chunked ppl (the open chunk-vs-loop gap, s8); compare loop to loop
  only. The 300-token loop ppls also do NOT track the CPU/PCIe split (0.55 lowest, 0.65 highest) -> few-% run noise.
- **DwarfStar** (antirez/ds4, ~/AI/ds4-ref fc80bd6) runs GLM-5.3-Flash on Metal/CUDA/ROCm: M5 Max 128 GB, Q4_K SSD
  streaming: prefill 121 tok/s, decode 11.9-14.9. It **refuses our GGUF** (expects 46 blocks with the MTP layer
  inside; ours keeps MTP in a separate file; its CUDA path also has no Q3_K). Head-to-head would need its own
  `glm53-q2` (90 GiB). Ideas taken: (1) prompt/KV cache across turns (`--kv-disk-dir`; full GLM 5.3: 16-token
  append 30.8 -> 2.9 s). GLM's carried state is all in GlmDense's sbuf (KDA conv+S per layer, MLA kvc rows < pos,
  ipool rows < pos/4, itail) -> a snapshot/restore is straightforward; worth it once GLM is served (glm_generate is
  one prompt per process). (2) Prefill streaming = read layer L+1's experts while L computes: our chunk_prestage
  does that for arena experts but is OFF with arena_adapt (soft prune) - a lever for prefill. (3) Its VRAM cache
  "protects every hit before evicting for a miss" - n/a, ours is static with flat routing.
- **Best config now:** `--prune prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-file 0.15
  --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35` -> decode ~10.8 tok/s (hard 25% 9.52; unpruned ~3.7),
  prefill 130-140 tok/s, German chunked ppl +2% vs unpruned.

## s14 (2026-10-09 17:56-18:12): prefill - prestage, bigger chunks; chunked ppl depends on residency (chains 11-12)
- `chunk_prestage` (DMA layer l+1's arena experts while layer l computes - DwarfStar's streamed-prefill idea) is now
  allowed with arena_adapt (prompt chunks never swap arena slots; decode's direct arena reads sync s_cp first). Its
  device half was sized to a whole layer at the largest blob (288 x 15.2 MB = 2 x 4.4 GB -> c_stage OOM even at
  margin 8.5); now capped by `STRATA_PRESTAGE_MIB` (default 1536) and a layer that does not fit is prestaged in part.
  **Result: no gain** (margin 6.5, soft+sfp0.15, code: 140.1 without, 139.3 with). GLM prefill is not waiting on
  the arena DMA. Leave it off.
- **Chunk 2048** (margin 8, + prestage): **160 tok/s** vs 140 at chunk 1024 (+14%), code ppl 3.626. Needs margin ~8
  (chunk 2048 buffers), which costs decode VRAM slots in the same process - only worth it with a VRAM re-seed /
  elastic cache after prefill, or for long prompts.
- **Chunked ppl is repeatable to ~0.1%** (chain12, chain8's soft+sfp0.15 code config x3: 3.6003 / 3.6034 / 3.6034;
  chain8: 3.6047), **but it moves 1-2% with the VRAM margin**: code 3.587 (margin 8.5) / 3.603 (5.5) / 3.620 (6.5);
  chat 5.665 (5.5) / 5.769 (6.5, prestage); hard code 3.564 (5.5) / 3.606 (6.5). The margin only changes which
  experts are VRAM-resident (grouped MMVQ, q8_1 activations) vs streamed (MMQ) -> **the two expert paths differ
  numerically by ~1-2% ppl**. Same family as the open chunk-vs-loop gap (s8). Consequences: (1) compare chunked
  ppls only at the same margin; (2) "soft prune +2% vs base" (s12) is within this path spread; the robust result
  is soft ~= base and hard = +61% on German; (3) the fork oracle (RESUME next step) should say which path is right.
