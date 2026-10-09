# Resume prompt: Strata for GLM-5.3-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-GLM`.

---

We're building Strata-GLM: the Strata inference engine for **GLM-5.3-Flash** (320B total / 18B active MoE, MIT) on my
RTX 4090 24 GB + Ryzen 7 5700X (8c, AVX2 only) + 90 GB DDR4, decoding faster than llama.cpp, with a runtime
abliteration LoRA. **GLM is the current priority** (Mal, 2026-10-08). MiMo is parked (kept for its image/audio
input); DeepSeek-V4 has its own session.
Worktree `~/AI/Strata-GLM`, branch **`glm`**, cut from `deepseek4` `a693545`. Same git repo as the other two
worktrees; the three stay separate for now and get joined later. **Work solo** (no /conductor, workers or
subagents). Don't push anywhere until Mal says.

Read first, in this order: this file; `docs/glm/PLAN.md` (target files, port order); `docs/glm/ARCHITECTURE.md`
(deep dive: formulas from the reference code, reuse map, memory/decode budget, decisions);
`docs/glm/FINDINGS.md`; then the DS4 docs we build on: `docs/ds4/RESUME_PROMPT.md` (rules, gates, lessons),
`docs/ds4/ENGINE_DENSE.md`, `docs/ds4/ENGINE_MOE.md`; `git log --oneline -15`.

## State (2026-10-09 morning) - **CURRENT, read this first**
- Public repo: github.com/Indras-Mirror/Strata-GLM (remote `glm`, branch glm -> main). Mal: English + Chinese (+code)
  are what matter. GPU is SHARED with `strata-ds4-gpu` over the relay MCP: blocks <= 10 min, ask/announce, kill exact
  PIDs only (pkill/pgrep -f matched this shell twice), CPU tests niced with --threads 2 while DS4 benchmarks.
- FINDINGS s7-s8: routing is flat -> residency is what counts. Done: exclusive byte-sized VRAM/arena residency,
  chunked prefill (GlmDense passes <= 4096), REAP saliency + `--prune` / `--prune-penalty`, `--skip-miss`, lightning
  indexer (long context; 524K allocates), shared-allocator stale-pointer fix, FA mask-stride fix.
- Numbers (REAP 25%, fully resident): prefill 156-175 tok/s (chunk 1024), decode 8.5-10.4 tok/s (16.4 with
  --skip-miss 0.10, +4% ppl). Hard prune hurts uncalibrated languages (German -18..-29%).
- **Long context WORKS (FINDINGS s9):** `--ctx 524288` allocates (~6 GB latent cache, 734 VRAM slots left at margin
  5.5); 50K-token prefill **229.6 tok/s**; decode at 50K depth **8.23 tok/s** (vs ~8.5 at 2K).
- **Next (in order):** (1) port DS4's soft prune: `--prune-penalty 0.5 --arena-adapt` with an en+zh+code+tools
  calibration list (cal_code, cal_prose, cal_multi, cal_chat, cal_python, cal_json); eval code/chat/German.
  (2) Fork oracle: rebuild ~/AI/llama.cpp-glm53 CPU (patched, `GLM_NO_FUSED_LID=1`) and compare indexer scoring on the
  mini fixture past 11 tokens; then the GPU chunk-vs-loop ppl gap (18.47 vs 20.46 on neutral, CPU fixture exact).
  (3) Long context deeper: prefill ~25% of 200K/500K (Mal: prove it, don't fill it) and decode at that depth; a
  gather path for decode if masked FA over the whole latent cache starts to cost (it didn't at 50K). (4) Re-seed the VRAM cache after chunked prefill (margin
  5.5 costs ~500 slots / ~1 tok/s). (5) Abliteration: transplant drowzeys' 30 o_proj tensors (L15-43 + MTP) -> Q4_K
  attn_output, instead of the expert LoRA. (6) Bigger chunks (4096) for 300-500+ prefill.

## State (2026-10-08 night) - **superseded by 2026-10-09**
- **LATE UPDATE (FINDINGS s6)**: `--lora` is wired for the DENSE half and **verified active** (adapter changes
  generation; `logits[0]` max|d| 2.14); the routed-expert LoRA is still NOT wired (not abliterated). A real-prompt
  warm-cache A/B measured **cold 1.75 -> warm 1.88 tok/s decode (+7% only)** - the arena caps at ~55% of the 12096
  (layer,expert) pairs, so the profile barely helps, and file reads stayed 472 ms/pass. Speed levers, measured:
  bigger arena (~72 GiB holds every expert), higher `--pcie`, lower `--pf-b`, and chunked MMQ prefill (prefill is
  still the 1-token loop at ~1.9 tok/s). Prompts + run scripts live in `bench/glm-2026-10-08/`.
- **The model is downloaded + verified**: `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf`
  (113585695168 B, sha256 `5f03b74f...38d9f6d`, verified + renamed by the fetcher) and the MTP head
  `mtp/GLM-5.3-Flash-MTP-Q4_K.gguf` (4.6 GB). `bash ~/AI/glm_fetch_all.sh` is the resumable fetcher (now done).
- **FIRST real-model run DONE** (`glm_generate`, CUDA, memguard 84 70, `--slots auto --arena-gib 60 --vram-lru --ctx 2048
  -n 16`): 45 layers (34 KDA, 11 MLA), dense half on CUDA in 1.4 s, 1269 VRAM slots, arena 60 GiB = 6917 experts + file
  tier 5179, tier load 55.9 s. Decode **1.75 tok/s COLD** (expert hit 18.6%, file reads 460 ms/pass) = **no routing
  profile yet**. Details in FINDINGS s4.
- **Blocker fixed and BUILT**: `STRATA_D_FMTS` (src/kernels/cuda/iq_kernels.cu) was missing Q4_K (12), so GLM's Q4_K
  expert layers 3-5 were refused at tier init. Added `X(12)`; `build-glm-gpu` rebuilt; the run succeeded.
- **Abliteration LoRA: loader DONE + verified, NOT wired** (`tools/glm/glm_lora.hpp`, FINDINGS s5). Remaining: a
  `--lora <adapter.gguf>` flag, apply `solo()` deltas in the `GlmDense` graph (`attn_output`, `ffn_down_shexp`) and
  `exps()` deltas in `Ds4MoeTier` (gate/up BEFORE the SwiGLU, down after) in both CPU and CUDA paths.
- Both trees build clean: `build-glm` (CPU), `build-glm-gpu` (CUDA). **No builds while the `strata-ds4` session runs.**

## State (2026-10-08 evening) - superseded by the night block above
- **Code written; the CPU tree builds clean and the mini fixture runs end to end - NOT run on the real model yet**
  (commits `e889e75` + this session):
  - `tools/glm/glm_dense.{hpp,cpp}` - GlmDense, phase 1: mHC (fused ggml_dsv4_hc_* ops), KDA via
    `ggml_gated_delta_net` (KDA path, K=1) + causal conv state, nope-MLA over an F16 latent cache with flash-attn
    (no indexer: exact up to 2051 tokens, refuses beyond unless allow_long_ctx), dense FFN layers 0-2 and shared
    expert with `ggml_swiglu_clamp` (limit = swiglu_clamp_shexp, 10), sigmoid router + exp_probs_b selection bias +
    norm + x2.5, head = mean of 4 streams -> output_norm -> output.  One input upload per pass (slot + causal mask +
    route bias), one readback per layer ([fn|ids|wts] block).  Graph per (layer, n, MLA cap); per-graph allocators
    with uid reuse.  KDA state cannot rewind: positions must arrive in order (no verify/MTP rollback yet).
  - `tools/glm/glm_generate.cpp` - driver (GlmDense + Ds4MoeTier, decode-loop prefill, --ppl, --route-bias,
    --vram-lru, --dump-routes FILE writes a GLM routing profile in seed_from_routes(file, 512) format; without
    --profile the arena/VRAM cache are seeded in (layer, expert) index order).
  - `tools/glm/cmake/glm.cmake` (included from root CMakeLists after ds4_engine.cmake): `glm_generate` (CUDA tree,
    needs ds4_moe_cuda) / `glm_generate_cpu` (CPU tree).
  - `third_party/llama.cpp` copied from Strata-DS4 (untracked, gitignored; 175 MB): the ggml with every GLM op.
- Verified while writing (no build yet): ggml ops gated_delta_net (scales q by 1/sqrt(S) itself; identical to
  the fork's), swiglu_clamp semantics == Ds4MoeTier's CPU clamp (gate min lim, up clamp +-lim), fork's MoE routing
  order, GGUF tensor shapes (ARCHITECTURE.md; header dump in
  /tmp/claude-1000/.../scratchpad/glm_hdr.json is gone after reboot - re-read with the range-read trick if needed).
  Expert types: layers 3-5 Q4_K, 6-8 Q3_K, 9-44 Q2_K; the tier handles per-layer types + dense layers already.
- **Build (2026-10-08 evening):** `build-glm` (CPU) builds clean on the first try and `glm_generate_cpu` runs the new
  `tools/glm/make_mini_glm.py` fixture end to end (4 layers, finite logits, exit 0) - no GPU, no real model.
  `build-glm-gpu` (CUDA, arch 89, MMQ_KQUANTS) was building as this was written -> `glm_generate`.
- Model download: `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/` (~34/105.8 GiB at 18:5x, ~12 MiB/s; resumable:
  `bash ~/AI/glm_fetch_all.sh` - curl `-C -` appends to `$D/GLM-...q4kattn.gguf.part`, then sha256 + rename + the MTP
  head into `mtp/`; launch detached with `setsid nohup`). The old `download_glm53_flash.sh` `FORCE_EXFAT` command is
  stale (restarts a partial from zero) - do NOT use it. LoRAs in `lora/`.
- Reference fork at `~/AI/llama.cpp-glm53` (neurall/llama.cpp 2e0435a). A **CPU-only** build now exists there (`build/`:
  `llama-cli`, `llama-perplexity`, `llama-tokenize`). The CUDA fork build (for the real-model tok/s baseline) is still
  TODO - do it only with DS4 idle. Fixture oracle is pending: the fork accepts the fixture's tokenizer now, but its
  DSA/k-pool graph asserts (`ggml_set_rows` in `glm5-next.cpp`) on the tiny dims - the fixture's `attention.indexer.*` /
  MLA dims must match the fork before it can serve as a gate. Likely primary gate instead: real-model `llama-perplexity`
  vs our `--ppl` on the same ids.
- Relay: the `strata-glm` session's DS4 ask (`e85816a6`) was still open; the 2026-10-08 evening session had no relay
  tools, so it only checked `ps`/`nvidia-smi` - DS4 was idle (no `ds4_generate`, GPU ~9%), which is why the builds ran.
  Before any GPU block > ~15 min, still coordinate with `strata-ds4` (compiles skew its timings).

## Next (in order)
1. **Wire the abliteration LoRA** (loader DONE + verified, `tools/glm/glm_lora.hpp`). Split in two:
   - **(a) DENSE HALF - IN THE TREE (2026-10-08 late)**: `--lora <adapter.gguf>` on `glm_generate`; `GlmDense` applies
     the `solo()` deltas in `build_attn` as `y += mul_mat(B, mul_mat(A, x))` for `attn_output` (KDA + MLA inputs) and
     `ffn_down_shexp` (the SwiGLU output `z`); the A `[in,1]` / B `[1,out]` F32 payloads go in their own backend
     buffer (`Impl::lctx`/`lbuf`, uploaded in `init`). No-op when `--lora` is absent (all deltas NULL).  NOT yet built
     or run - verify with `bench/glm-2026-10-08/run_lora.sh` (logits with vs without the adapter).
   - **(b) ROUTED EXPERTS - NOT WIRED (the abliteration the adapter mostly carries)**. The adapter's
     `ffn_{gate,up,down}_exps` (layers 3-28, all 288 experts) are Ds4MoeTier's, and the expert math does NOT have a
     single hook: the CPU path is `pool->run_split_multi_native` (ds4_moe.cpp:1123, from `cpu_run` at :1103), and
     `gpu_run`/`gpu_run_n`/`gpu_run_chunk` each have their own CPU-pool fallback + the CUDA grouped/MMQ kernels. Plan:
     (i) add `native_gu_rows_lora` / `native_down_rows_lora` to `src/kernels/cpu/native_expert.{hpp,cpp}` (rank-1:
     `g += b_g[r]*(a_g.x)`, `u += b_u[r]*(a_u.x)` before the clamp, `out += b_d[r]*(a_d.h)` after) - additive, DS4
     passes nothing; (ii) a `Ds4ExpertLora` interface on the tier + three `NativeLora1` handles plumbed into the CPU
     pool's expert jobs; (iii) for the CUDA paths, force LoRA-active (layer, expert) to the CPU tier (their kernel
     would otherwise silently skip the adapter). Start with `--experts cpu` correctness, then wire the CPU-fallback.
2. **Warm the expert cache** (independent, cheap, no code): `--dump-routes routes.bin` on a few prompts ->
   `--profile routes.bin --vram-lru` -> rerun; the 1.75 tok/s is a COLD number (18.6% hit) and should climb.
   `bench/glm-2026-10-08/run_warm.sh` does both on a 690-token real prompt.
3. **Real prompt**: ids via the HF `tokenizers` lib on `bench/glm-2026-10-08/upstream/tokenizer.json` (the fork's
   `llama-tokenize` refuses - its GLM path needs `ctx_other`); prompts in `bench/glm-2026-10-08/prompts/`
   (`neutral*.i32`, `chat.i32`). Then compare first-token logits vs the fork.
4. **Fork CUDA oracle build** (`~/AI/llama.cpp-glm53`, sm_89) still TODO (needed for real-model tok/s + the `--lora`
   reference); its CPU tools are done. `make_mini_glm.py` is fork-loadable EXCEPT the fork's DSA/k-pool path asserts on
   tiny dims - gate on the real model instead (FINDINGS s3).
5. **SPEED (Mal's goal: 20+ decode, 300+ prefill; Qwen-Flash-Next level = 50-70 decode / 1500 prefill)**: the two
   levers not yet switched on are (a) the routing profile (task 2) and (b) **chunked MMQ prefill** - `GlmDense` already
   supports `max_tokens` up to 64 and the tier has `run_chunk`/`chunk_mmq`, but `glm_generate` still prefils with the
   one-token decode loop (1.6 tok/s). Wiring `dc.max_tokens = chunk` + `attn_router_n(l, pos0, n)` + `tier.run_chunk`
   is the prefill path (DS4 measured 392 tok/s chunked). Then ARCHITECTURE.md Decisions 6-7: CUDA graphs, Strata-fied
   MTP (upstream llama.cpp PR #29928 added GLM5Next MTP support 2026-10-08), the lightning indexer.

## Rules (non-negotiable; same as DS4)
- **Every full-model load through `tools/ds4/memguard.sh <cap> <need> -- <cmd>`** (cgroup RAM cap, swap off, shared
  GPU lock `~/.quetza-data/conductor/ds4-gpu.lock`). One full-model process box-wide. memguard checks MemAvailable
  BEFORE the lock, so to queue behind another engine wait first:
  `until flock -n ~/.quetza-data/conductor/ds4-gpu.lock true && [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 70 ]; do sleep 30; done`
- Before any GPU run: `ss -ltn | grep 8188` (ComfyUI up -> stop and tell Mal) and `nvidia-smi`.
- **GPU sharing with the DS4 session** over Quetza Relay (use the CLI `~/.local/bin/relay` - the relay MCP tools were
  NOT available in this session): it is `strata-ds4`; claim this session with `relay rename strata-glm`. Before a GPU
  block > ~15 min, `relay ask strata-ds4 "need/ for/ eta:"`, announce done. NB: the CLI acts as an ephemeral
  `relay-cli-<pid>`, and `strata-ds4` connects then disconnects, so a `peer_not_found` just means retry until it is in
  its connected window; `relay reply` to an older ask can return `unknown_ask` - send a fresh `relay ask` instead.
- **No builds while the DS4 session is running** - a compile (CMake/ninja/nvcc) skews its benchmark timings. Check
  `pgrep -af ds4_generate` and `nvidia-smi` first; only let finish a build that is already almost done.
- Never edit `~/AI/Strata-DS4` or `~/AI/Strata-MiMo` (other sessions' worktrees); read them and `git show` freely.
  Port shared code as your own commits on `glm`. Shared kernels (`src/kernels/`, `include/strata/kernels/`):
  backward-compatible additions only.
- If `/media/mal/SSD NVME` or `NVME1TB` is missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3` / `nvme1n1p2`.
- Never `pkill -f` a pattern in your own command line; kill only your own PIDs.
- Test the real model end to end early (quality + tok/s), fix after; keep the gates mutation-tested.
- Keep results in `bench/glm-YYYY-MM-DD/`, update `docs/glm/FINDINGS.md` + this file at milestones, say what was NOT
  tested.
