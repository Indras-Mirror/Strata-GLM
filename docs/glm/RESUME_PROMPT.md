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

## State (2026-10-08 evening) - READ THIS
- **Code written, NOT compiled or run yet** (commit after `82bf0f2`):
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
- Model download: `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/` (~26/106 GiB at 18:40, ~30-50 MiB/s; resumable:
  `FORCE_EXFAT=1 bash ~/AI/download_glm53_flash.sh /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO`). A background job
  then downloads the MTP head into `mtp/` (if that session died, run
  `hf download neuralll/GLM-5.3-Flash-MTP-GGUF --local-dir .../GLM-5.3-Flash-RCO/mtp` by hand). LoRAs in `lora/`.
- Reference fork source cloned at `~/AI/llama.cpp-glm53` (neurall/llama.cpp 2e0435a, NOT built yet).
- Relay: this session was `strata-glm`; asked `strata-ds4` (ask_id e85816a6) for a CPU gap to compile - no reply
  yet; DS4 was running ds4_generate benchmarks (GPU 74%, 23.7 GB) - compiles skew their timings, so ask first.

## Next (in order)
1. Configure + build (after DS4 says OK or is idle): CPU tree first for compile errors, e.g.
   `cmake -B build-glm -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DCMAKE_BUILD_TYPE=Release` (copy the other
   options from `~/AI/Strata-DS4/build-ds4/CMakeCache.txt`), `nice -n 10 ninja -C build-glm glm_generate_cpu`; then the
   CUDA tree like `~/AI/Strata-DS4/build-ds4-gpu` (GGML_CUDA=ON, STRATA_GGML_CUDA=ON, CUDA arch 89,
   STRATA_MMQ_KQUANTS=ON) -> `glm_generate`.
2. Build the fork (`~/AI/llama.cpp-glm53`, CUDA sm_89, `nice -n 19 -j4`) = the oracle; tokenizer via its
   `llama-tokenize`, or the upstream tokenizer.json in bench/glm-2026-10-08/upstream/ (chat_template.jinja there too).
3. Mini fixture `tools/glm/make_mini_glm.py` (glm5-next keys as in the real header: block_count, head_count_kv
   per-layer array, kda.*, ssm.conv_kernel, attention.*_mla, indexer.*, hyper_connection.*, swiglu_clamp_* arrays,
   expert_gating_func 2; tensor shapes in ARCHITECTURE.md) -> compare glm_generate_cpu logits vs the fork's
   llama-cli/eval-callback on it (CPU, no GPU needed).
4. Real model when downloaded (memguard 84 70, GPU lock, ask strata-ds4): short prompt, compare first-token logits vs
   the fork; then --dump-routes on a few prompts -> profile -> --profile + --vram-lru tok/s; quality harness port.
5. Then ARCHITECTURE.md "Decisions" 5-7 (LoRA, speed, Strata-fied MTP, indexer).

## Rules (non-negotiable; same as DS4)
- **Every full-model load through `tools/ds4/memguard.sh <cap> <need> -- <cmd>`** (cgroup RAM cap, swap off, shared
  GPU lock `~/.quetza-data/conductor/ds4-gpu.lock`). One full-model process box-wide. memguard checks MemAvailable
  BEFORE the lock, so to queue behind another engine wait first:
  `until flock -n ~/.quetza-data/conductor/ds4-gpu.lock true && [ $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) -ge 70 ]; do sleep 30; done`
- Before any GPU run: `ss -ltn | grep 8188` (ComfyUI up -> stop and tell Mal) and `nvidia-smi`.
- **GPU sharing with the DS4 session** over Quetza Relay: it is `strata-ds4`; name this session `strata-glm`
  (`relay_rename`). Before a GPU block > ~15 min, `relay_ask` it (`need: / for: / eta:`), announce when done; answer
  its asks with `relay_reply`.
- Never edit `~/AI/Strata-DS4` or `~/AI/Strata-MiMo` (other sessions' worktrees); read them and `git show` freely.
  Port shared code as your own commits on `glm`. Shared kernels (`src/kernels/`, `include/strata/kernels/`):
  backward-compatible additions only.
- If `/media/mal/SSD NVME` or `NVME1TB` is missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3` / `nvme1n1p2`.
- Never `pkill -f` a pattern in your own command line; kill only your own PIDs.
- Test the real model end to end early (quality + tok/s), fix after; keep the gates mutation-tested.
- Keep results in `bench/glm-YYYY-MM-DD/`, update `docs/glm/FINDINGS.md` + this file at milestones, say what was NOT
  tested.
