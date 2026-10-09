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

## State (2026-10-09 LATE) - **CURRENT, read this first**
**Serving is DONE and the model is fast. The abliteration is half-applied (mechanically working now) and its
QUALITY is the open problem.** Public repo github.com/Indras-Mirror/Strata-GLM (remote `glm`, branch `glm`);
last pushes `62808ddf`, `52bd0ad3`. Commit as you go; push to `glm` (Mal asks for it).

**Run it now:** `~/.local/bin/strata-glm-quetza` (uncensored LoRA, port 8140, 512K ctx) / `--fast` (skip-miss 0.15)
/ `--stock` (8141). It starts the server (~55-70 s load) then QuetzaCodetl. Verified end to end: plain chat,
`/v1/messages` tool call, streaming + streaming tool call, 2-turn reuse, and a real harness round trip (`-p`).

**Speed (FINDINGS s18).** Serving flags at 512K ctx: **decode 20.65 tok/s, prefill ~165-175 tok/s**.
`--skip-miss 0.15` is the lever (from 9.72 at 0.05): code ppl +0.21%, chat ppl -0.97% (both inside the 1-2%
residency noise of s14) -> it is the DEFAULT in `strata-glm-unc.json` now. Context barely matters (32K = +7%).

**Serving plumbing (FINDINGS s16-s18).** `glm_generate --serve` speaks Strata's engine line protocol;
GlmDense `snapshot()/restore()` give multi-turn reuse; `tools/glm/serve/` = tokenizer export (verified exact),
`serve_glm.py` (Strata's server + GLM's tokenizer/template/tool calls), `glm-engine.sh` (memguard protections in
the foreground), configs. Two traps that cost hours:
1. **A config must NOT pass `--slots`** - Strata's server.py:490 does `int(args[index("--slots")+1])`, so a
   `--slots auto` config dies in the SERVER (traceback in the SERVER log, engine log looks silent); the engine
   default `slots=0` IS auto.
2. **Multi-turn refill**: `dense.snapshot()` can FAIL (device copy) and the serve loop then left `snap_len=-1`, so
   the "shares the last prompt" restore never fired and EVERY follow-up refilled from 0 (real traffic: 19160 then
   19317 tokens, 0 reused, ~117 s each). It now logs every decision (`serve: reuse lcp=.. hist=.. snap=.. ->
   reused=..`) and any snapshot failure. Verified: lcp 423 >= snap 421 -> 423 reused, 6.5 s -> 3.3 s. Any request
   from a DIFFERENT conversation still resets the one state, so benchmarking against a live session thrashes it.

**Metrics (FINDINGS s18).** `/metrics` (JSON, Monitor tab) carries live `thinking`/`output`; `/slots` is a full
view; `?format=prometheus` (or `Accept: text/plain`) emits llama.cpp's names (`llamacpp:prompt_tokens_seconds`,
`llamacpp:predicted_tokens_seconds`, `llamacpp:kv_cache_tokens`, `n_decode_total`, `requests_processing/deferred`)
plus `strata:*` (prefill/decode rate, phase, expert tiers, VRAM/RAM/GPU). They live in `~/AI/Strata/serve/server.py`
- that checkout's remote is the UPSTREAM **Niko1221/Strata**, so **do not push there**; the GLM shim imports it.
To put them in Mal's repo, port into `Strata-GLM/serve/server.py` (a diverged copy) and repoint the shim.

**Abliteration (FINDINGS s17, s19, s20) - WORKING.** The adapter's 135 A/B pairs = 57 "solo" (attn_output 30,
ffn_down_shexp 27, applied in the ggml graph) + 78 routed-expert (`ffn_{gate,up,down}_exps`, layers 3-28 x 288; the
README confirms the gate/up factors are structural ZEROS - only routed/shared down_proj + o_proj matter, and
alpha=1/r=1 so there is no scale to apply). `--lora --lora-exps` applies the deltas on BOTH decode (`gpu_run`) and the
non-MMQ chunk path. FOUR bugs now fixed: (i) the down correction read the float `h` the default SwiGLU path never
writes (`sw_v1 = v1 || (lora && lora->a_d)`); (ii) the chunk path silently did nothing (`kMaxEnt`=32 vs 8192 entries ->
dedicated `gp.c_exp`); (iii) `--chunk-mmq` sent prefill through the UNINSTRUMENTED MMQ path (un-ablated prefill +
ablated decode = broken model); (iv) **THE LOOP - `--skip-miss`/`--skip-file` drop a low-weight expert's read on the
miss path, and a dropped expert loses its delta, so the ablation stops being all-or-nothing and the model degenerates
into a repetition loop.** Matched A/B (effort=low, temp 0.6, top_k 20, rep 1.05): skip 0.15 -> "from ctypes import
LPVOID, LPVOID, ..." x107, finish=length, no answer; skip 0 -> finish=stop, 6740 chars, a COMPLETE coherent keylogger.
`--lora-exps` now forces `chunk_mmq=false`, `pcie_frac=1.0` AND `skip_miss=skip_file=skip_file_chunk=0`.
Verified: code ppl 3.5628 -> 3.5967 (+0.95%); benign prompts normal; **the keylogger prompt now COMPLIES with a
complete, coherent answer and terminates** (the stock model refuses it with a clean refusal trajectory; the ablated
one does not).
**Caveat:** the `skip -> 0` guard is a SOURCE edit that is compile-verified but NOT yet in the built binary - the
ds4 session held the GPU lock. The CONFIG `tools/glm/serve/strata-glm-unc-ablated.json` already has the skip flags at
0, which fixes it with the CURRENT binary; rebuild `glm_generate` (`cmake --build build-glm-gpu --target glm_generate`)
in the next GPU-free window. **Cost:** the ablated config cannot use skip-miss, so it decodes at the pre-skip rate
(~15-18 tok/s in these probes) - correctness over speed. **Still untested:** the other refusal categories, adapter v1,
and the reference cross-check (`~/AI/llama.cpp-glm53` glm5next LoRA on the same GGUF+adapter - the way to prove our
application matches upstream).

**30+ tok/s = MTP - and the prior evidence says be careful.** `FINDINGS s7`: "a 2-token verify touches 15.2 distinct
experts -> MTP ~0.9x here (memory-bound)". Plan: `ARCHITECTURE.md:78` (Strata-fied MTP: the block's dense part
resident, its 288 experts through the tier, `--mtp-resident`); head at
`/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/mtp/GLM-5.3-Flash-MTP-Q4_K.gguf` (4.58 GB, depth-1 acceptance 70-75%).
Strata already has `include/strata/core/{mtp,verify,coupled_draft,native_head}.hpp` + the tier's multi-token
`gpu_run_n`. **The skip-miss sweep is DONE** (decode, 64 tok, ctx 524288, serving flags): 0.05 -> 9.72,
0.15 -> 20.65, 0.25 -> 24.44, **0.35 -> 28.70 tok/s** - so **30+ is reachable by skip-miss alone** (~0.4-0.5), at a
growing quality cost that MUST be gated (skipped weight 248.9 -> 260.1/token; ppl at 0.15 was +0.21% code /
-0.97% chat - only 0.15 has been gated so far). `bench/glm-2026-10-09/dec-0.25.log` / `dec-0.35.log`.
**The tier profile changed with the lever**: at 20 tok/s a token is COMPUTE-bound (wall ~19 ms = gpu hits 9.17 +
cpu pool 5.23) with the CPU/PCIe misses nearly gone (cpu 2.1%, pcie 1.1%, file tier 19) - unlike s7's
memory-bound 8.5 tok/s, which is the case the "0.9x MTP" verdict was measured in. Re-derive the MTP economics
from a CURRENT profile before investing. It is still a large build: load the 4.58 GB head, its own DSA layer +
MoE + eh_proj, then draft/verify plumbing (Strata's `verify.hpp`/`coupled_draft.hpp` and the tier's `gpu_run_n`
are the starting points).

**Traps that cost real time**
- **Never `pgrep`/`pkill -f` a pattern that also appears in your OWN command line** - it matched this shell several
  times (`glm_generate --serve`, `serve_glm.py`, and a bracketed pattern whose literal text sat elsewhere on the
  same line). Use `fuser -k 8140/tcp`, or a /proc scan that skips `$$` and `$PPID`.
- `--ppl` slows prefill a lot: measure speed WITHOUT it. Chunked ppl is only comparable at the SAME margin
  (residency moves it 1-2%, s14).
- GPU shared with `strata-ds4-gpu` (relay); one full-model process box-wide; `memguard.sh` takes the lock itself -
  never wrap a chain in `flock` on it (deadlock).

## State (2026-10-09 morning) - superseded by the block above
- **First action: `relay_rename` -> `strata-glm`** (the DS4 session `strata-ds4-gpu` addresses us by that name).
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
- **DONE (FINDINGS s16): serving + Quetza wrapper `~/.local/bin/strata-glm-quetza`.** Engine `glm_generate --serve`
  + GlmDense snapshot/restore + tools/glm/serve/ (tokenizer verified exact, Strata server shim, configs, foreground
  memguard launcher). The "silent" first start was Strata's server doing `int(args[args.index("--slots")+1])` on our
  `--slots auto`: **drop `--slots` from the configs** (engine default `slots=0` = auto). Verified end to end at ctx
  524288: plain chat, `/v1/messages` tool call, 2-turn reuse (`31 reused`), streaming + streaming tool call; decode
  ~8-9 tok/s. Wrapper starts the server (uncensored LoRA 8140 / stock 8141) then Quetza. Not yet run as a full
  Quetza session. Open: serve-mode `--vram-grow` shrink-before-prefill.
- **NEXT: the same for DS4** (`ds4_generate --serve` in `~/AI/Strata-DS4`, the strata-ds4-gpu session's tree - do not
  edit it) + a wrapper; coordinate with strata-ds4-gpu over the relay.
- **Best real-use config (FINDINGS s15):** `--prune bench/glm-2026-10-09/prune-ezct-0.25.txt --prune-penalty 0.05
  --arena-adapt --skip-file 0.15 --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35 --prefill-chunk 2048
  --chunk-mmq --vram-margin 8 --vram-grow 1 --arena-skip-resident --arena-gib 72 --slots auto` -> prefill ~220-230,
  decode ~9.8 tok/s in one process; German ~= unpruned. Measure real use WITHOUT --ppl (it slows prefill).
- **Soft prune DONE (FINDINGS s10-s12):** GLM's sigmoid router needs penalty ~0.05 (not DS4's 0.5). Recommended:
  `--prune bench/glm-2026-10-09/prune-ezct-0.25.txt --prune-penalty 0.05 --arena-adapt --skip-file 0.15
  --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35` -> decode ~10.8 tok/s (hard prune 9.5), prefill 130-140, German +2% vs unpruned (hard +61%),
  code -1.7%. `--arena-admit` (heat gate) is a loss on GLM: leave off. Clean base: chat 5.547, code 3.668.
- **Next (in order):** (1) more speed: chunk 2048 = 160 tok/s prefill but needs margin ~8 -> VRAM re-seed / elastic ExpertCache
  (shrink/grow exists, --vram-elastic) so decode keeps margin-1 slots; prestage = no gain (s14); gate quality with the 2000-token evals (chunked) + the decode-loop chat
  gate (chain10; loop vs loop only). Prompt cache across turns (DwarfStar idea, FINDINGS s13) once GLM is served.
  (2) **Numerics (s14): resident (MMVQ) vs streamed (MMQ) experts move chunked ppl 1-2%** - compare at equal
  margin; find the off path with the fork oracle: rebuild ~/AI/llama.cpp-glm53 CPU (patched, `GLM_NO_FUSED_LID=1`) and compare indexer scoring on the
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
