# Parallel work on GLM-5.3-Flash / Strata: what is portable to Strata-GLM

Survey (2026-10-10) of the community engines that overlap Strata-GLM, and exactly what to borrow. Nothing here was
built by us; every claim is from the peer's own README/docs/bench (links at the bottom). No GPU needed to read or port.

The three that matter, in order: **project-maya** (same upstream + same model, MIT), **bodhi37/strata** (the SSD-tier
engineering), **PolyStrata** (ordinary-GGUF GLM + lossless spec). antirez/ds4 is a sibling to compare against.

---

## 1. project-maya (`mw00/project-maya`, MIT) - the big one

An engine for **GLM-5.3-Flash alone**, started from `Niko1221/Strata`'s engine and rewritten for GLM (expert tiers
across VRAM/RAM/SSD, two-GPU split, MTP decoding), plus its own server/dashboard. Same upstream as us, so the code
shapes are familiar. It solves three of our open problems.

### 1a. A quant toolchain that works FROM FP8 - answers our s24 blocker

`tools/maya_quant/`: `quantize.py`, `gptq.py` (error-feedback rounding), `calib.py` / `prep_calib.py`, `hf_map.py`,
`mtp_gguf.py`, `draft_vocab.py`, + `recipes/*.json`, + the eval harness (1e). The README/MODEL_CARD say it is made
**from Z.ai's own FP8 release** (not from a re-quant), with **per-expert input statistics from the FP8 model** and
GPTQ-style error feedback on the experts.

Recipe `maya-s-v2.json` (rules, first match wins; default Q6_K):
```
blk.45 gate/up        -> Q2_K     (the NextN/MTP draft block)
blk.45 down           -> Q3_K
ffn_(gate|up)_exps    -> IQ2_XXS  (all 42 layers, error-feedback rounded)
blk.(3-6|41-44) down  -> IQ3_XXS  (first + last four MoE layers, one step up)
ffn_down_exps         -> IQ2_S    (the rest)
ffn_gate_inp|exp_probs_b|hc_*|indexer_compressor_ape -> F32
ssm_(beta|f_a|f_b|g_a|g_b)|indexer -> Q8_0
default (attention, shared experts, dense, embeddings, output) -> Q6_K
```
Their shipped sizes + measured quality (Maya-S ~90-96.5 GB):
- **97.9% of the FP8 model's zero-shot accuracy** (ARC-E/C, HellaSwag, WinoGrande, PIQA, 400 q each); PPL 4.19 vs FP8
  3.51; same top token as FP8 83.3%; **no loop in 14 long (6k-14k token) answers**; a fact found at 8k/16k/30k ctx.
- Maya-M 116 GB (IQ2_S g/u + IQ3_XXS down + IQ3_S sensitive): 23% lower KL than Maya-S.
- Maya-L 156 GB (IQ3_S g/u + IQ4_XS down + Q5_K sensitive): **99.2% of FP8**, KL 0.188.

**What we can use:** their recipes are a *reference allocation* finer than ours (we run q2_k experts + q4_k dense), and
the *toolchain* is the thing that lets us make our own quant **from the FP8-uncensored source** (`orcarouter`/`dealignai`)
instead of needing a BF16 base - i.e. the abliterated+small model from s24 becomes: ablate the FP8, apply a Maya-style
recipe. MIT, so portable; we'd adapt `hf_map.py` to our tensor names.

### 1b. MTP/spec decode - RECONCILES our s21 rejection

`src/core/mtp.cpp`, `src/spec/{controller,draft_policy,suffix_drafter}.cpp`, `include/strata/core/mtp.hpp`,
`include/strata/spec/*.hpp`. Their GLM result (`bench/results/2026-10-06-glm-spec/README.md`, 2x V100-32GB):
- keeps the **NextN (MTP) draft block blk.45**; draft == next greedy token **64-68% teacher-forced, 83-99% free-gen**;
- **pipelined** speculative decode: the **head half drafts the token after next while the tail finishes the current
  one**; recurrent states saved/restored on a miss; "token-identical to the plain loop";
- **26 -> 36-40 tok/s**, 97-99% drafts accepted (split 22/24); auto +2 head layers when the draft runs.

**Why our s21 result stands and theirs does not contradict it:** s21 measured a **batched k-token verify** (the whole
model forward over k+1 tokens) on **one GPU** - the MoE expert fetch scales with k, so r_verify = 1.25-1.7x. Theirs is
**not a batch**: the draft's compute on the head half overlaps the tail half's verify. It needs the two-GPU head/tail
split; their **single-GPU** number (Uranus, 1x V100) is 6.2-7.2 t/s and disk-bound, with single-GPU spec explicitly
"Part 2" (unfinished). So: MTP pays **only with head/tail pipelining**, which on our single 4090 means the untested
version - overlap the draft's compute with the expert *fetch* - not a batched verify.
Their own negative: a **suffix/lookup drafter** (prompt n-gram lookup) "lost 2-8% on ordinary text" vs MTP; the
`DraftPolicy` learns online which to use (expected committed tokens per ms, E/cost over MTP vs lookup) - "it only
chooses which drafts to verify: the output is unchanged."

### 1c. Expert tiers, cache policy, prefill

- Three-tier (VRAM hot pool / RAM tier / disk, staged through a **ring**), per-layer VRAM slots sized by **routing
  share** (`expert_counts.txt`), MMQ experts run **in place on the VRAM pool** in the prompt path.
- **Cache-policy finding (contradicts our arena_adapt):** replaying 1000 tokens of routes, **LRU = LFU (97.7/96.4%
  per GPU) and even Belady's optimum only reaches 98.6%** - "the misses are capacity (30 GB RAM), not policy." Worth
  re-examining our TinyLFU `--arena-adapt` investment.
- **Prefill:** batched layer-major (`glm_prefill.cu`), cuBLAS FP16 projections, MMQ experts in place, RAM/disk experts
  staged by a background thread, **prompt buffers borrowed from the expert pool's tail** -> prompt 31-55 -> **4.7
  ms/token (213 tok/s)** on 2x V100, 2.6k-token prompt 91 s -> 12 s. Two GPUs pipelined.
- Tried and **NOT** faster (kept off): reading predicted disk-only experts ahead in decode (`STRATA_GLM_AHEAD` - only
  ~40% of disk misses are predictable within 4 layers, the wasted reads share the one NVMe); CPU expert path slower
  than the PCIe pull on a Xeon (1.4 vs 0.56 ms/expert). Both echo our own residency notes.

### 1d. Silent-corruption bug classes - worth adding to our gates

- **"DSA attention was silently zero since e7570df":** the split-K MLA kernel asked for a 96 KB shared-memory opt-in,
  which fails on V100; **every launch errored but attention silently produced zero**. They made kernel launch errors
  sticky (the forward now errors out). *A silent-zero attention that a ppl/loop gate might not catch.*
- **"one between-token table batch could edit a key twice (made live, then freed)":** the parallel update kernel let
  either value win, so the device could later fetch **another expert's bytes**. Found by "an A/B that should have been
  token-identical". *A token-identical A/B across tier states is a cheap, high-value gate we don't run.*

### 1e. An eval harness we can adopt

`tools/maya_quant/`: `kl_eval.py` (KL(FP8||quant) over top-64 on held-out text + same-top-token %), `zs_*.py`
(zero-shot task accuracy vs FP8), `loop_test.py` (6k-14k-token answers, temperature 1.0 and greedy - the loop check),
`ctx_fill_test.py` (a fact at 8k/16k/30k), `ref_check.py`, `canonical_xcheck.py`. Their `bench/results/` has the
methodology and numbers. **This is a far better quality gate than our repo-ppl**, and directly reusable.

---

## 2. bodhi37/strata (`orca-port` branch, frozen) - the SSD-tier engineering

A deep fork of `Niko1221/Strata` for boxes that cannot hold the model in RAM. Even frozen, its design notes are the
closest reference to our file tier:
- **Three-tier expert serving** (`expert_source.cpp`, `expert_cache.cpp`): VRAM + **mlocked hot-RAM tier** + **NVMe via
  a `pread` ring + reader pool**; trace-derived routing profiles (`--dump-profile`/`--dump-counts` +
  `make_profile_from_counts.py`); `--expert-cache -2` baseline; memlock/nice wrapper.
- **A 6.4x read-amplification fix:** pre-R5 disk reads faulted 4 KiB via the mapping and it re-`WILLNEED`ed 96,231
  blobs for 15,007 real misses; fixed with a hot-tier guard, whole-blob `pread` + `DONTNEED` + `MADV_HUGEPAGE`,
  two-pass dispatch (`begin_layer`/`wait_layer`: resident experts compute while the layer's reads fly), and a ring
  invalidated at layer 0. **"Remainder wall: drive QD1 latency (~2.8 ms) x 48 serial layers, not bandwidth" (~10-11
  tok/s decode).**
- **`O_DIRECT` expert reads** (twin fd + 4 KiB-aligned bounce) to kill the per-read page-cache cycle (~39% CPU in
  kernel page management); `STRATA_NO_ODIRECT=1` A/B arm.
- **Elastic tier / pressure governor:** `cudaHostRegister` pins pages (unswappable) -> a 24 GiB registered + mlocked
  tier + KV + desktop = zero headroom -> swap storm -> `cublasCreate` death. Fix: register in ~2 GiB whole-slot slices,
  `shed_pressure`/`regrow_pressure` (score-based, hysteresis on `MemAvailable`). **This is the failure our memguard
  guards against from outside; they solve it inside.**
- **Prefill by tiling, not bigger chunks** (P-CHUNK): warm 11k prefill 50 -> 694 tok/s.
- Kernels: **AVX-512 multi-token IQ4_NL down** (3x per-token ggml at nt=5), Orca IQ4_XS decode-once, `native_moe`
  combine (per-block smem weight broadcast + `__ldg` hygiene).
- **Bug classes:** a `vpsignb(g,g)` sign bug meant **weight signs were never applied** - exposed only by *repetitive
  output* (float parity hid it); and an OOB in the mix kernel where `m.mixed` was tile-sized but indexed with the chunk
  offset `t0` -> **"!!!-forever degenerate output on any chunk > tile"**.

---

## 3. PolyStrata (`VecSzn/PolyStrata`, built on `Niko1221/Strata`)

Runs **GLM 5.3 Flash from ordinary GGUFs, 79-114 GB**, most staying in RAM; one NVIDIA card. Lossless speculative
decode: "guesses a few tokens ahead, from **the model's own draft block or from a repeat of the conversation**, and
checks the guesses in one pass. With every expert on the CPU the answer is the same tokens with and without guessing."
(The conversation-repeat drafter is the same idea as maya's `suffix_drafter`; maya measured it *losing* to MTP on
ordinary text, so treat PolyStrata's version as the case to check, not a win.)
- Their supported GLM files include **`patrickbdevaney` REAP50-Q3_K_M (79 GB - half the routed experts removed)** -
  a smaller-GLM lead (we prune 25%).
- Their post references *strata-glm* (us): "the same person actually tried it with their other project (strata-glm)
  and ... had to put experts on NVMe" - so our file tier is a known quantity to them.

---

## 4. antirez/ds4 (DwarfStar)

Small native C engine, **now supports GLM 5.x Flash** and Qwen3.8 Flash Next; Metal main target, CUDA/DGX Spark,
multi-GPU, ROCm Strix Halo; ships its own server/tools/coding agent. **Only loads GGUFs the project itself produces.**
A sibling engine to compare architecture against; not a drop-in.

---

## 5. Reconciliation with our open items

| our item | the peers' answer | action |
| --- | --- | --- |
| **MTP rejected** (s21, 0.18-0.31x) | maya: MTP pays **only head/tail-pipelined on 2 GPUs** (26->40 t/s, token-identical); single-GPU unfinished. Our batched-verify rejection stands for 1 GPU. | keep rejected for a batched verify; the *overlap draft-compute with the expert fetch* on one GPU is the open test |
| **quant from FP16/FP8** (s24) | maya does it **from FP8** with GPTQ error-feedback + per-expert FP8 stats -> 97.9% recovery; toolchain is MIT | port `maya_quant` (recipe + `hf_map` to our names) to make our own abliterated GGUF from the FP8-uncensored source |
| **file tier / NVMe spill** | bodhi37: pread ring, `O_DIRECT`, the 6.4x read-amp fix, the "QD1 latency x serial layers" wall; maya: ring staging, prompt buffers from the pool tail | borrow the read-amp fix + `O_DIRECT`; measure our own QD1 wall |
| **expert-cache policy** (`--arena-adapt`) | maya: policy is **not** the lever (LRU = LFU ~ Belady 98.6%); capacity is | re-examine the TinyLFU work; spend on capacity/residency instead |
| **quality gate** | maya's `kl_eval` / `zs_*` / `loop_test` / `ctx_fill` harness | adopt at least `loop_test` (the s20 class) + KL-vs-source |
| **ability to abliterate** | none of them do it | our runtime LoRA + `--renorm-skip` is the differentiator - keep it |

## 6. Recommended next steps (no GPU)

1. **Read `project-maya`'s `tools/maya_quant/` end to end** (`quantize.py`, `gptq.py`, `calib.py`, `hf_map.py`, a
   recipe) and write our adaptation plan: FP8-uncensored source -> abliteration -> Maya-style recipe -> our GGUF.
2. **Port the eval harness** (at minimum `loop_test.py` + a KL-vs-source check) so our quant/skip changes gate on task
   quality and looping, not just repo-ppl.
3. **Read `project-maya`'s `src/spec/` + `mtp.cpp`** and decide if the one-GPU "overlap draft with fetch" variant is
   worth building (it would be a *new* experiment vs s21's batched verify).
4. **Read `bodhi37/strata`'s `expert_source.cpp` + the read-amp fix** and port the `O_DIRECT` + whole-blob-pread +
   hot-guard pattern to our file tier.
5. Add the two silent-corruption gates: **token-identical A/B across tier states**, and **sticky kernel-launch
   errors** (maya's silent-zero-attention).

## Links

- project-maya: https://github.com/mw00/project-maya · quants: https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF
- bodhi37/strata (orca-port): https://github.com/bodhi37/strata
- PolyStrata: https://github.com/VecSzn/PolyStrata
- antirez/ds4 (DwarfStar): https://github.com/antirez/ds4
- upstream Strata: https://github.com/Niko1221/Strata
