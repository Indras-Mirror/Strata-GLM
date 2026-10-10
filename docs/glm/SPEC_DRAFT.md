# Speculative decode with GLM's NextN draft block (the "PolyStrata-style" speedup) - port plan

Goal: get the ~2x decode that PolyStrata reports for **our exact file** (`GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf`,
113.6 GB): their table is **26.7-27.7 t/s without the draft block -> 47-51 t/s with it** (and 55-62 t/s in a later
build; 78-90% drafts accepted). The mechanism is MTP (NextN) speculative decoding, which we currently reject.

## What we already have (most of the machinery)

Our tree already carries the full spec stack - it is just **not wired into the GLM generation path**:
- `include/strata/spec/controller.hpp` + `src/spec/controller.cpp`: picks `none | lookup | MTP` and `k` each step,
  maximizing `E[tokens committed] / T(step)` with a `min_gain` (5%) gate.
- `include/strata/spec/draft_policy.hpp` + `src/spec/draft_policy.cpp`: per-verify-round policy (MTP window vs lookup
  window by E/cost). **Ours is a SUPERSET of project-maya's** - we have `chain()` / `observe_chain` / `chain_rate()` /
  `stale()` and the chained-lookup counts that maya's `draft_policy.hpp` lacks.
- `draft_source.hpp` + `suffix_drafter.*` + `PromptLookupSource` (prompt-lookup drafter).
- `src/core/mtp.cpp` (1501 lines) + `include/strata/core/mtp.hpp`: our existing MTP (DS4-shaped).

**The economics are already modeled**: `CostModel.distinct_ratio` (1.0, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80,
5.2 for n=1..9) encodes exactly the s21 finding - a k-token verify touches up to 5.2x more **distinct** experts, so on
a bandwidth-starved box the controller picks `k = 0`. **That is why we rejected MTP, and the same model predicts it
will reject it again on our DDR4 box** unless the measurements say otherwise. So this port is a *measurement*, not a
guaranteed 2x.

## What is missing

1. **The GLM draft block (NextN layer).** Our file has none (`max blk = 44`, no `nextn/mtp` tensors). GLM's NextN is
   the `mtp.*` tensor set in zai-org's release: `mtp.fc_embedding.weight` [2560,2560], `mtp.fc_hidden.weight` [2560,2560],
   `mtp.hyper_connection_mixer.hc_norm.weight` [10240], `shared_head`, and a full MoE layer. PolyStrata fetches them
   (`tools/mtp_fetch.py`, pinned revision `de4b8e4d...`, sha256-verified) -> `mtp_pack.py` -> `mtp-q2_0.gguf` ->
   `mtp_rt.py`. project-maya carries the same block as **`blk.45`** inside its GGUF (recipe: `blk.45.ffn_{gate,up}_exps`
   Q2_K, `blk.45.ffn_down_exps` Q3_K) - i.e. **the Maya-S-v2 we are downloading already contains it**.
2. **A GLM NextN forward** in our engine (the DS4 `mtp.cpp` is shaped for DeepSeek's MTP; GLM's is its own:
   fc_embedding/fc_hidden/hyper_connection_mixer/one MoE layer/shared_head).
3. **Wiring**: the Controller + draft source into `glm_generate`'s decode loop and the verify pass (currently `glm_generate`
   does not reference `strata::spec` at all).
4. **A GLM CostModel** (`dense_ms`, `dense_ratio`, `cpu_all_miss_ms`, `hit_rate`, `distinct_ratio`, `extra_use_cost`,
   `sync_ms`, `mtp_draft_ms`) measured **on our box** - the current defaults are the 2026-09-23 DS4/Qwen measurements.

## Plan

| step | what | GPU? |
| --- | --- | --- |
| A | **Acquire the draft block**: extract `blk.45.*` from Maya-S-v2 (or `mtp_fetch` zai-org's `mtp.*`) -> a small standalone draft GGUF | no |
| B | **GLM NextN forward**: load the draft GGUF; run fc_embedding -> fc_hidden -> mixer -> 1 MoE layer -> shared_head; predict the token at t+2 | write only; GPU to test |
| C | **Wire the Controller + draft into `glm_generate`** (and the server): choose k, run the (k+1)-token verify, observe acceptance | write only |
| D | **Calibrate the CostModel on our box** (measure dense_ms, distinct_ratio, extra_use_cost) | yes |
| E | **Gate**: token-identical to the plain loop (maya's own discipline: "token-identical to the plain loop"), then decode tok/s vs the 16-25 baseline | yes |

## Honest risks

- **Our box may just say no.** s21 measured the batched verify as a 0.18-0.31x loss; the Controller exists precisely to
  decide this per-machine. PolyStrata's 2x is on a **5090 (1.8 TB/s) + 120 GB DDR5 8-channel**; ours is a 4090 (1 TB/s)
  + 90 GB **DDR4** → the expert-fetch cost of the verify is far higher here. Realistic outcomes: (a) controller picks
  k=1-2 → a small gain, (b) controller picks k=0 → no gain (and that is a *result*, not a failure).
- The lookup/suffix drafter alone is **not** the win: maya measured it **losing 2-8%** on ordinary text vs MTP. The
  draft block is what makes PolyStrata's number.
- PolyStrata also quantizes activations to 8 bits between layers and ships an expert profile; those are separate gains.

## Note on the draft-block artifact

PolyStrata's "3 GB draft block" is a **separate file** from the model GGUF (their doc: "3 GB for its draft block").
So we do not need to touch our model file - we load the draft block alongside it, the way `mtp.cpp` already loads DS4's
MTP from `dense.bin`.

## Sources

- PolyStrata `docs/GLM.md` (the 47-51 vs 26.7-27.7 table) and `docs/BACKENDS.md`; `tools/mtp_fetch.py`/`mtp_pack.py`.
- project-maya `tools/maya_quant/recipes/maya-s-v2.json` (the `blk.45` recipe) and `src/core/mtp.cpp`, `src/spec/`.
- PolyStrata `bench/results/*/mtp-manifest.json`: `mtp.fc_embedding`, `mtp.fc_hidden`, `mtp.hyper_connection_mixer.*`.
