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

## State (2026-10-08)
- No GLM code yet. Branch `glm` = DS4 engine (Ds4Dense + Ds4MoeTier, mHC, MLA, clamped SwiGLU, CUDA graphs,
  chunked MMQ prefill incl. Q2_K/Q3_K, VRAM LRU, split direct reads) - all reusable.
- Model download STARTED 2026-10-08 ~18:01 to `/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/` (exFAT; Mal chose it).
  Check: `tail -3 /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/download.log` and the file
  `GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf` (complete = 113,59x,xxx,xxx bytes and "sha256 ok" in the log). If it
  died, re-run `FORCE_EXFAT=1 bash ~/AI/download_glm53_flash.sh /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO`
  (resumes). First 10 min ran 8-15 MiB/s while another download shared the link.
- Abliteration LoRAs downloaded to `.../GLM-5.3-Flash-RCO/lora/` (PLAN.md lists them; default = GCSA Abliterix v2).

## Next (in order)
1. **Reference oracle** (CPU only, can start before the model lands): clone + build `neurall/llama.cpp` into
   `~/AI/llama.cpp-glm53` (CUDA, sm_89). Don't build while another session is benchmarking (ask on the relay).
   The fork is already cloned there (source read in full - ARCHITECTURE.md); only the build is left.
   Point STRATA's ggml at a copy of DS4's `third_party/llama.cpp` (it has every op GLM needs).
2. Answer ARCHITECTURE.md "Open questions", then the mini glm5-next fixture + `GlmDense` phase 1 (dense MLA).
3. When the model is complete: baseline the fork on this box (decode/prompt tok/s, ppl wikitext-2 40x512, through
   memguard + GPU lock), golden dumps for the gates.
4. Then PLAN.md "Port order" steps 2-7.

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
