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
