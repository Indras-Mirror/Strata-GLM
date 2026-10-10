#!/usr/bin/env bash
# Residency lever test: context size x --vram-margin -> VRAM expert slots -> decode.
# All arms at margin 1.5, ablated (skip 0.15 + --renorm-skip), same 128-token decode. Waits for the ds4-gpu lock.
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
BASE=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --allow-long-ctx
      --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt --pcie 1.0
      --lora "$A" --lora-exps --prefill-chunk 1024 --stop 154820,154827,154829
      --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip)
: > "$D/vram-margin.summary"
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 30; done
for spec in "ctx32k:32768" "ctx300k:307200"; do
  tag="${spec%%:*}"; CTX="${spec##*:}"
  echo "--- $tag (ctx $CTX, margin 1.5) $(date +%T) ---" | tee -a "$D/vram-margin.summary"
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${BASE[@]}" --vram-margin 1.5 --ctx "$CTX" \
      --ids-file "$D/prompts/eval_code.i32" -n 128 > "$D/vram-margin-$tag.log" 2>&1
  grep -hE "slots auto|^tier:|^decode :" "$D/vram-margin-$tag.log" | tee -a "$D/vram-margin.summary"
done
echo "=== vram-margin done $(date +%T) ===" | tee -a "$D/vram-margin.summary"
