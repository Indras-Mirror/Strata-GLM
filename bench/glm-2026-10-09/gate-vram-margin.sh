#!/usr/bin/env bash
# Residency lever test: does a smaller --vram-margin buy more VRAM expert slots (and more decode)?
# Runs the SAME ablated decode as gate-abl-speed.sh, at margin 5.5 (default) vs 1.5, and prints the
# tier line (slots) + the decode line. Waits for the ds4-gpu lock.
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
BASE=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 524288
      --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt --pcie 1.0
      --lora "$A" --lora-exps --prefill-chunk 1024 --stop 154820,154827,154829
      --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip)
: > "$D/vram-margin.summary"
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 30; done
for tag in ctx32k-m1.5 ctx512k-m1.5; do
  MARGIN=1.5; CTX=32768; [ "$tag" = ctx512k-m1.5 ] && CTX=524288
  echo "--- $tag (vram-margin $MARGIN) $(date +%T) ---" | tee -a "$D/vram-margin.summary"
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${BASE[@]}" --vram-margin "$MARGIN" --ctx "$CTX" \
      --ids-file "$D/prompts/eval_code.i32" -n 128 > "$D/vram-margin-$tag.log" 2>&1
  grep -hE "slots auto|^tier:|^decode :" "$D/vram-margin-$tag.log" | tee -a "$D/vram-margin.summary"
done
echo "=== vram-margin done $(date +%T) ===" | tee -a "$D/vram-margin.summary"
