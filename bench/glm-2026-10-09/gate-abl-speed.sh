#!/usr/bin/env bash
# Abliterated decode A/B: does skip 0.15 + --renorm-skip give the ~20 t/s back WITH the full ablation?
#   ablated-skip0        : full ablation, skip 0 (the shipped slow path, ~15-18 t/s)
#   ablated-skip15renorm : full ablation + skip 0.15 + --renorm-skip (the new --full path, expect ~20)
# Decode only (generate N tokens; glm_generate prints the "decode :" line). memguard takes the GPU lock.
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
COMMON=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 524288 --vram-margin 5.5
        --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt --pcie 1.0
        --lora "$A" --lora-exps --prefill-chunk 1024 --stop 154820,154827,154829)
: > "$D/abl-speed.summary"
run() {  # tag, extra args
  local tag=$1; shift
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" "$@" \
      --ids-file "$D/prompts/eval_code.i32" -n 128 > "$D/abl-speed-$tag.log" 2>&1
  printf '%-22s exit=%s  %s\n' "$tag" "$?" "$(grep -E '^decode :|^tier:' "$D/abl-speed-$tag.log" | tr '\n' ' | ')" | tee -a "$D/abl-speed.summary"
}
echo "=== abl-speed start $(date +%T) ===" | tee -a "$D/abl-speed.summary"
run ablated-skip0        --skip-miss 0    --skip-file 0    --skip-file-prefill 0
run ablated-skip15renorm --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip
echo "=== abl-speed done $(date +%T) ===" | tee -a "$D/abl-speed.summary"
