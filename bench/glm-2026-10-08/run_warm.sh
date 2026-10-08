#!/usr/bin/env bash
# Warm-cache baseline for Strata-GLM (2026-10-08 night).
#   run 1 (cold): a 690-token real prompt, dumps a routing profile (routes.bin)
#   run 2 (warm): the same prompt, --profile routes.bin --vram-lru
# Both go through memguard (cgroup RAM cap + the shared ds4-gpu.lock).
set -euo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B="${GLM_BIN:-build-glm-gpu/glm_generate}"
D=bench/glm-2026-10-08
COMMON=(--backend cuda --experts gpu --slots auto --arena-gib 60 --ctx 2048 --vram-lru)
EXTRA=("$@")

echo "### run 1/2: COLD + --dump-routes ($(date '+%H:%M:%S'))"
bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" --ids-file "$D/prompts/neutral6x.i32" -n 16 \
    --dump-routes "$D/routes.bin" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$D/run-cold.log"
ls -l "$D/routes.bin"

echo "### run 2/2: WARM --profile ($(date '+%H:%M:%S'))"
bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" --ids-file "$D/prompts/neutral6x.i32" -n 16 \
    --profile "$D/routes.bin" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$D/run-warm.log"

echo "### done ($(date '+%H:%M:%S'))"
