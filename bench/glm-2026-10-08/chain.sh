#!/usr/bin/env bash
# One watcher for the GLM GPU window: wait for the shared ds4-gpu.lock, build both trees, run the warm-cache
# baseline, then the LoRA A/B.  A failed build leaves the previous binary in place, so the warm run still runs.
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
D=bench/glm-2026-10-08
LOG=$D/chain.log
exec >>"$LOG" 2>&1
echo "=== chain start $(date '+%F %H:%M:%S') ==="

# DS4 holds the lock for its 128K --comp-q8 run; its parked queue waits for glm-gpu-done, so nothing competes.
while fuser ~/.quetza-data/conductor/ds4-gpu.lock >/dev/null 2>&1; do sleep 20; done
echo "=== lock free $(date '+%H:%M:%S') ==="
sleep 3

echo "--- ninja build-glm (CPU) ---"
ninja -C build-glm glm_generate_cpu && echo "CPU BUILD OK" || echo "CPU BUILD FAILED"

echo "--- ninja build-glm-gpu (CUDA) ---"
BUILD_OK=0
ninja -C build-glm-gpu glm_generate && { echo "CUDA BUILD OK"; BUILD_OK=1; } || echo "CUDA BUILD FAILED"

echo "--- warm-cache baseline ---"
bash "$D/run_warm.sh" || echo "WARM FAILED"

if [ "$BUILD_OK" = 1 ]; then
    echo "--- lora A/B ---"
    bash "$D/run_lora.sh" || echo "LORA FAILED"
else
    echo "--- lora A/B SKIPPED (no fresh binary) ---"
fi
echo "=== chain done $(date '+%F %H:%M:%S') ==="
