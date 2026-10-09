#!/usr/bin/env bash
# 30+ decode. Bounded: each skip-miss value is one model load, so callers pass just the values they want.
#   run-30plus.sh speed 0.45 0.50     -> decode tok/s per value
#   run-30plus.sh ppl   0.45 0.50     -> decode-LOOP ppl (the decode path is what skip-miss changes; loop vs loop)
# The caller MUST have the GPU free: memguard takes ds4-gpu.lock itself, and memguard checks MemAvailable BEFORE the
# lock, so wait first (never wrap in flock - deadlock).
set -u
cd "$(dirname "$0")/../.."
MODE=${1:-speed}; shift || true
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
D=bench/glm-2026-10-09
EVAL=$D/prompts/eval_code.i32        # 1999 positions
EVAL300=$D/prompts/eval_code300.i32  # short, for the (loop) ppl gate
COMMON=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 524288 --vram-margin 5.5
        --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt
        --skip-file 0.15 --skip-file-prefill 0.15 --pcie 0.35 --stop 154820,154827,154829)
for SM in "$@"; do
    echo "### $MODE skip-miss $SM  $(date +%T)"
    if [ "$MODE" = speed ]; then
        bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" --skip-miss "$SM" \
            --prefill-chunk 1024 --chunk-mmq --ids-file "$EVAL" -n 64 > "$D/dec-$SM.log" 2>&1
        echo "  exit $? $(date +%T)"; grep -E "decode :|tok/s|experts/token" "$D/dec-$SM.log" | tail -3
    else
        # no --prefill-chunk => the decode loop, which is the path skip_miss applies to
        bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" --skip-miss "$SM" --ctx 4096 \
            --ids-file "$EVAL300" -n 400 --ppl > "$D/ppl-loop-$SM.log" 2>&1
        echo "  exit $? $(date +%T)"; grep -E "^ppl:" "$D/ppl-loop-$SM.log" | tail -1
    fi
done
echo "=== done $(date +%T) ==="
