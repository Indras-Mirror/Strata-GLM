#!/usr/bin/env bash
# MTP economics probe. The verify in speculative decoding is a BATCHED forward of k+1 tokens, so measure the
# cost of a small batched forward (the prefill-chunk path at a small chunk size) and compare its per-token cost
# to a sequential decode.  r_verify(k) = per_token(chunk k) / per_token(decode); then
#   speedup ~ E[accepted] / (k * r_verify + k * r_draft),  E[accepted] = (1 - a^(k+1)) / (1 - a).
# No MTP code needed - this decides whether the head is worth wiring.  Run with the GPU free (memguard takes the lock).
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
D=bench/glm-2026-10-09
COMMON=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 524288 --vram-margin 5.5
        --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt
        --skip-file 0.15 --skip-file-prefill 0.15 --skip-miss 0.15 --pcie 0.35 --stop 154820,154827,154829)
python3 - "$D/prompts/eval_code.i32" "$D/prompts/eval96.i32" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read()
open(sys.argv[2], 'wb').write(a[:4 * 96])
print("wrote", sys.argv[2], "(96 tokens)")
PY
echo "### decode baseline (sequential, no batch) $(date +%T)"
bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" --prefill-chunk 1024 --chunk-mmq \
    --ids-file "$D/prompts/eval_code.i32" -n 64 > "$D/econ-decode.log" 2>&1
echo "  exit $?"; grep -E "decode :|tok/s|decode ms/token" "$D/econ-decode.log" | tail -3
for K in 6 8 16; do
    echo "### chunk $K (a batched forward = the verify) $(date +%T)"
    bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${COMMON[@]}" --prefill-chunk "$K" --chunk-mmq \
        --ids-file "$D/prompts/eval96.i32" -n 1 > "$D/econ-chunk$K.log" 2>&1
    echo "  exit $?"; grep -E "prefill:|prefill chunks of" "$D/econ-chunk$K.log" | tail -2
done
echo "=== done $(date +%T) ==="
