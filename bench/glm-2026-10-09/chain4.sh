#!/usr/bin/env bash
# 2026-10-09 chain 4 (~10 min): lightning indexer on the real model (REAP 25% prune, fully resident)
#  1. 2300-token prompt crossing the 2048 boundary: indexer vs forced-dense ppl (both chunked, ctx 4096)
#  2. ctx 524288 allocated, 50K-token prefill, 32-token decode at that depth
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
C=(--backend cuda --experts gpu --slots auto --vram-margin 5.5 --arena-gib 72 --arena-skip-resident --prune $D/prune-0.25.txt --prefill-chunk 1024 --chunk-mmq)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "slots auto|tier:|ppl:|prefill|decode :|experts/token|ms/token|error|fail" "$D/$name.log"; }
run ppl-chunk-fixed --ids-file bench/glm-2026-10-08/prompts/neutral.i32 -n 2 --ctx 2048 --ppl --backend cuda --experts gpu --slots auto --vram-margin 3.5 --arena-gib 72 --arena-skip-resident --prefill-chunk 128 --chunk-mmq
run x2300-idx   --ids-file $D/prompts/cross2300.i32 -n 16 --ctx 4096 --ppl "${C[@]}"
run x2300-dense --ids-file $D/prompts/cross2300.i32 -n 16 --ctx 4096 --ppl --allow-long-ctx "${C[@]}"
run long50k     --ids-file $D/prompts/long50k.i32 -n 32 --ctx 524288 "${C[@]}"
echo "=== chain4 done $(date +%T)"
