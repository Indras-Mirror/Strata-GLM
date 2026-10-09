#!/usr/bin/env bash
# 2026-10-09 chain 14 (~3 min): REAL-USE decode with the elastic VRAM cache (--vram-grow 1) - chunked prefill then decode in one process (best config): margin 5.5 / chunk 1024 vs margin 8 / chunk 2048. Decode refs at margin 1: 10.82 tok/s.
# NOTE: memguard.sh takes ds4-gpu.lock per run - never wrap this script in another flock on that lock.
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
P=$D/prune-ezct-0.25.txt
C=(--backend cuda --experts gpu --slots auto --arena-gib 72 --arena-skip-resident --ctx 2048)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "grow_cache|skipped:|tier:|ppl:|prefill chunks|OOM|out of memory|prefill:|decode :|experts/token|tier moves|decode ms|error|fail" "$D/$name.log"; }
R=(--prune $P --prune-penalty 0.05 --arena-adapt --skip-file 0.15 --skip-file-prefill 0.15 --skip-miss 0.05 --pcie 0.35)
run c14-m55c1024 --ids-file $D/prompts/eval_code.i32 -n 64 "${C[@]}" --ctx 4096 --vram-margin 5.5 --prefill-chunk 1024 --chunk-mmq "${R[@]}" --vram-grow 1
run c14-m8c2048  --ids-file $D/prompts/eval_code.i32 -n 64 "${C[@]}" --ctx 4096 --vram-margin 8 --prefill-chunk 2048 --chunk-mmq "${R[@]}" --vram-grow 1
echo "=== chain14 done $(date +%T)"
