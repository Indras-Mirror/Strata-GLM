#!/usr/bin/env bash
# 2026-10-09 chain 12 (~4 min): is chunked prefill ppl deterministic? chain8 c8-sfp0.15-code config (3.6047) x3, unchanged.
# NOTE: memguard.sh takes ds4-gpu.lock per run - never wrap this script in another flock on that lock.
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
P=$D/prune-ezct-0.25.txt
C=(--backend cuda --experts gpu --slots auto --arena-gib 72 --arena-skip-resident --ctx 2048)
E=(-n 1 --vram-margin 5.5 --prefill-chunk 1024 --chunk-mmq --ppl)
X=(--ids-file $D/prompts/eval_code300.i32 -n 64 --vram-margin 1 --ppl)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "skipped:|tier:|ppl:|prefill chunks|OOM|out of memory|prefill:|decode :|experts/token|tier moves|decode ms|error|fail" "$D/$name.log"; }
R=(--prune $P --prune-penalty 0.05 --arena-adapt --skip-file-prefill 0.15)
for i in 1 2 3; do run c12-rep$i-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" "${R[@]}"; done
echo "=== chain12 done $(date +%T)"
