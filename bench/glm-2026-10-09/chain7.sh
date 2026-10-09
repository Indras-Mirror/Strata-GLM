#!/usr/bin/env bash
# 2026-10-09 chain 7 (~8 min): close the soft-prune decode gap to hard 25% (9.52 tok/s): skip-file sweep, penalty 0.10,
# skip-file + skip-miss. Same decode setup as chain6 (300-token decode-loop ppl, 64 tokens timed, margin 1).
# Reference (s11): soft 0.05 + adapt + skip-file 0.10 = 8.46 tok/s, ppl 11.554.
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
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "tier:|ppl:|prefill:|decode :|experts/token|tier moves|decode ms|error|fail" "$D/$name.log"; }
S=(--prune $P --arena-adapt)
run c7-sf0.15-dec       "${X[@]}" "${C[@]}" "${S[@]}" --prune-penalty 0.05 --skip-file 0.15
run c7-sf0.20-dec       "${X[@]}" "${C[@]}" "${S[@]}" --prune-penalty 0.05 --skip-file 0.20
run c7-p0.10sf0.10-dec  "${X[@]}" "${C[@]}" "${S[@]}" --prune-penalty 0.10 --skip-file 0.10
run c7-sf0.10sm0.05-dec "${X[@]}" "${C[@]}" "${S[@]}" --prune-penalty 0.05 --skip-file 0.10 --skip-miss 0.05
run c7-hardsm0.05-dec   "${X[@]}" "${C[@]}" --prune $P --skip-miss 0.05
echo "=== chain7 done $(date +%T)"
