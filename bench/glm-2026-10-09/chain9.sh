#!/usr/bin/env bash
# 2026-10-09 chain 9 (~8 min): decode speed on the recommended soft config (s12): CPU pool is the long pole
# (79 ms vs PCIe 34 ms per token) -> sweep --pcie (share of misses DMAd to the GPU, default 0.25), + skip-miss 0.05.
# Reference: soft 0.05 + adapt + skip-file 0.15 = 9.82 tok/s (chain7 c7-sf0.15-dec).
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
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "tier ms|tier:|ppl:|prefill:|decode :|experts/token|tier moves|decode ms|error|fail" "$D/$name.log"; }
R=(--prune $P --prune-penalty 0.05 --arena-adapt --skip-file 0.15 --skip-file-prefill 0.15)
for pc in 0.35 0.45 0.55 0.65; do
  run c9-pcie$pc-dec "${X[@]}" "${C[@]}" "${R[@]}" --pcie $pc
done
run c9-sm0.05-dec "${X[@]}" "${C[@]}" "${R[@]}" --skip-miss 0.05
echo "=== chain9 done $(date +%T)"
