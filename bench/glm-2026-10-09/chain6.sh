#!/usr/bin/env bash
# 2026-10-09 chain 6 (~9 min): (1) re-take the no-prune base after the allocator fix (s7's base ran with the bug);
# (2) win back soft-prune decode speed: arena admission gate (--arena-admit, heat half-life 64 tokens) and
# file-tier-only skipping (--skip-file 0.10), against hard ezct 25% on the same list. Decode runs: 300-token
# decode-loop prompt (= the ppl, and 300 steps of adapt warm-up), then 64 decoded tokens timed.
# Reference (s10): soft 0.05 + adapt = 7.57 tok/s, ppl 11.764, 282 file reads, 211 swaps over -n 32.
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
run c6-base-chat --ids-file $D/prompts/eval_chat.i32 "${C[@]}" "${E[@]}"
run c6-base-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}"
run c6-hard-dec  "${X[@]}" "${C[@]}" --prune $P
run c6-admit-dec "${X[@]}" "${C[@]}" --prune $P --prune-penalty 0.05 --arena-adapt --arena-admit 64
run c6-skipf-dec "${X[@]}" "${C[@]}" --prune $P --prune-penalty 0.05 --arena-adapt --skip-file 0.10
run c6-both-dec  "${X[@]}" "${C[@]}" --prune $P --prune-penalty 0.05 --arena-adapt --arena-admit 64 --skip-file 0.10
echo "=== chain6 done $(date +%T)"
