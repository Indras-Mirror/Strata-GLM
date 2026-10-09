#!/usr/bin/env bash
# 2026-10-09 chain 5 (~9 min): soft prune on GLM. List = REAP 25% from en+zh+code+tools calibration
# (cal_code/prose/multi/chat/python/json; romance + German held out). GLM selects on sigmoid(logit) + exp_probs_b
# (scores in [0,1], bias spread across experts p10-p90 0.03-0.13), so DS4's 0.5 penalty is ~a hard prune here:
# sweep 0.02/0.05/0.10 with --arena-adapt (pruned experts on the file tier, promoted on first read).
# Baselines (FINDINGS s7): no prune eval_code 3.676 / eval_chat 5.633; hard 25% (4-domain) decode 10.38 tok/s (sm0).
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
P=$D/prune-ezct-0.25.txt
C=(--backend cuda --experts gpu --slots auto --arena-gib 72 --arena-skip-resident --ctx 2048)
E=(-n 1 --vram-margin 5.5 --prefill-chunk 1024 --chunk-mmq --ppl)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "tier:|prune:|ppl:|prefill:|decode :|tier moves|error|fail" "$D/$name.log"; }
run c5-hard-chat  --ids-file $D/prompts/eval_chat.i32 "${C[@]}" "${E[@]}" --prune $P
run c5-hard-code  --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" --prune $P
run c5-rand-code  --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" --prune $D/prune-rand-0.25.txt
for pp in 0.02 0.05 0.10; do
  run c5-soft$pp-chat --ids-file $D/prompts/eval_chat.i32 "${C[@]}" "${E[@]}" --prune $P --prune-penalty $pp --arena-adapt
done
run c5-soft0.05-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" --prune $P --prune-penalty 0.05 --arena-adapt
# decode speed + decode-loop ppl (compare sm0: hard 25% 4-domain, 10.38 tok/s, ppl 11.159)
run c5-soft0.05-dec --ids-file $D/prompts/eval_code300.i32 -n 32 "${C[@]}" --vram-margin 1 --ppl --prune $P --prune-penalty 0.05 --arena-adapt
echo "=== chain5 done $(date +%T)"
