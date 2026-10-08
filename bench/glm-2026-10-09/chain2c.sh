#!/usr/bin/env bash
# 2026-10-09 chain 2b (trimmed to ~8 min, GPU shared with strata-ds4-gpu): chunk ppl gate + prune sweep
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
while pgrep -f "build-glm-gpu/glm_generate" >/dev/null; do sleep 2; done
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
C=(--backend cuda --experts gpu --slots auto --vram-margin 5.5 --arena-gib 72 --arena-skip-resident --ctx 2048)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "tier:|prune:|ppl:|prefill:|decode :|experts/token|file reads" "$D/$name.log"; }
for cfg in base fit92 rand41 0.25; do
  X=(); [ $cfg != base ] && X=(--prune $D/prune-$cfg.txt)
  run ev-$cfg-eval_code --ids-file $D/prompts/eval_code.i32 -n 32 "${C[@]}" --prefill-chunk 1024 --chunk-mmq --ppl "${X[@]}"
done
for cfg in base fit92; do
  X=(); [ $cfg != base ] && X=(--prune $D/prune-$cfg.txt)
  run ev-$cfg-eval_chat --ids-file $D/prompts/eval_chat.i32 -n 32 "${C[@]}" --prefill-chunk 1024 --chunk-mmq --ppl "${X[@]}"
done
echo "=== chain2c done $(date +%T)"
