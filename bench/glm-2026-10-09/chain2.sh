#!/usr/bin/env bash
# 2026-10-09 chain 2: held-out ppl + speed: base / REAP fit92 (14.2%) / REAP 25% / random 14.2%; loop-vs-chunk ppl gate
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
C=(--backend cuda --experts gpu --slots auto --vram-margin 3.5 --arena-gib 72 --arena-skip-resident --ctx 2048)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "slots auto|tier:|prune:|ppl:|prefill|decode :|experts/token|tier moves|ms/token" "$D/$name.log"; }
P=bench/glm-2026-10-08/prompts/neutral.i32
run ppl-loop-a  --ids-file $P -n 2 "${C[@]}" --ppl
run ppl-loop-b  --ids-file $P -n 2 "${C[@]}" --ppl
run ppl-chunk   --ids-file $P -n 2 "${C[@]}" --ppl --prefill-chunk 128 --chunk-mmq
run ppl-chunk-nommq --ids-file $P -n 2 "${C[@]}" --ppl --prefill-chunk 128
for cfg in base fit92 0.25 rand41; do
  X=(); [ $cfg != base ] && X=(--prune $D/prune-$cfg.txt)
  for p in eval_code eval_prose eval_chat; do
    run ev-$cfg-$p --ids-file $D/prompts/$p.i32 -n 32 "${C[@]}" --prefill-chunk 1024 --chunk-mmq --ppl "${X[@]}"
  done
done
echo "=== chain2 done $(date +%T)"
