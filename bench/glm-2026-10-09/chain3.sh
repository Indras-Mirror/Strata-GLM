#!/usr/bin/env bash
# 2026-10-09 chain 3 (<=10 min): broader calibration (+romance, python, json/tool-call) -> prune7 on the German-heavy
# chat eval; skip-miss sweep on the decode path (REAP 25%, fully resident; decode-loop ppl over 300 tokens)
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
C=(--backend cuda --experts gpu --slots auto --arena-gib 72 --arena-skip-resident --ctx 2048)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "tier:|prune:|ppl:|prefill:|decode :|experts/token|ms/token" "$D/$name.log"; }
nice -n 5 cmake --build build-glm-gpu --target glm_generate -j 8 2>&1 | grep -E "error|Linking"
for p in cal_romance cal_python cal_json; do
  run sal-$p --ids-file $D/prompts/$p.i32 -n 1 "${C[@]}" --vram-margin 5.5 --prefill-chunk 1024 --chunk-mmq --saliency $D/sal-$p.bin
done
python3 tools/glm/reap_prune.py $D/sal-cal_*.bin --fit-gb 92 -o $D/prune7-fit92.txt
run ev-p7fit92-eval_chat --ids-file $D/prompts/eval_chat.i32 -n 1 "${C[@]}" --vram-margin 5.5 --prefill-chunk 1024 --chunk-mmq --ppl --prune $D/prune7-fit92.txt
for sm in 0 0.05 0.10; do
  run sm$sm --ids-file $D/prompts/eval_code300.i32 -n 32 "${C[@]}" --vram-margin 1 --ppl --prune $D/prune-0.25.txt --skip-miss $sm
done
echo "=== chain3 done $(date +%T)"
