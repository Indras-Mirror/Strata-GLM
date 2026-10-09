#!/usr/bin/env bash
# 2026-10-09 chain 8 (~5 min): quality of skip-file at full size + prefill speed: 2000-token chunked evals with
# --skip-file-prefill 0.15 / 0.20 (soft 0.05 + adapt). Compare s10/s11: soft 0.05 chat 5.516 code 3.583 prefill ~100;
# base 5.547 / 3.668; hard 25% prefill 161 tok/s.
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
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "skipped:|tier:|ppl:|prefill:|decode :|experts/token|tier moves|decode ms|error|fail" "$D/$name.log"; }
S=(--prune $P --arena-adapt)
S=(--prune $P --arena-adapt --prune-penalty 0.05)
for t in 0.15 0.20; do
  run c8-sfp$t-chat --ids-file $D/prompts/eval_chat.i32 "${C[@]}" "${E[@]}" "${S[@]}" --skip-file-prefill $t
  run c8-sfp$t-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" "${S[@]}" --skip-file-prefill $t
done
echo "=== chain8 done $(date +%T)"
