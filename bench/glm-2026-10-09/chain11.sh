#!/usr/bin/env bash
# 2026-10-09 chain 11 (~7 min; margin 6.5 (8 at chunk 2048); prestage half capped at 1.5 GiB, partial prestage): prefill speed - chunk_prestage now allowed with arena_adapt (soft prune). Same
# math as chain8 -> ppl must match exactly (c8-sfp0.15-chat 5.6651, -code 3.6047; hard code 3.5643).
# Prefill refs: soft+sfp0.15 130 (chat) / 140 (code) tok/s, hard 161 (code).
# NOTE: memguard.sh takes ds4-gpu.lock per run - never wrap this script in another flock on that lock.
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
P=$D/prune-ezct-0.25.txt
C=(--backend cuda --experts gpu --slots auto --arena-gib 72 --arena-skip-resident --ctx 2048)
E=(-n 1 --vram-margin 6.5 --prefill-chunk 1024 --chunk-mmq --ppl)
X=(--ids-file $D/prompts/eval_code300.i32 -n 64 --vram-margin 1 --ppl)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -E "skipped:|tier:|ppl:|prefill chunks|OOM|out of memory|prefill:|decode :|experts/token|tier moves|decode ms|error|fail" "$D/$name.log"; }
R=(--prune $P --prune-penalty 0.05 --arena-adapt --skip-file 0.15 --skip-file-prefill 0.15)
run c11-nopre-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" "${R[@]}"
run c11-pre-code  --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" "${R[@]}" --chunk-prestage
run c11-pre-chat  --ids-file $D/prompts/eval_chat.i32 "${C[@]}" "${E[@]}" "${R[@]}" --chunk-prestage
run c11-hardpre-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" "${E[@]}" --prune $P --chunk-prestage
run c11-pre2048-code --ids-file $D/prompts/eval_code.i32 "${C[@]}" -n 1 --vram-margin 8 --prefill-chunk 2048 --chunk-mmq --ppl "${R[@]}" --chunk-prestage
echo "=== chain11 done $(date +%T)"
