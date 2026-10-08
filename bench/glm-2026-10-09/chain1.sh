#!/usr/bin/env bash
# 2026-10-09 chain 1: chunked-prefill correctness, then calibration (REAP saliency) + speed with an exclusive 72 GiB arena
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B=build-glm-gpu/glm_generate
D=bench/glm-2026-10-09
C=(--backend cuda --experts gpu --slots auto --vram-margin 5 --arena-gib 72 --arena-skip-resident --ctx 2048)
run() { local name=$1; shift; echo "### $name $(date +%T)"; bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "$@" > "$D/$name.log" 2>&1; echo "exit $? $(date +%T)"; grep -vE '^[0-9]+$' "$D/$name.log" | tail -12; }
P=bench/glm-2026-10-08/prompts/neutral.i32
run gate-loop   --ids-file $P -n 4 "${C[@]}" --dump-logits $D/gate-loop.f32
run gate-chunk  --ids-file $P -n 4 "${C[@]}" --prefill-chunk 48 --dump-logits $D/gate-chunk.f32
python3 - <<'PY'
import numpy as np
a=np.fromfile('bench/glm-2026-10-09/gate-loop.f32',dtype=np.float32); b=np.fromfile('bench/glm-2026-10-09/gate-chunk.f32',dtype=np.float32)
print('GATE logits max|d| %.4f mean|d| %.4f argmax %d %d'%(np.abs(a-b).max(),np.abs(a-b).mean(),a.argmax(),b.argmax()))
PY
for p in cal_code cal_prose cal_multi cal_chat; do
  run sal-$p --ids-file $D/prompts/$p.i32 -n 16 "${C[@]}" --prefill-chunk 1024 --chunk-mmq --saliency $D/sal-$p.bin --ppl
done
echo "=== chain1 done $(date +%T)"
