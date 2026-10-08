#!/usr/bin/env bash
# Runtime-LoRA check for Strata-GLM: the same real chat prompt with and without the abliteration adapter.
# Compares generated ids + first-token logits (dump) so the adapter's effect is visible.
set -euo pipefail
cd "$HOME/AI/Strata-GLM"
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
B="${GLM_BIN:-build-glm-gpu/glm_generate}"
D=bench/glm-2026-10-08
L=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
COMMON=(--backend cuda --experts gpu --slots auto --arena-gib 60 --ctx 2048 --vram-lru)
PROMPT=$D/prompts/chat.i32

echo "### A: no adapter ($(date '+%H:%M:%S'))"
bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" --ids-file "$PROMPT" -n 24 \
    --dump-logits "$D/logits-nolora.f32" "${COMMON[@]}" > "$D/gen-nolora.txt" 2> "$D/run-nolora.log"
tail -4 "$D/run-nolora.log"

echo "### B: --lora $L ($(date '+%H:%M:%S'))"
bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" --ids-file "$PROMPT" -n 24 --lora "$L" \
    --dump-logits "$D/logits-lora.f32" "${COMMON[@]}" > "$D/gen-lora.txt" 2> "$D/run-lora.log"
tail -4 "$D/run-lora.log"

echo "### no adapter:"; tr '\n' ' ' < "$D/gen-nolora.txt"; echo
echo "### adapter   :"; tr '\n' ' ' < "$D/gen-lora.txt"; echo
python3 - "$D/logits-nolora.f32" "$D/logits-lora.f32" <<'PY'
import sys, struct, math
a = open(sys.argv[1],'rb').read(); b = open(sys.argv[2],'rb').read()
va = struct.unpack("<%df" % (len(a)//4), a); vb = struct.unpack("<%df" % (len(b)//4), b)
n = min(len(va), len(vb))
d = [abs(va[i]-vb[i]) for i in range(n)]
print(f"logits[0]: n={n} max|d|={max(d):.5f} mean|d|={sum(d)/n:.6f} argmax nolora={va.index(max(va))} lora={vb.index(max(vb))}")
PY
