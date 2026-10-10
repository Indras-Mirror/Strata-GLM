#!/usr/bin/env bash
# Is the greedy parity split (ctx 4096 vs 2560, id ~25) the CAPMAX change, or run-to-run noise?  ctx 4096 again.
set -u; cd "$(dirname "$0")/../.."
D=bench/glm-2026-10-09; LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
while pgrep -f "gate-ctx300k.s[h]" >/dev/null; do sleep 10; done
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 10; done
bash tools/ds4/memguard.sh 84 70 -- build-glm-gpu/glm_generate -m /media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
  --backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --slots 600 --pcie 1.0 --temp 0 --prefill-chunk 1024 \
  --ids-file $D/prompts/cross2300.i32 -n 64 --ctx 4096 > $D/ctx300k-par4096b.log 2>&1
a=$(grep -nE '^[0-9]+$' $D/ctx300k-par4096.log | head -64 | cut -d: -f2 | tr '\n' ' ')
b=$(grep -nE '^[0-9]+$' $D/ctx300k-par4096b.log | head -64 | cut -d: -f2 | tr '\n' ' ')
python3 -c "
a='$a'.split(); b='$b'.split()
n=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
print(f'determinism (4096 vs 4096 again): first {n} of {min(len(a),len(b))} ids agree')" | tee -a $D/ctx300k.summary
