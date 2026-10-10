#!/usr/bin/env bash
# The real-300K serving config (s25d).  CAPMAX now rounds ctx to 256, not to a power of two (--ctx 307200 used to
# allocate the 512K MLA cache).  Three steps, one GPU lock:
#  1. parity: ctx 4096 vs ctx 2560 (a non-pow2 last bucket), fixed slots, no skip, greedy, the 2300-token prompt: its
#     first generated token (the dumped logits) is past idx_top_k 2048 and in the 2560 bucket -> logits must match
#  2. serve300k: the strata-glm-unc-ablated-300k.json args, margin 5.5, chunk-mmq; VRAM peak sampled at 0.2 s
#  3. serve300k-lazy: the same + --arena-lazy (wall time to first output)
set -u
cd "$(dirname "$0")/../.."
B=$PWD/build-glm-gpu/glm_generate
M=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
A=/media/mal/NVME1TB/Models/GLM-5.3-Flash-RCO/lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf
D=bench/glm-2026-10-09
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
S="$D/ctx300k.summary"
: > "$S"
while ! flock -n "$LOCK" true 2>/dev/null; do sleep 30; done

PAR=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --slots 600 --pcie 1.0 --temp 0
     --prefill-chunk 1024 --ids-file "$D/prompts/cross2300.i32" -n 64)
for ctx in 4096 2560; do
  echo "--- parity ctx $ctx $(date +%T) ---" | tee -a "$S"
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${PAR[@]}" --ctx $ctx --dump-logits "$D/ctx300k-par$ctx.f32" > "$D/ctx300k-par$ctx.log" 2>&1
  echo "exit $?" | tee -a "$S"
  grep -hE "^decode :|out of memory|error" "$D/ctx300k-par$ctx.log" | tee -a "$S"
done
python3 - "$D" <<'EOF' | tee -a "$S"
import numpy as np, sys
d = sys.argv[1]
a = np.fromfile(f"{d}/ctx300k-par4096.f32", np.float32); b = np.fromfile(f"{d}/ctx300k-par2560.f32", np.float32)
if a.size != b.size or a.size == 0: print(f"parity: logit dumps differ in size {a.size} vs {b.size}"); sys.exit()
V = 154880; a = a.reshape(-1, V); b = b.reshape(-1, V)
same = (a.argmax(1) == b.argmax(1)).mean()
print(f"parity: {a.shape[0]} rows, argmax agree {same*100:.2f}%, max |dlogit| {np.abs(a-b).max():.4g}, mean {np.abs(a-b).mean():.3g}")
EOF

SERVE=(--backend cuda --experts gpu --arena-gib 72 --arena-skip-resident --ctx 307200 --prefill-chunk 1024 --chunk-mmq
       --vram-margin 5.5 --prune "$D/prune-ezct-0.25.txt" --prune-penalty 0.05 --arena-adapt
       --skip-miss 0.15 --skip-file 0.15 --skip-file-prefill 0.15 --renorm-skip --pcie 1.0
       --stop 154820,154827,154829 --lora "$A" --lora-exps --ids-file "$D/prompts/eval_code.i32" -n 128)
for arm in serve300k serve300k-lazy; do
  EXTRA=(); [ "$arm" = serve300k-lazy ] && EXTRA=(--arena-lazy)
  echo "--- $arm $(date +%T) ---" | tee -a "$S"
  : > "$D/ctx300k-$arm.vram"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$D/ctx300k-$arm.vram"; sleep 0.2; done ) &
  smi=$!
  t0=$(date +%s)
  bash tools/ds4/memguard.sh 84 70 -- "$B" -m "$M" "${SERVE[@]}" "${EXTRA[@]}" > "$D/ctx300k-$arm.log" 2>&1
  rc=$?
  kill $smi 2>/dev/null
  grep -hE "slots auto|^tier:|^prefill:|^decode :|out of memory|error" "$D/ctx300k-$arm.log" | tee -a "$S"
  echo "exit $rc, wall $(( $(date +%s) - t0 )) s, VRAM peak $(sort -n "$D/ctx300k-$arm.vram" | tail -1) MiB of 24564" | tee -a "$S"
done
echo "=== ctx300k done $(date +%T) ===" | tee -a "$S"
