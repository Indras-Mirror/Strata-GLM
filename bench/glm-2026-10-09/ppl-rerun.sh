#!/usr/bin/env bash
# s28: run-to-run reproducibility of the exact-math chat ppl (q5_k came out +6% vs q6 once, code only +1.1%)
set -u
cd "$(dirname "$0")/../.."
D=bench/glm-2026-10-09
for q in none q5_k; do
  IDS=eval_chat bash $D/oom-repro.sh X2-$q-chat --ctx 4096 --prefill-chunk 1024 --vram-margin 5.5 -n 1 --ppl \
    --skip-miss 0 --skip-file 0 --skip-file-prefill 0 --pf-b 1.0 --arena-gib 64 --dense-requant $q
  echo "rerun $q chat: $(grep -h 'ppl:' $D/X2-$q-chat.log)"
done
