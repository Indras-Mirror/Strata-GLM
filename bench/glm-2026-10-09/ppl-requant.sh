#!/usr/bin/env bash
# s28: the dense requant's quality with EXACT math (no skips): gate 4's X-* arms, which were cut to free the GPU.
# Native q6_k (none) vs --dense-requant q5_k / q4_k, on eval_chat and eval_code (4K tokens each).
set -u
cd "$(dirname "$0")/../.."
D=bench/glm-2026-10-09
S=$D/ppl-requant.summary
: > $S
for set in chat code; do
  for q in none q5_k q4_k; do
    rq=(--dense-requant $q)   # "none" parses to -1 = native q6_k (overrides the script's q5_k)
    IDS=eval_$set bash $D/oom-repro.sh X-$q-$set --ctx 4096 --prefill-chunk 1024 --vram-margin 5.5 -n 1 --ppl \
      --skip-miss 0 --skip-file 0 --skip-file-prefill 0 --pf-b 1.0 --arena-gib 64 "${rq[@]}"
    echo "$set $q: $(grep -h 'ppl' $D/X-$q-$set.log | tail -1)" | tee -a $S
  done
done
