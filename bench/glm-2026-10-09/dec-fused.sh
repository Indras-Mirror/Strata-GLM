#!/usr/bin/env bash
# s28: fused predict (finish(il) computes predict(il+1)) A/B, native q6 dense, + q4_k speed
set -u
cd "$(dirname "$0")/../.."
D=bench/glm-2026-10-09
F=(--ctx 524288 --prefill-chunk 4096 --vram-margin 7.5 --vram-grow 1.5 -n 256 --pf-b 1.0 --arena-gib 64)
IDS=chat_code6k bash $D/oom-repro.sh dec-fused-q6 "${F[@]}" --dense-requant none
GLM_NO_FUSED_PREDICT=1 IDS=chat_code6k bash $D/oom-repro.sh dec-unfused-q6 "${F[@]}" --dense-requant none
IDS=chat_code6k bash $D/oom-repro.sh dec-fused-q4 "${F[@]}" --dense-requant q4_k
for n in dec-fused-q6 dec-unfused-q6 dec-fused-q4; do echo "== $n"; grep -hE "decode :|experts/token|decode ms/token|tier ms" $D/$n.log; done
