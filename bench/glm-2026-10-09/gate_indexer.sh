#!/usr/bin/env bash
# CPU gate for the lightning indexer on the mini fixture (idx_top_k 8, kpool 4: dense-exact up to 11 tokens)
#  G1 11-token prompt: indexer == forced dense (--allow-long-ctx), every token still selects everything
#  G2 30-token prompt: indexer != dense (the selection restricts attention)
#  G3 30-token prompt: chunked (passes of 7: pools straddle passes, tail carried) == one-token loop
set -uo pipefail
cd "$HOME/AI/Strata-GLM"
B=${GLM_CPU_BIN:-build-glm/glm_generate_cpu}; F=/tmp/mini-glm.gguf; D=bench/glm-2026-10-09/gate
mkdir -p $D
[ -f $F ] || python3 tools/glm/make_mini_glm.py $F >/dev/null
ids() { python3 -c "print(','.join(str(3 + (i * 7) % 50) for i in range($1)))"; }
run() { local o=$1; shift; nice -n 10 "$B" --threads 2 -m $F --backend cpu --experts cpu --ctx 64 -n 2 --dump-logits $D/$o.f32 "$@" > $D/$o.log 2>&1 || { echo "$o FAILED"; tail -3 $D/$o.log; }; }
run g1-idx   --ids $(ids 11)
run g1-dense --ids $(ids 11) --allow-long-ctx
run g2-idx   --ids $(ids 30)
run g2-dense --ids $(ids 30) --allow-long-ctx
run g3-loop  --ids $(ids 30)
run g3-chunk --ids $(ids 30) --prefill-chunk 7
run g4-chunk5 --ids $(ids 30) --prefill-chunk 5
run g4-dense-chunk7 --ids $(ids 30) --prefill-chunk 7 --allow-long-ctx
python3 - <<'PY'
import numpy as np
D='bench/glm-2026-10-09/gate/'
def d(a,b):
    x=np.fromfile(D+a+'.f32',np.float32); y=np.fromfile(D+b+'.f32',np.float32); return np.abs(x-y).max(), x.argmax(), y.argmax()
g1=d('g1-idx','g1-dense'); g2=d('g2-idx','g2-dense'); g3=d('g3-loop','g3-chunk')
print('G1 idx vs dense @11  max|d| %.3g  %s'%(g1[0],'PASS' if g1[0]==0 else 'FAIL'))
print('G2 idx vs dense @30  max|d| %.3g  %s'%(g2[0],'PASS (differs)' if g2[0]>1e-3 else 'FAIL (no effect)'))
print('G3 loop vs chunk7    max|d| %.3g  %s'%(g3[0],'PASS' if g3[0]<1e-3 else 'FAIL'))
g4=d('g3-loop','g4-chunk5'); print('G4 loop vs chunk5    max|d| %.3g  %s'%(g4[0],'PASS' if g4[0]<1e-3 else 'FAIL'))
g5=d('g2-dense','g4-dense-chunk7'); print('G5 dense loop vs dense chunk7 max|d| %.3g  %s'%(g5[0],'PASS' if g5[0]<1e-3 else 'FAIL'))
PY
