#!/usr/bin/env python3
"""Bootstrap a glm_generate --census file from --saliency dumps ([i32 n_layer][i32 n_expert][f64 sum x L*E]
[i64 count x L*E]): the per-(layer, expert) routing counts, summed over the files.  The census then keeps itself
current from served decode.  usage: census_from_saliency.py OUT.census sal1.bin [sal2.bin ...]"""
import sys
import numpy as np

out, files = sys.argv[1], sys.argv[2:]
tot = None
for f in files:
    raw = open(f, "rb").read()
    L, E = np.frombuffer(raw[:8], "<i4")
    n = int(L) * int(E)
    cnt = np.frombuffer(raw[8 + 8 * n: 8 + 16 * n], "<i8").reshape(L, E)
    tot = cnt.astype(np.float64) if tot is None else tot + cnt
    print(f"{f}: {cnt.sum()} routes, {(cnt > 0).sum()} experts used")
# scale to ~decode-sized counts so served decode can move the ranking (a prompt's 2000 tokens x 8 per layer)
tot = np.round(tot / max(1.0, tot.sum() / (tot > 0).sum() / 4.0))
with open(out, "w") as fo:
    fo.write("# glm_generate census: layer expert decode-uses (bootstrapped from saliency: " + " ".join(files) + ")\n")
    for l in range(tot.shape[0]):
        for e in range(tot.shape[1]):
            if tot[l, e] > 0:
                fo.write(f"{l} {e} {tot[l, e]:.0f}\n")
print(f"wrote {out}: {(tot > 0).sum()} entries")
