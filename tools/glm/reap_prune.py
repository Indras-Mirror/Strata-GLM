#!/usr/bin/env python3
"""REAP-style expert pruning list for Strata-GLM (glm_generate --prune).

Reads one or more saliency files written by `glm_generate --prefill-chunk N --saliency FILE` (layout: i32 n_layer,
i32 n_expert, f64 sum[L*E] of routing-weight * ||expert output||_2, i64 count[L*E]) and writes "layer expert" lines
for the experts to mask out of routing.

REAP score (Lasby et al. 2025, "REAP: Router-weighted Expert Activation Pruning"): S_j = mean over the tokens routed
to j of g_j(x) * ||f_j(x)||.  An expert never routed on the calibration text scores 0 and goes first.

Budget: --frac F prunes the F lowest-scoring experts in every routed layer, or --fit-gb B picks the smallest uniform
fraction whose kept experts fit B GB (the arena + exclusive VRAM slots), using GLM-5.3-Flash 3.0-bit blob sizes
(layers 3-5 Q4_K, 6-8 Q3_K, 9-44 Q2_K; gate+up+down 4096x2048x3 weights).

  reap_prune.py sal1.bin [sal2.bin ...] (--frac 0.15 | --fit-gb 93) [--keep-layers 3-8] -o prune.txt
"""
import argparse
import numpy as np

W = 3 * 4096 * 2048
BYTES_PER_256 = {"q4_k": 144, "q3_k": 110, "q2_k": 84}


def blob_bytes(layer):
    t = "q4_k" if layer <= 5 else "q3_k" if layer <= 8 else "q2_k"
    return W // 256 * BYTES_PER_256[t]


def load(paths):
    tot = cnt = None
    for p in paths:
        with open(p, "rb") as f:
            L, E = np.frombuffer(f.read(8), dtype=np.int32)
            s = np.frombuffer(f.read(8 * L * E), dtype=np.float64).reshape(L, E)
            c = np.frombuffer(f.read(8 * L * E), dtype=np.int64).reshape(L, E)
        tot = s.copy() if tot is None else tot + s
        cnt = c.copy() if cnt is None else cnt + c
    return tot, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sal", nargs="+")
    ap.add_argument("--frac", type=float)
    ap.add_argument("--fit-gb", type=float)
    ap.add_argument("--keep-layers", default="", help="e.g. 3-8: never prune these layers")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    s, c = load(a.sal)
    L, E = s.shape
    routed = [l for l in range(L) if c[l].sum() > 0]
    keep = set()
    if a.keep_layers:
        lo, hi = map(int, a.keep_layers.split("-"))
        keep = set(range(lo, hi + 1))
    score = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    order = {l: np.argsort(score[l], kind="stable") for l in routed}

    def kept_bytes(frac):
        n = int(round(frac * E))
        return sum(blob_bytes(l) * (E - (0 if l in keep else n)) for l in routed)

    if a.fit_gb is not None:
        frac = 0.0
        while frac < 0.9 and kept_bytes(frac) > a.fit_gb * 1e9:
            frac += 1.0 / E
    else:
        frac = a.frac
    n = int(round(frac * E))
    lines = [f"{l} {int(e)}" for l in routed if l not in keep for e in order[l][:n]]
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    tok = c[routed].sum() // 8
    never = int((c[routed] == 0).sum())
    lost = []
    for l in routed:
        if l in keep:
            continue
        pr = order[l][:n]
        lost.append(c[l, pr].sum() / max(1, c[l].sum()))
    print(f"{len(routed)} routed layers, {E} experts, calibration {c[routed].sum() // (8 * len(routed))} tokens, {never} (layer,expert) never routed")
    print(f"prune {n}/layer ({100 * n / E:.1f}%), {len(lines)} pairs; kept experts {kept_bytes(frac) / 1e9:.1f} GB "
          f"(all: {kept_bytes(0) / 1e9:.1f} GB)")
    print(f"calibration routes landing on pruned experts: mean {100 * np.mean(lost):.2f}% per layer "
          f"(max {100 * np.max(lost):.2f}%)")


if __name__ == "__main__":
    main()
