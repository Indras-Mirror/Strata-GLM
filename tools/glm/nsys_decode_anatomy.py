#!/usr/bin/env python3
"""Decode anatomy from an nsys cuda_gpu_trace CSV (captured with --cuda-graph-trace=node, so graph nodes count).

usage: nsys_decode_anatomy.py <prefix>_cuda_gpu_trace.csv <n_tokens>

Prints per token: kernel busy (union over streams), copy busy (H2D / D2H unions), the overlap of the two, and the
kernel time by family - the numbers that say whether decode is GPU-compute-, PCIe- or host-bound.
"""
import csv
import re
import sys
from collections import defaultdict


def union(iv):
    iv.sort()
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


def merged(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def inter(a, b):
    i = j = tot = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


FAMILIES = [
    ("moe grouped/native", r"native_expert|grouped|expert_lora|lora"),
    ("mul_mat (dense, q)", r"mul_mat|mmvq|mmq|gemv|gemm|cublas|cutlass|ampere|sm80|sm89"),
    ("flash_attn", r"flash_attn"),
    ("kda / ssm", r"ssm|gated_delta|kda|delta|scan|conv"),
    ("indexer/top-k", r"lightning|top_k|argsort|topk"),
    ("norm/rope/act", r"norm|rope|silu|swiglu|sigmoid|gelu|softplus|exp"),
    ("quantize/convert", r"quantize|convert|cpy|dequant|to_fp|f32_to|q8_1"),
    ("elementwise/misc", r"."),
]


def main():
    path, ntok = sys.argv[1], int(sys.argv[2])
    rows = list(csv.DictReader(open(path)))
    k_iv, h2d_iv, d2h_iv = [], [], []
    fam = defaultdict(float)
    names = defaultdict(lambda: [0.0, 0])
    t0, t1 = None, None
    for r in rows:
        s = int(r["Start (ns)"])
        d = int(r["Duration (ns)"])
        n = r["Name"]
        t0 = s if t0 is None else min(t0, s)
        t1 = s + d if t1 is None else max(t1, s + d)
        if "[CUDA memcpy" in n or "[CUDA memset" in n:
            if "Host-to-Device" in n:
                h2d_iv.append((s, s + d))
            elif "Device-to-Host" in n:
                d2h_iv.append((s, s + d))
            continue
        k_iv.append((s, s + d))
        names[n][0] += d
        names[n][1] += 1
        for f, pat in FAMILIES:
            if re.search(pat, n, re.I):
                fam[f] += d
                break
    span = (t1 - t0) / 1e6
    km, hm = merged(k_iv), merged(h2d_iv)
    kb, hb, db = union(list(k_iv)), union(list(h2d_iv)), union(list(d2h_iv))
    pt = lambda ns: ns / 1e6 / ntok
    print(f"capture span {span:.1f} ms = {span / ntok:.2f} ms/token over {ntok} tokens")
    print(f"kernel busy {pt(kb):.2f} ms/token ({100 * kb / 1e6 / span:.0f}% of span)")
    print(f"H2D busy {pt(hb):.2f}  D2H busy {pt(db):.2f}  kernel&H2D overlap {pt(inter(km, hm)):.2f} ms/token")
    print(f"neither kernel nor H2D: {span / ntok - pt(union(list(k_iv) + list(h2d_iv))):.2f} ms/token")
    print("kernel time by family (ms/token, summed over streams):")
    for f, _ in FAMILIES:
        print(f"  {f:22s} {pt(fam[f]):7.2f}")
    print("top kernels (ms/token, calls/token):")
    for n, (d, c) in sorted(names.items(), key=lambda x: -x[1][0])[:22]:
        print(f"  {pt(d):6.2f} {c / ntok:7.1f}  {n[:110]}")


if __name__ == "__main__":
    main()
