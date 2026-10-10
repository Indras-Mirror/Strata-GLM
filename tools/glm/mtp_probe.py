#!/usr/bin/env python3
"""List (and optionally extract) the GLM NextN / MTP draft-block tensors in a GGUF.

The draft block is GLM's NextN layer: tensors named `blk.45.*` in project-maya's GGUFs, or `mtp.*` in zai-org's
safetensors.  Our model file has neither (max blk 44), so the block must be loaded beside it.  This tool is the
first step of docs/glm/SPEC_DRAFT.md (step A): confirm the block's exact tensors, shapes and types, and dump them
to raw .bin + a manifest (the PolyStrata mtp-manifest.json shape) so the engine's GLM NextN forward can load them.

  mtp_probe.py MODEL.gguf                 # list the draft-block tensors
  mtp_probe.py MODEL.gguf --extract DIR   # write DIR/<name>.bin + DIR/manifest.json
"""
import hashlib
import json
import os
import struct
import sys

# ggml type id -> (name, bytes_per_block, elems_per_block)
TYPES = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32), 6: ("Q5_0", 22, 32),
         7: ("Q5_1", 24, 32), 8: ("Q8_0", 34, 32), 9: ("Q8_1", 36, 32), 10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256),
         12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 15: ("Q8_K", 292, 256),
         16: ("IQ2_XXS", 66, 256), 17: ("IQ2_XS", 74, 256), 18: ("IQ3_XXS", 98, 256), 19: ("IQ1_S", 50, 256),
         20: ("IQ4_NL", 18, 32), 21: ("IQ3_S", 110, 256), 22: ("IQ2_S", 82, 256), 23: ("IQ4_XS", 136, 256),
         24: ("I8", 1, 1), 25: ("I16", 2, 1), 26: ("I32", 4, 1), 27: ("I64", 8, 1), 28: ("F64", 8, 1),
         30: ("BF16", 2, 1)}

MATCH = ("blk.45.", "mtp.", "nextn", "draft")


def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF"
    struct.unpack("<I", f.read(4))                      # version
    n_t, = struct.unpack("<Q", f.read(8))
    n_k, = struct.unpack("<Q", f.read(8))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf8", "replace")

    def skip(t):
        if t in (0, 1, 7):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t == 8:
            rstr()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                if et == 8:
                    rstr()
                elif et in (0, 1, 7):
                    f.read(1)
                elif et in (2, 3):
                    f.read(2)
                elif et in (4, 5, 6):
                    f.read(4)
                elif et in (10, 11, 12):
                    f.read(8)
        elif t in (10, 11, 12):
            f.read(8)
        else:
            raise ValueError("kv type %d" % t)

    for _ in range(n_k):
        rstr()
        t, = struct.unpack("<I", f.read(4))
        skip(t)

    tensors = []
    for _ in range(n_t):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append({"name": name, "shape": dims, "type": tt, "offset": off})
    hdr_end = f.tell()
    data_off = (hdr_end + 31) // 32 * 32                  # gguf default alignment 32
    return f, tensors, data_off


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    extract = None
    if "--extract" in sys.argv:
        extract = sys.argv[sys.argv.index("--extract") + 1]
    f, tensors, data_off = read_gguf(path)
    hits = [t for t in tensors if any(m in t["name"] for m in MATCH)]
    if not hits:
        print("no draft-block tensors (blk.45.* / mtp.* / nextn) in %s" % path)
        print("max blk index:", max((int(t["name"].split(".")[1]) for t in tensors
                                     if t["name"].startswith("blk.") and t["name"].split(".")[1].isdigit()),
                                    default=-1))
        return 1
    manifest = []
    if extract:
        os.makedirs(extract, exist_ok=True)
    total = 0
    for t in hits:
        tn, bpb, epb = TYPES.get(t["type"], ("t%d" % t["type"], 1, 1))
        ne = 1
        for d in t["shape"]:
            ne *= d
        nbytes = ne // epb * bpb
        total += nbytes
        print("%-46s %-8s %-22s %9.2f MB" % (t["name"], tn, str(t["shape"]), nbytes / 1e6))
        if extract:
            f.seek(data_off + t["offset"])
            blob = f.read(nbytes)
            fn = os.path.join(extract, t["name"] + ".bin")
            open(fn, "wb").write(blob)
            manifest.append({"name": t["name"], "dtype": tn, "shape": t["shape"], "bytes": nbytes,
                             "file": os.path.relpath(fn, extract), "sha256": hashlib.sha256(blob).hexdigest()})
    print("--- %d tensors, %.2f GB total" % (len(hits), total / 1e9))
    if extract:
        json.dump(manifest, open(os.path.join(extract, "manifest.json"), "w"), indent=1)
        print("wrote %s/manifest.json" % extract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
