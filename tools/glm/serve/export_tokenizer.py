#!/usr/bin/env python3
"""Export GLM-5.3-Flash's tokenizer from the GGUF into the directory layout Strata's server reads
(vocab.json, merges.txt, token_type.json, chat_template.jinja).

  export_tokenizer.py MODEL.gguf OUT_DIR
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "..", "third_party", "llama.cpp", "gguf-py"))
from gguf import GGUFReader  # noqa: E402


def strings(r, key):
    f = r.fields[key]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def main():
    model, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    r = GGUFReader(model)
    tokens = strings(r, "tokenizer.ggml.tokens")
    merges = strings(r, "tokenizer.ggml.merges")
    tt = r.fields["tokenizer.ggml.token_type"]
    types = [int(tt.parts[i][0]) for i in tt.data]
    tpl = strings(r, "tokenizer.chat_template")[0]
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({t: i for i, t in enumerate(tokens)}, f, ensure_ascii=False)
    with open(os.path.join(out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(merges))
    with open(os.path.join(out, "token_type.json"), "w") as f:
        json.dump(types, f)
    with open(os.path.join(out, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(tpl)
    print(f"{len(tokens)} tokens, {len(merges)} merges, template {len(tpl)} chars -> {out}")


if __name__ == "__main__":
    main()
