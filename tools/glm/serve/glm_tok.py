"""GLM-5.3-Flash tokenizer on Strata's BPE (tools/strata_tokenizer.py): same byte-level BPE, but llama.cpp's
CHATGLM4 pre-tokenizer split (numbers in runs of up to 3 digits, no \\p{M} classes) instead of Qwen 3.5's."""
import json
import sys
from pathlib import Path

import regex

STRATA = Path.home() / "AI" / "Strata"
sys.path.insert(0, str(STRATA / "tools"))
import strata_tokenizer as ST  # noqa: E402

GLM4_PATTERN = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def load(tdir) -> "ST.Tokenizer":
    tdir = Path(tdir)
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    tok = ST.Tokenizer(tokens, merges, types, pre="glm4")
    tok._re = regex.compile(GLM4_PATTERN)
    return tok
