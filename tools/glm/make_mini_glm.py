#!/usr/bin/env python3
"""tools/glm/make_mini_glm.py - write a tiny `glm5-next` GGUF for GlmDense smoke/gate tests.

The real artifact is 105.8 GiB; this is a 4-layer model with the same architecture (mHC streams, KDA linear
attention, nope-MLA, dense FFN + MoE) but tiny dims, so `glm_generate_cpu` exercises the whole path (loader,
mHC, KDA recurrence + conv state, dense MLA, dense FFN, router + shared expert + Ds4MoeTier, head) with no
GPU and no real model.

Layer plan (4 blocks, leading_dense_block_count = 1, head_count_kv = [0, NH, 0, NH]):
  blk.0  KDA  + dense FFN      (dense lead)
  blk.1  MLA  + MoE
  blk.2  KDA  + MoE
  blk.3  MLA  + MoE  (last: also drives next_pos / head)
so all four (attention kind x FFN kind) combinations are covered.

Tensor shapes are given in GGUF order (ne0 first), as glm_dense.cpp / the fork's glm5-next.cpp read them; gguf-py
stores dims reversed, so numpy arrays are built with the shape reversed.

    python3 tools/glm/make_mini_glm.py /tmp/mini-glm.gguf
    ./build-glm/glm_generate_cpu -m /tmp/mini-glm.gguf --backend cpu --experts cpu --ids 1,2,3 -n 4
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))

from _paths import add_gguf_py  # noqa: E402

add_gguf_py()
from gguf import GGMLQuantizationType as Q, GGUFWriter, quants  # noqa: E402

# ---- miniature geometry (same structure as the real model, tiny widths)
D   = 256      # n_embd
L   = 4        # block_count
NH  = 4        # attention.head_count (also the KDA head count: d_inner = kda_head_dim * NH)
SK  = 32       # kda.head_dim
DI  = SK * NH  # KDA inner (q/k/v/o width) = 128
DC  = 4        # ssm.conv_kernel
QL  = 64       # attention.q_lora_rank
KVL = 64       # attention.kv_lora_rank
KM  = 32       # attention.key_length_mla (qk_rope = 0 -> all nope)
VM  = 32       # attention.value_length_mla
NFF = 512      # feed_forward_length (dense FFN width)
NFFE = 256     # expert_feed_forward_length
NE  = 8        # expert_count
TOPK = 2       # expert_used_count
NSHEXP = 1     # shared experts
NDENSE = 1     # leading_dense_block_count
V   = 64       # vocab_size
HC  = 4        # hyper_connection.count
HC_MIX = (2 + HC) * HC  # 24
IDX_TOPK, IDX_KPOOL = 8, 4
IDXH, IDXK = 2, 32  # indexer heads x key_length (the fork needs these + the indexer tensors)
CTX = 4096
HEAD_COUNT_KV = [0, 1, 0, 1]  # 0 = KDA layer (from head_count_kv==0); MLA layers carry the shared latent, so kv heads = 1

BLK = {"Q2_K": 256}  # elements per quant block


def tensors():
    """(name, gguf ne, ggml type) for every tensor the miniature model holds."""
    t = [
        ("token_embd.weight", [D, V], "F16"),
        ("output_norm.weight", [D], "F32"),
        ("output.weight", [D, V], "Q8_0"),
    ]
    for i in range(L):
        p = "blk.%d." % i
        t += [
            (p + "attn_norm.weight", [D], "F32"),
            (p + "ffn_norm.weight", [D], "F32"),
            (p + "hc_attn_fn.weight", [HC * D, HC_MIX], "F16"),
            (p + "hc_attn_base.weight", [HC_MIX], "F32"),
            (p + "hc_attn_scale.weight", [3], "F32"),
            (p + "hc_ffn_fn.weight", [HC * D, HC_MIX], "F16"),
            (p + "hc_ffn_base.weight", [HC_MIX], "F32"),
            (p + "hc_ffn_scale.weight", [3], "F32"),
        ]
        kda = HEAD_COUNT_KV[i] == 0
        if kda:
            t += [
                (p + "attn_q.weight", [D, DI], "Q8_0"),
                (p + "attn_k.weight", [D, DI], "Q8_0"),
                (p + "attn_v.weight", [D, DI], "Q8_0"),
                (p + "attn_output.weight", [DI, D], "Q8_0"),
                (p + "ssm_conv1d_q.weight", [DC, 1, DI], "F32"),
                (p + "ssm_conv1d_k.weight", [DC, 1, DI], "F32"),
                (p + "ssm_conv1d_v.weight", [DC, 1, DI], "F32"),
                (p + "ssm_f_a.weight", [D, SK], "F16"),
                (p + "ssm_f_b.weight", [SK, DI], "F16"),
                (p + "ssm_beta.weight", [D, NH], "F16"),
                (p + "ssm_a", [NH], "F32"),
                (p + "ssm_dt.bias", [DI], "F32"),
                (p + "ssm_g_a.weight", [D, SK], "F16"),
                (p + "ssm_g_b.weight", [SK, DI], "F16"),
                (p + "ssm_norm.weight", [SK], "F32"),
            ]
        else:
            t += [
                (p + "attn_q_a.weight", [D, QL], "Q8_0"),
                (p + "attn_q_a_norm.weight", [QL], "F32"),
                (p + "attn_q_b.weight", [QL, NH * KM], "Q8_0"),
                (p + "attn_k_b.weight", [KM, KVL, NH], "F16"),
                (p + "attn_kv_a_mqa.weight", [D, KVL], "Q8_0"),
                (p + "attn_kv_a_norm.weight", [KVL], "F32"),
                (p + "attn_v_b.weight", [KVL, VM, NH], "F16"),
                (p + "attn_output.weight", [NH * VM, D], "Q8_0"),
                # the lightning indexer: glm_dense skips anything named `indexer` (dense MLA is exact up to
                # dense_attn_ctx), but the reference fork requires these to load the model - so the fixture
                # carries them and can serve the oracle comparison too.
                (p + "indexer.k_norm.weight", [IDXK], "F32"),
                (p + "indexer.k_norm.bias", [IDXK], "F32"),
                (p + "indexer.proj.weight", [D, IDXH], "F16"),
                (p + "indexer.attn_k.weight", [D, IDXK], "F16"),
                (p + "indexer.attn_q_b.weight", [QL, IDXH * IDXK], "F16"),
                (p + "indexer_compressor_gate.weight", [D, IDXK], "F16"),
                (p + "indexer_compressor_ape.weight", [IDXK, IDX_KPOOL], "F16"),
            ]
        if i < NDENSE:
            t += [
                (p + "ffn_gate.weight", [D, NFF], "Q8_0"),
                (p + "ffn_up.weight", [D, NFF], "Q8_0"),
                (p + "ffn_down.weight", [NFF, D], "Q8_0"),
            ]
        else:
            t += [
                (p + "ffn_gate_inp.weight", [D, NE], "F32"),
                (p + "exp_probs_b.bias", [NE], "F32"),
                (p + "ffn_gate_exps.weight", [D, NFFE, NE], "Q2_K"),
                (p + "ffn_up_exps.weight", [D, NFFE, NE], "Q2_K"),
                (p + "ffn_down_exps.weight", [NFFE, D, NE], "Q2_K"),
                (p + "ffn_gate_shexp.weight", [D, NFFE * NSHEXP], "Q8_0"),
                (p + "ffn_up_shexp.weight", [D, NFFE * NSHEXP], "Q8_0"),
                (p + "ffn_down_shexp.weight", [NFFE * NSHEXP, D], "Q8_0"),
            ]
    return t


def _quant_array(f32: np.ndarray, ty: Q) -> np.ndarray:
    """Quantize a float32 array to a K-quant, or fall back to zero bytes if gguf-py cannot (keeps the file loadable)."""
    try:
        return quants.quantize(f32, ty)
    except Exception:
        e = BLK[ty.name]
        np_shape = f32.shape
        byte_shape = np_shape[:-1] + (np_shape[-1] // e * (84 if ty == Q.Q2_K else 0),)
        return np.zeros(int(np.prod(byte_shape)), dtype=np.uint8).reshape(byte_shape)


def write_model(path: Path) -> None:
    w = GGUFWriter(str(path), "glm5-next")
    u32 = {
        "block_count": L, "embedding_length": D, "attention.head_count": NH,
        "feed_forward_length": NFF, "vocab_size": V, "context_length": CTX,
        "hyper_connection.count": HC, "hyper_connection.sinkhorn_iterations": 20,
        "kda.head_dim": SK, "ssm.conv_kernel": DC,
        "attention.q_lora_rank": QL, "attention.kv_lora_rank": KVL,
        "attention.key_length_mla": KM, "attention.value_length_mla": VM,
        "attention.indexer.top_k": IDX_TOPK, "attention.indexer.kpool": IDX_KPOOL,
        "expert_count": NE, "expert_used_count": TOPK, "expert_feed_forward_length": NFFE,
        "leading_dense_block_count": NDENSE, "expert_gating_func": 2,
        "expert_shared_count": NSHEXP, "rope.dimension_count": 0,
        "attention.indexer.head_count": IDXH, "attention.indexer.key_length": IDXK,
    }
    for k, v in u32.items():
        w.add_uint32("glm5-next." + k, v)
    for k, v in {
        "hyper_connection.epsilon": 1e-6, "attention.layer_norm_rms_epsilon": 1e-5,
        "kda.gate_lower_bound": -5.0, "expert_weights_scale": 2.5,
    }.items():
        w.add_float32("glm5-next." + k, v)
    w.add_bool("glm5-next.expert_weights_norm", True)
    w.add_array("glm5-next.attention.head_count_kv", [int(x) for x in HEAD_COUNT_KV])
    w.add_array("glm5-next.swiglu_clamp_exp", [10.0] * L)
    w.add_array("glm5-next.swiglu_clamp_shexp", [10.0] * L)
    # a minimal SPM-style tokenizer (dummy tokens) so llama.cpp - and so the reference fork, the oracle - can
    # load the fixture; our own engine ignores it (it consumes raw ids via --ids/--ids-file)
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("glm4")
    w.add_token_list(["<unk>"] + ["t%d" % i for i in range(1, V)])
    w.add_token_scores([0.0] * V)
    w.add_token_types([1] * V)  # TokenType.NORMAL
    w.add_uint32("tokenizer.ggml.bos_token_id", 1)
    w.add_uint32("tokenizer.ggml.eos_token_id", 2)
    w.add_uint32("tokenizer.ggml.unknown_token_id", 0)

    for name, ne, ty in tensors():
        # a per-tensor seed (name-keyed) keeps every tensor's values stable as the fixture grows, so gate
        # records stay comparable across edits
        rng = np.random.default_rng(11 + zlib.crc32(name.encode()))
        n = int(np.prod(ne))
        np_shape = tuple(reversed(ne))          # gguf-py writes dims reversed
        if ty == "F16":
            w.add_tensor(name, rng.standard_normal(n).astype(np.float16).reshape(np_shape))
        elif ty == "F32":
            w.add_tensor(name, rng.standard_normal(n).astype(np.float32).reshape(np_shape))
        elif ty == "Q8_0":
            assert ne[0] % 32 == 0, (name, ne)
            arr = quants.quantize(rng.standard_normal(n).astype(np.float32).reshape(np_shape), Q.Q8_0)
            w.add_tensor(name, arr, raw_dtype=Q.Q8_0)
        else:
            e = BLK[ty]
            assert ne[0] % e == 0, (name, ne)
            f32 = rng.standard_normal(n).astype(np.float32).reshape(np_shape)
            q = _quant_array(f32, Q[ty])
            byte_shape = np_shape[:-1] + (ne[0] // e * (66 if ty == "IQ2_XXS" else 84),)
            w.add_tensor(name, q.reshape(byte_shape), raw_shape=byte_shape, raw_dtype=Q[ty])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini-glm.gguf")
    write_model(out)
    kinds = ["KDA" if k == 0 else "MLA" for k in HEAD_COUNT_KV]
    print("wrote", out, out.stat().st_size, "bytes; layers:",
          ", ".join("blk.%d %s+%s" % (i, kinds[i], "dense" if i < NDENSE else "MoE") for i in range(L)))
