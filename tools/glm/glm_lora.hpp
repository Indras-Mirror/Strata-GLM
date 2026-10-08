// tools/glm/glm_lora.hpp - the run-time rank-1 abliteration adapter (a GGUF LoRA) for GLM-5.3-Flash.
//
// The default adapter (lora/gcsa-abliterix/GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf) is rank 1, F16, with 135 A/B
// pairs: routed-expert ffn_{gate,up,down}_exps (layers 3-28, all 288 experts), shared ffn_down_shexp (18-44) and
// attn_output (15-44).  Each pair is one linear map y = W x, so a merge at run time is  y += b * (a . x)
// (quantized expert blobs are never touched).  This class just PARSES the adapter into host float vectors; the
// callers (GlmDense's graph for attn_output / down_shexp, Ds4MoeTier for the routed exps) apply them.
//
// A/B layout in the GGUF (ne0 first):
//   *_exps    lora_a [in, 1, n_expert]   lora_b [1, out, n_expert]
//   solo      lora_a [in, 1]             lora_b [1, out]
#pragma once

#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

/// One rank-1 delta: y(out) += b * dot(a, x(in)).
struct Lora1 {
    std::vector<float> a;   // [in]
    std::vector<float> b;   // [out]
};

class LoraAdapter {
public:
    enum Mod : int { GATE = 0, UP = 1, DOWN = 2, SHEXP_DOWN = 3, ATTN_OUT = 4, NMOD = 5 };

    /// Open + parse the adapter.  Returns false with `err` set on any malformed pair (never partial-loads).
    bool load(const std::string & path, std::string & err) {
        try {
            model_ = std::make_unique<GgufModel>(GgufModel::open(path));
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
        if (!model_ || model_->size() == 0) { err = "cannot open " + path; return false; }
        const GgufFile & f = model_->shard(0);
        std::map<std::string, const TensorInfo *> a_of, b_of;
        for (const TensorInfo & t : f.tensors()) {
            const size_t p = t.name.rfind(".lora_");
            if (p == std::string::npos) continue;
            const std::string suf = t.name.substr(p + 6);
            const std::string stem = t.name.substr(0, p);
            if (suf == "a") a_of[stem] = &t;
            else if (suf == "b") b_of[stem] = &t;
        }
        (void) 0;
        for (const auto & kv : a_of) {
            const std::string & stem = kv.first;
            auto itb = b_of.find(stem);
            if (itb == b_of.end()) { err = "adapter: " + stem + " has lora_a without lora_b"; return false; }
            int64_t layer = 0;
            Mod mod = NMOD;
            if (!parse_stem(stem, layer, mod)) continue;   // a module we do not apply (ignored)
            const TensorInfo * ta = kv.second, * tb = itb->second;
            if (ta->shape.empty() || tb->shape.empty()) { err = "adapter: empty shape for " + stem; return false; }
            const int64_t n_exp = ta->shape.size() >= 3 ? (int64_t) ta->shape[2] : 1;
            const int64_t a_len = (int64_t) ta->shape[0];
            const int64_t b_len = tb->shape.size() >= 2 ? (int64_t) tb->shape[1] : (int64_t) tb->shape[0];
            std::vector<Lora1> & dst = by_[(int64_t) layer].m[mod];
            dst.assign((size_t) n_exp, Lora1());
            const std::vector<float> va = f16_vec(f, *ta);
            const std::vector<float> vb = f16_vec(f, *tb);
            if (va.size() < (size_t) (n_exp * a_len) || vb.size() < (size_t) (n_exp * b_len)) {
                err = "adapter: " + stem + " payload too small";
                return false;
            }
            for (int64_t e = 0; e < n_exp; ++e) {
                dst[(size_t) e].a.assign(va.begin() + (size_t) e * a_len, va.begin() + (size_t) (e + 1) * a_len);
                dst[(size_t) e].b.assign(vb.begin() + (size_t) e * b_len, vb.begin() + (size_t) (e + 1) * b_len);
            }
        }
        if (by_.empty()) { err = "adapter: no recognised targets in " + path; return false; }
        return true;
    }

    bool has(int64_t layer, Mod m) const {
        auto it = by_.find(layer);
        return it != by_.end() && it->second.m.count(m) != 0;
    }
    /// Routed-expert delta for (layer, module, expert), or nullptr.  GATE/UP for the pre-SwiGLU weights, DOWN after.
    const Lora1 * exps(int64_t layer, Mod m, int64_t expert) const {
        auto it = by_.find(layer);
        if (it == by_.end()) return nullptr;
        auto im = it->second.m.find(m);
        if (im == it->second.m.end() || expert < 0 || (size_t) expert >= im->second.size()) return nullptr;
        return &im->second[(size_t) expert];
    }
    /// Whole-module delta for a 2-D target (SHEXP_DOWN, ATTN_OUT), or nullptr.
    const Lora1 * solo(int64_t layer, Mod m) const {
        auto it = by_.find(layer);
        if (it == by_.end()) return nullptr;
        auto im = it->second.m.find(m);
        if (im == it->second.m.end() || im->second.empty()) return nullptr;
        return &im->second[0];
    }
    /// True when the adapter carries this module on any layer (a cheap "is the LoRA active for X" test).
    bool any(Mod m) const {
        for (const auto & kv : by_) if (kv.second.m.count(m)) return true;
        return false;
    }

private:
    struct PerLayer { std::map<Mod, std::vector<Lora1>> m; };
    std::map<int64_t, PerLayer> by_;
    std::unique_ptr<GgufModel> model_;

    /// "blk.<N>.<module>.weight" -> (N, Mod).  Unknown modules return false (ignored, not an error).
    static bool parse_stem(const std::string & stem, int64_t & layer, Mod & mod) {
        if (stem.rfind("blk.", 0) != 0) return false;
        size_t i = 4;
        int64_t n = 0;
        while (i < stem.size() && stem[i] >= '0' && stem[i] <= '9') n = n * 10 + (stem[i++] - '0');
        if (i >= stem.size() || stem[i] != '.') return false;
        const std::string mod_s = stem.substr(i + 1);
        if (mod_s == "ffn_gate_exps.weight") mod = GATE;
        else if (mod_s == "ffn_up_exps.weight") mod = UP;
        else if (mod_s == "ffn_down_exps.weight") mod = DOWN;
        else if (mod_s == "ffn_down_shexp.weight") mod = SHEXP_DOWN;
        else if (mod_s == "attn_output.weight") mod = ATTN_OUT;
        else return false;
        layer = n;
        return true;
    }

    /// F16 payload of a tensor as host floats (the adapter is F16; anything else yields an empty vector).
    static std::vector<float> f16_vec(const GgufFile & f, const TensorInfo & t) {
        std::vector<float> v;
        if ((int) t.type != (int) GGML_TYPE_F16) return v;
        const ggml_fp16_t * h = (const ggml_fp16_t *) f.tensor_data(t);
        const size_t n = (size_t) tensor_payload_bytes(t) / 2;
        v.resize(n);
        for (size_t i = 0; i < n; ++i) v[i] = ggml_fp16_to_fp32(h[i]);
        return v;
    }
};

}  // namespace strata::glm
