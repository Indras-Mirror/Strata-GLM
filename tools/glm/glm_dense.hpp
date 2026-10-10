// tools/glm/glm_dense.hpp - GLM-5.3-Flash (`glm5-next`) decode of the dense (non-expert) half.
//
// The DS4 engine's shape (tools/ds4/ds4_dense.hpp) for GLM's layers: mHC streams (identical to DeepSeek V4), KDA
// linear attention (34 layers, recurrent state), nope-MLA (11 layers, latent cache), dense FFN (layers 0-2) and the
// MoE router + shared expert (layers 3-44).  The routed experts are Ds4MoeTier's (tools/ds4/ds4_moe.hpp); this class
// hands out the ffn_norm activation they consume and takes back their weighted sum in `finish_layer`.
//
//   begin_tokens(tids, n)                 -> hc_init for n consecutive tokens
//   predict(l, top_ids, n_top)            -> prefetch hint (the layer's router on rms_norm(hc_attn_pre)*ffn_norm.w)
//   attn_router_n(l, pos0, n, ...)        -> attention + router ids/weights + shared expert (or the dense FFN on
//                                            layers 0-2: no routed experts, `routed()` is false)
//   finish_layer_n(l, n, routed_sum)      -> shared/dense FFN + routed sum, hc_post -> next layer state
//   logits_n(n, out)                      -> mean of the 4 streams, output_norm, output
//
// Formulas: docs/glm/ARCHITECTURE.md (from neurall/llama.cpp src/models/glm5-next.cpp; the graph mirrors it op for op).
//
// Phase 1 limits (checked, error otherwise):
//  - The lightning indexer is not built: MLA attends every earlier position, which is exactly the model while the
//    context holds <= indexer_top_k + kpool - 1 tokens (2051).  Longer contexts need the indexer (phase 2).
//  - KDA state is recurrent, not position-indexed: a position cannot be decoded twice (no speculative rollback yet;
//    ggml_gated_delta_net's K-snapshot slots are the planned mechanism).  Positions must arrive in order.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

namespace strata::glm { class LoraAdapter; }

/// What the GGUF says about the dense half (glm5-next.* keys).
struct GlmGeometry {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, vocab = 0, n_ff = 0;
    int64_t hc = 4, hc_sinkhorn_iters = 20;
    double  hc_eps = 1e-6, rms_eps = 1e-5;
    int64_t kda_head_dim = 128, d_conv = 4;
    double  kda_gate_lower_bound = -5.0;
    int64_t q_lora = 1536, kv_lora = 512, k_mla = 256, v_mla = 256;
    int64_t idx_top_k = 2048, idx_kpool = 4, idx_heads = 32, idx_dim = 128;
    double  ln_eps = 1e-6;              ///< attention.layer_norm_epsilon (the indexer key LayerNorm)
    int64_t n_expert = 288, n_expert_used = 8, n_ff_exp = 2048, n_dense_lead = 3;
    double  expert_weights_scale = 2.5;
    bool    expert_weights_norm = true;
    std::vector<double> swiglu_clamp_exp, swiglu_clamp_shexp;
    std::vector<bool>   is_kda;   ///< per layer: head_count_kv == 0
    int64_t context_length = 0;

    bool kda(int64_t l) const { return is_kda[(size_t) l]; }
    bool routed(int64_t l) const { return l >= n_dense_lead; }
    /// Longest context the indexer-free attention is exact for.
    int64_t dense_attn_ctx() const { return idx_top_k + idx_kpool - 1; }
};

struct GlmDenseConfig {
    ggml_backend_t backend = nullptr;   ///< NULL: create and own a CPU backend
    int     n_threads = 8;
    int64_t ctx = 4096;                 ///< MLA latent-cache rows (positions) to allocate
    int     max_tokens = 4;             ///< tokens one pass may carry (decode 1; prompt chunks up to this, <= 4096)
    bool    skip_routed_experts = true; ///< the tier owns ffn_*_exps (never upload them)
    bool    gate_taps = false;          ///< keep host copies of l_out per layer (tap_l_out) for the gates
    bool    allow_long_ctx = false;     ///< dense MLA at any length (NOT the model's math past dense_attn_ctx())
    /// Run-time rank-1 adapter (tools/glm/glm_lora.hpp), or NULL.  The dense half applies the adapter's whole-module
    /// targets (`attn_output`, `ffn_down_shexp`) in the graph as y += B (A x); the routed experts' adapter entries
    /// are Ds4MoeTier's (set separately on the tier).  The adapter must outlive the GlmDense.
    const strata::glm::LoraAdapter * lora = nullptr;
    /// Requantize the dense half's Q6_K matrices (attention, shared experts, dense FFN, output head) to this ggml
    /// type at load (-1 = keep).  Maya-S-v2 -> Q4_K = Maya-S24's layout (its card: +14% decode on a 24 GB card,
    /// 97.7% vs 97.9% of FP8): ~1.9 GB less dense read per token and ~1.9 GB more VRAM for expert slots.
    int dense_requant = -1;
};

class GlmDense {
public:
    struct Impl;

    GlmDense();
    ~GlmDense();
    GlmDense(const GlmDense &) = delete;
    GlmDense & operator=(const GlmDense &) = delete;

    bool init(const std::string & model_path, const GlmDenseConfig & cfg, std::string & err);

    const GlmGeometry & geom() const;
    int64_t n_layer() const;
    int max_tokens() const;
    ggml_backend_t backend() const;

    bool begin_tokens(const int * tids, int n);
    /// Frees the prompt chunks' shared compute buffer (re-acquired by the next chunk): its VRAM goes back to the
    /// device after a prompt, for the elastic expert cache.
    void release_big();
    /// The largest prompt chunk (<= want) starting at pos0 whose MLA cap x chunk stays within the VRAM-safe
    /// product (s28: the indexer's [pools x tokens] scratch grows with cap x n; 65536 x 4096 is the measured-safe size
    /// at margin 7.5).  Halves the chunk for every cap doubling past 64K - long prompts slow down, never OOM.
    int safe_chunk(int pos0, int want) const;
    bool begin_token(int tid) { return begin_tokens(&tid, 1); }
    /// Router hint for token 0 of the pass; false (no hint) on a dense layer.
    bool predict(int il, int * top_ids, int n_top);
    /// n_expert_used ids/weights per token (token-major); *ffn_norm_host = n * n_embd floats (token-major).  On a dense
    /// layer (il < n_dense_lead) no ids are written and the caller passes a zero routed sum to finish_layer_n.
    bool attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w, const float ** ffn_norm_host);
    bool finish_layer_n(int il, int n, const float * routed_sum);
    bool logits_n(int n, const float ** out, int * n_vocab);
    /// Logits of rows [r0, r0+nr) of the last pass (nr <= 16) into out[nr * vocab]; works for any pass size (prompt
    /// chunks: their head is never built for all n rows - 2048 x 154880 floats would not fit).
    bool logits_rows(int r0, int nr, float * out);
    /// Selection-only bias per expert of layer il (cache-aware routing); zeros = the model's routing.
    void set_route_bias(int il, const float * bias);
    /// Forget every position (KDA state, conv state, MLA cache rows are simply overwritten from position 0 on).
    void reset();
    /// Conversation reuse (serve mode): save the recurrent state - KDA S + conv states, the indexer's tail carry -
    /// and the position.  MLA latent rows and complete indexer pools below that position are never rewritten, so
    /// restore() rewinds the model exactly to it (~70 MB device copies, not the whole latent cache).
    bool snapshot();
    bool restore();
    int snapshot_pos() const;   ///< -1: none

    const float * tap_l_out(int il) const;   ///< [n_embd * hc] of the last token of the last pass (gate_taps)
    const std::string & last_error() const;

private:
    std::unique_ptr<Impl> p_;
};
