// tools/glm/glm_dense.cpp - see glm_dense.hpp and docs/glm/ARCHITECTURE.md.
//
// One ggml graph per (layer, pass size n, MLA capacity) for attention + FFN input + router, one per (layer, n) for the
// finish (hc_post), plus init/head/predict.  The op sequence mirrors neurall/llama.cpp src/models/glm5-next.cpp
// (commit 2e0435a) so its intermediate tensors can be compared layer by layer.  Infrastructure (weight upload, one
// input upload per pass, one router readback per layer, graph reuse) follows tools/ds4/ds4_dense.cpp.

#include "glm_dense.hpp"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"   // ggml_cgraph::uid + ggml_graph_next_uid (graph reuse)

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

using namespace strata;

namespace {

constexpr float NEG_INF = -INFINITY;

// Each graph has its own allocator and never changes after it is built: allocate once, then stamp a uid so ggml-cuda
// reuses the captured CUDA graph without re-checking node properties (tools/ds4/ds4_dense.cpp alloc_graph).
bool alloc_graph(ggml_gallocr_t a, ggml_cgraph * g) {
    static const bool reuse = [] { const char * e = std::getenv("GLM_GRAPH_REUSE"); return !e || std::atoi(e) != 0; }();
    if (reuse && g->uid != 0) return true;
    if (!ggml_gallocr_alloc_graph(a, g)) return false;
    if (reuse) g->uid = ggml_graph_next_uid();
    return true;
}

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ------------------------------------------------------------------ metadata

const MetaValue * meta(const GgufFile & f, const std::string & k) { return f.get(k); }

bool meta_num(const GgufFile & f, const std::string & k, double & out) {
    const MetaValue * v = meta(f, k);
    if (!v) return false;
    if (v->is_num()) { out = v->num(); return true; }
    if (v->type == MetaType::ARRAY && !v->items.empty() && v->items[0].is_num()) { out = v->items[0].num(); return true; }
    return false;
}

bool meta_arr(const GgufFile & f, const std::string & k, std::vector<double> & out) {
    const MetaValue * v = meta(f, k);
    if (!v || v->type != MetaType::ARRAY) return false;
    out.clear();
    for (const MetaValue & e : v->items) out.push_back(e.is_num() ? e.num() : 0.0);
    return true;
}

std::string read_geometry(const GgufFile & f, GlmGeometry & g) {
    const MetaValue * a = meta(f, "general.architecture");
    if (!a || a->s != "glm5-next") return "not a glm5-next GGUF (general.architecture = " + (a ? a->s : "?") + ")";
    const std::string p = "glm5-next.";
    struct K { const char * key; int64_t * i; double * d; bool req; };
    double tmp = 0;
    const K keys[] = {
        { "block_count", &g.n_layer, nullptr, true },        { "embedding_length", &g.n_embd, nullptr, true },
        { "attention.head_count", &g.n_head, nullptr, true }, { "vocab_size", &g.vocab, nullptr, false },
        { "feed_forward_length", &g.n_ff, nullptr, true },   { "hyper_connection.count", &g.hc, nullptr, true },
        { "hyper_connection.sinkhorn_iterations", &g.hc_sinkhorn_iters, nullptr, true },
        { "hyper_connection.epsilon", nullptr, &g.hc_eps, true },
        { "attention.layer_norm_rms_epsilon", nullptr, &g.rms_eps, true },
        { "kda.head_dim", &g.kda_head_dim, nullptr, true },  { "ssm.conv_kernel", &g.d_conv, nullptr, true },
        { "kda.gate_lower_bound", nullptr, &g.kda_gate_lower_bound, false },
        { "attention.q_lora_rank", &g.q_lora, nullptr, true }, { "attention.kv_lora_rank", &g.kv_lora, nullptr, true },
        { "attention.key_length_mla", &g.k_mla, nullptr, true }, { "attention.value_length_mla", &g.v_mla, nullptr, true },
        { "attention.indexer.top_k", &g.idx_top_k, nullptr, true }, { "attention.indexer.kpool", &g.idx_kpool, nullptr, true },
        { "expert_count", &g.n_expert, nullptr, true },      { "expert_used_count", &g.n_expert_used, nullptr, true },
        { "expert_feed_forward_length", &g.n_ff_exp, nullptr, true },
        { "leading_dense_block_count", &g.n_dense_lead, nullptr, true },
        { "expert_weights_scale", nullptr, &g.expert_weights_scale, true },
        { "context_length", &g.context_length, nullptr, false },
    };
    for (const K & k : keys) {
        if (!meta_num(f, p + k.key, tmp)) {
            if (k.req) return "missing metadata " + p + k.key;
            continue;
        }
        if (k.i) *k.i = (int64_t) tmp;
        else *k.d = tmp;
    }
    if (const MetaValue * v = meta(f, p + "expert_weights_norm")) g.expert_weights_norm = v->u != 0;
    if (const MetaValue * gf = meta(f, p + "expert_gating_func"); gf && gf->num() != 2)
        return "expert_gating_func " + std::to_string((int) gf->num()) + " (only sigmoid = 2 is implemented)";
    if (const MetaValue * r = meta(f, p + "rope.dimension_count"); r && r->num() != 0)
        return "rope.dimension_count != 0: GLM5-Next MLA is nope-only";
    std::vector<double> kv;
    if (!meta_arr(f, p + "attention.head_count_kv", kv) || (int64_t) kv.size() != g.n_layer)
        return "attention.head_count_kv must be a per-layer array (0 = KDA layer)";
    g.is_kda.assign((size_t) g.n_layer, false);
    for (int64_t l = 0; l < g.n_layer; ++l) g.is_kda[(size_t) l] = kv[(size_t) l] == 0;
    if (!meta_arr(f, p + "swiglu_clamp_exp", g.swiglu_clamp_exp)) g.swiglu_clamp_exp.assign((size_t) g.n_layer, 0.0);
    if (!meta_arr(f, p + "swiglu_clamp_shexp", g.swiglu_clamp_shexp)) g.swiglu_clamp_shexp = g.swiglu_clamp_exp;
    if ((int64_t) g.swiglu_clamp_exp.size() < g.n_layer || (int64_t) g.swiglu_clamp_shexp.size() < g.n_layer)
        return "swiglu_clamp arrays shorter than block_count";
    if (g.hc != 4) return "hyper_connection.count != 4 (the fused mHC ops assume 4)";
    if (g.vocab == 0) {
        const TensorInfo * te = f.find("output.weight");
        if (!te) return "no output.weight";
        g.vocab = (int64_t) te->shape.at(1);
    }
    return "";
}

// ------------------------------------------------------------------ weights (tools/ds4/ds4_dense.cpp load_weights)

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;
    ~WStore() {
        for (ggml_backend_buffer_t b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "[glm_dense] missing weight tensor %s\n", n.c_str());
            std::abort();
        }
        return it->second;
    }
};

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Not loaded: the routed experts (the tier's), the indexer (phase 1), the MTP block.  Off-CPU also token_embd (looked
// up on the host).
bool skip_weight(const std::string & n, bool cpu, bool skip_experts, int64_t n_layer) {
    if (skip_experts && ends_with(n, "_exps.weight")) return true;
    if (n.find("indexer") != std::string::npos) return true;
    if (n.rfind("blk.", 0) == 0 && std::atoll(n.c_str() + 4) >= n_layer) return true;
    return !cpu && n == "token_embd.weight";
}

bool load_weights(const GgufModel & model, ggml_backend_t backend, WStore & w, bool skip_experts, int64_t n_layer,
                  std::string & err) {
    ggml_init_params ip = { /*mem_size*/ 256ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    w.ctx = ggml_init(ip);
    if (!w.ctx) { err = "ggml_init(weights) failed"; return false; }
    const bool cpu = ggml_backend_is_cpu(backend);
    if (cpu) {   // bind every tensor onto the GGUF mmap
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & tensors = sh.tensors();
            if (tensors.empty()) continue;
            const uint8_t * any = sh.tensor_data(tensors[0]);
            uint8_t * base = const_cast<uint8_t *>(any) - sh.data_start() - tensors[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            if (!buf) { err = "cannot wrap " + sh.path(); return false; }
            w.bufs.push_back(buf);
            for (const auto & ti : tensors) {
                if (skip_weight(ti.name, true, skip_experts, n_layer)) continue;
                if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
                int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
                for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
                ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
                ggml_set_name(t, ti.name.c_str());
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
        return true;
    }
    // off-CPU: one backend buffer, payloads uploaded.  BF16 matrices -> Q8_0 (a BF16 mul_mat goes through cuBLAS on
    // CUDA, whose kernels land in VRAM after the expert cache was sized - tools/ds4/ds4_dense.cpp)
    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip_weight(ti.name, false, skip_experts, n_layer)) continue;
            if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_type ty = (ggml_type) ti.type;
            if (ty == GGML_TYPE_BF16 && ti.shape.size() >= 2 && ne[0] % 32 == 0) ty = GGML_TYPE_Q8_0;
            ggml_tensor * t = ggml_new_tensor(w.ctx, ty, (int) ti.shape.size(), ne);
            ggml_set_name(t, ti.name.c_str());
            w.t[ti.name] = t;
        }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buf) { err = "cannot allocate the weight buffer on the backend"; return false; }
    w.bufs.push_back(buf);
    for (size_t s = 0; s < model.size(); ++s)
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip_weight(ti.name, false, skip_experts, n_layer)) continue;
            ggml_tensor * t = w.t[ti.name];
            const uint8_t * src = model.shard(s).tensor_data(ti);
            if ((int) t->type == (int) ti.type) { ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t)); continue; }
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = (int) std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
        }
    return true;
}

}  // namespace

// ================================================================== Impl

struct GlmDense::Impl {
    GlmGeometry g;
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;
    int  n_threads = 8;
    bool cpu = true;
    bool gate_taps = false;
    bool allow_long = false;
    WStore w;
    std::string err;

    int64_t D = 0, HC = 0, NH = 0, SK = 0, DI = 0, DC = 0, KVL = 0, KM = 0, VM = 0;
    int64_t NEXP = 0, NUSED = 0, NT = 1, CAPMAX = 0;

    ggml_context * sctx = nullptr;   // persistent state (sbuf)
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_context * ictx = nullptr;   // input span + router output blocks
    ggml_backend_buffer_t ibuf = nullptr, obuf = nullptr;
    ggml_context * gctx = nullptr;   // graph node structs

    // pass state
    ggml_tensor * x_state = nullptr;     // [D, HC, NT]
    ggml_tensor * routed_sum = nullptr;  // [D, NT]
    ggml_tensor * i_emb = nullptr;       // [D, NT]
    const uint8_t * embd_data = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    std::vector<float> host_emb;

    // inputs (one upload per pass): MLA write slots + causal mask, per-layer route bias
    ggml_tensor * i_slot = nullptr;      // I32 [NT]
    ggml_tensor * i_mask = nullptr;      // F16 [CAPMAX * NT], viewed [cap, n]
    ggml_tensor * i_span = nullptr;
    std::vector<uint8_t> in_host;
    uint8_t * in_base = nullptr;
    int in_pos = -1, in_n = 0;
    int64_t in_cap = -1;
    bool in_dirty = true;
    int next_pos = 0;                    // the position the recurrent state expects next

    int64_t o_blk = 0, o_ids_off = 0, o_wts_off = 0;

    struct Var { int64_t cap = 0, n = 1; ggml_cgraph * gf = nullptr; ggml_gallocr_t allo = nullptr; };
    struct Layer {
        bool kda = false, routed = false;
        // KDA
        ggml_tensor * conv[3] = {};      // [DC-1, DI] q/k/v conv state (time innermost)
        ggml_tensor * S = nullptr;       // [SK, SK, NH] recurrent state
        // MLA
        ggml_tensor * kvc = nullptr;     // F16 [KVL, CAPMAX] latent cache
        // hand-offs
        ggml_tensor * hap = nullptr;     // [D, HC, NT]
        ggml_tensor * post_f = nullptr;  // [HC, NT]
        ggml_tensor * comb_f = nullptr;  // [HC, HC, NT]
        ggml_tensor * ffo = nullptr;     // [D, NT] shared expert (MoE layers) or dense FFN output
        ggml_tensor * t_lout = nullptr;  // [D, HC, NT] tap
        ggml_tensor * rbias = nullptr;   // F32 [NEXP] (input span)
        ggml_tensor * o_f = nullptr, * o_i = nullptr, * o_span = nullptr;   // router output block
        std::vector<uint8_t> host_out;
        std::vector<float> host_fn, host_lout;
        std::vector<Var> vars;
        ggml_cgraph * gf_finish[65] = {};
        ggml_gallocr_t allo_finish[65] = {};
        ggml_cgraph * gf_predict = nullptr;
        ggml_tensor * predict_out = nullptr;
        ggml_gallocr_t allo_predict = nullptr;
    };
    std::vector<Layer> ly;
    std::vector<ggml_cgraph *> gf_init, gf_head;
    std::vector<ggml_gallocr_t> allo_init, allo_head;
    std::vector<ggml_tensor *> logits_t;
    std::vector<float> host_logits;

    ggml_gallocr_t new_allo() const { return ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); }

    ~Impl() {
        for (Layer & L : ly) {
            for (Var & v : L.vars) if (v.allo) ggml_gallocr_free(v.allo);
            for (ggml_gallocr_t a : L.allo_finish) if (a) ggml_gallocr_free(a);
            if (L.allo_predict) ggml_gallocr_free(L.allo_predict);
        }
        for (ggml_gallocr_t a : allo_init) if (a) ggml_gallocr_free(a);
        for (ggml_gallocr_t a : allo_head) if (a) ggml_gallocr_free(a);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (obuf) ggml_backend_buffer_free(obuf);
        if (sctx) ggml_free(sctx);
        if (ictx) ggml_free(ictx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }

    int64_t cap_for(int64_t pos_last) const {
        const int64_t c = std::max<int64_t>(cpu ? 16 : 256, next_pow2(pos_last + 1));
        return std::min<int64_t>(c, CAPMAX);
    }
    bool ensure_n(int64_t n);
};

// ================================================================== graph helpers

namespace {

struct B {
    GlmDense::Impl * im = nullptr;
    ggml_context * c = nullptr;
    const GlmGeometry * g = nullptr;

    ggml_tensor * W(const std::string & n) const { return im->w.get(n); }
    ggml_tensor * BL(int il, const std::string & s) const { return im->w.get("blk." + std::to_string(il) + "." + s); }
    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * v2(ggml_tensor * a, int64_t n0, int64_t n1, size_t nb1, size_t off) const {
        return ggml_view_2d(c, a, n0, n1, nb1, off);
    }
    ggml_tensor * fill_f32(int64_t n0, int64_t n1, float v) const {
        return ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, n0, n1), v);
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wgt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wgt);
    }
    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) const {
        return ggml_add(c, ggml_mul(c, x, scale), base);
    }
    // glm5-next.cpp build_hc_pre (fused ops, as with cparams.fused_dsv4_hc_*)
    ggml_tensor * hc_pre(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                         ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * flat = ggml_reshape_2d(c, x, hc * D, nt);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, ggml_rms_norm(c, flat, (float) g->rms_eps));
        ggml_tensor * pre = v2(mixes, hc, nt, mixes->nb[1], 0);
        pre = ggml_sigmoid(c, hc_affine(pre, v1(sc, 1, 0), v1(bs, hc, 0)));
        pre = ggml_scale_bias(c, pre, 1.0f, (float) g->hc_eps);
        *post = v2(mixes, hc, nt, mixes->nb[1], (size_t) hc * mixes->nb[0]);
        *post = ggml_sigmoid(c, hc_affine(*post, v1(sc, 1, sizeof(float)), v1(bs, hc, (size_t) hc * 4)));
        *post = ggml_scale(c, *post, 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_sinkhorn_iters);
        return ggml_dsv4_hc_pre(c, x, pre);
    }
    ggml_tensor * hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb) const {
        return ggml_dsv4_hc_post(c, x, residual, post, comb);
    }
    // SwiGLU FFN with GLM's clamp (ggml_swiglu_clamp: gate -> min(gate, lim), up -> clamp(up, +-lim))
    ggml_tensor * ffn(ggml_tensor * x, ggml_tensor * up_w, ggml_tensor * gate_w, ggml_tensor * down_w, float lim) const {
        ggml_tensor * up = ggml_mul_mat(c, up_w, x);
        ggml_tensor * gate = ggml_mul_mat(c, gate_w, x);
        ggml_tensor * z = lim > 1e-6f ? ggml_swiglu_clamp(c, gate, up, lim) : ggml_swiglu_split(c, gate, up);
        return ggml_mul_mat(c, down_w, z);
    }
    // FLA's l2 norm (glm5-next.cpp build_gdn_l2_norm)
    ggml_tensor * l2(ggml_tensor * x, float eps) const {
        const float n = (float) x->ne[0];
        return ggml_scale(c, ggml_rms_norm(c, x, eps / n), 1.0f / std::sqrt(n));
    }
};

// KDA: causal depthwise conv of one projection; the conv state keeps the last DC-1 inputs (time innermost)
ggml_tensor * kda_conv(B & b, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                       ggml_tensor * state, int64_t n) {
    GlmDense::Impl & im = *b.im;
    ggml_context * c = b.c;
    ggml_tensor * xp = ggml_mul_mat(c, proj_w, x);                                   // [DI, n]
    ggml_tensor * cx = ggml_concat(c, state, ggml_transpose(c, xp), 0);              // [DC-1+n, DI]
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_view_2d(c, cx, im.DC - 1, im.DI, cx->nb[1],
                                                           (size_t) n * ggml_element_size(cx)), state));
    ggml_tensor * w2 = ggml_reshape_2d(c, conv_w, im.DC, im.DI);
    ggml_tensor * y = ggml_ssm_conv(c, ggml_reshape_3d(c, cx, im.DC - 1 + n, im.DI, 1), w2);   // [DI, n, 1]
    y = ggml_silu(c, ggml_reshape_2d(c, y, im.DI, n));
    return ggml_reshape_4d(c, y, im.SK, im.NH, n, 1);
}

}  // namespace

// attention (KDA or MLA) + ffn hyper-connection + ffn_norm + router / shared expert (or dense FFN), n tokens
static ggml_cgraph * build_attn(GlmDense::Impl & im, int il, int64_t cap, int64_t n) {
    GlmDense::Impl::Layer & L = im.ly[(size_t) il];
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 2048, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC, NH = im.NH;

    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * cur = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                 b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * xn = b.rms_w(cur, b.BL(il, "attn_norm.weight"));                    // [D, n]
    ggml_build_forward_expand(gf, xn);

    ggml_tensor * attn_out = nullptr;
    if (L.kda) {
        ggml_tensor * q = kda_conv(b, gf, xn, b.BL(il, "attn_q.weight"), b.BL(il, "ssm_conv1d_q.weight"), L.conv[0], n);
        ggml_tensor * k = kda_conv(b, gf, xn, b.BL(il, "attn_k.weight"), b.BL(il, "ssm_conv1d_k.weight"), L.conv[1], n);
        ggml_tensor * v = kda_conv(b, gf, xn, b.BL(il, "attn_v.weight"), b.BL(il, "ssm_conv1d_v.weight"), L.conv[2], n);
        // per-channel decay: g = lb * sigmoid(exp(A_log) * (f_b f_a x + dt_bias)); ssm_a holds -exp(A_log)
        ggml_tensor * g1 = ggml_mul_mat(gc, b.BL(il, "ssm_f_b.weight"), ggml_mul_mat(gc, b.BL(il, "ssm_f_a.weight"), xn));
        g1 = ggml_add(gc, g1, b.BL(il, "ssm_dt.bias"));
        g1 = ggml_reshape_3d(gc, g1, im.SK, NH, n);
        g1 = ggml_mul(gc, g1, ggml_reshape_3d(gc, b.BL(il, "ssm_a"), 1, NH, 1));
        g1 = ggml_sigmoid(gc, ggml_scale(gc, g1, -1.0f));
        g1 = ggml_scale(gc, g1, (float) im.g.kda_gate_lower_bound);
        g1 = ggml_reshape_4d(gc, g1, im.SK, NH, n, 1);
        ggml_tensor * beta = ggml_mul_mat(gc, b.BL(il, "ssm_beta.weight"), xn);
        beta = ggml_sigmoid(gc, ggml_reshape_4d(gc, beta, 1, NH, n, 1));
        q = b.l2(q, 1e-6f);
        k = b.l2(k, 1e-6f);
        ggml_tensor * s = ggml_reshape_4d(gc, L.S, im.SK, im.SK, NH, 1);
        ggml_tensor * res = ggml_gated_delta_net(gc, q, k, v, g1, beta, s, 1);
        ggml_tensor * o = ggml_view_4d(gc, res, im.SK, NH, n, 1, ggml_row_size(res->type, im.SK),
                                       ggml_row_size(res->type, im.SK * NH), ggml_row_size(res->type, im.SK * NH * n), 0);
        ggml_tensor * s_new = ggml_view_1d(gc, res, im.SK * im.SK * NH, ggml_row_size(res->type, im.SK * NH * n));
        ggml_build_forward_expand(gf, ggml_cpy(gc, s_new, ggml_reshape_1d(gc, L.S, im.SK * im.SK * NH)));
        // output gate: rmsnorm(o) * w * sigmoid(g_b g_a x)
        ggml_tensor * g2 = ggml_mul_mat(gc, b.BL(il, "ssm_g_b.weight"), ggml_mul_mat(gc, b.BL(il, "ssm_g_a.weight"), xn));
        g2 = ggml_reshape_3d(gc, g2, im.SK, NH, n);
        ggml_tensor * on = b.rms_w(ggml_reshape_3d(gc, ggml_cont(gc, o), im.SK, NH, n), b.BL(il, "ssm_norm.weight"));
        ggml_tensor * gated = ggml_mul(gc, on, ggml_sigmoid(gc, g2));
        attn_out = ggml_mul_mat(gc, b.BL(il, "attn_output.weight"), ggml_cont_2d(gc, gated, im.DI, n));
    } else {
        // nope MLA over every earlier position (phase 1: no indexer - exact up to dense_attn_ctx())
        ggml_tensor * qr = b.rms_w(ggml_mul_mat(gc, b.BL(il, "attn_q_a.weight"), xn), b.BL(il, "attn_q_a_norm.weight"));
        ggml_tensor * q = ggml_mul_mat(gc, b.BL(il, "attn_q_b.weight"), qr);          // [KM*NH, n]
        q = ggml_cont(gc, ggml_permute(gc, ggml_reshape_3d(gc, q, im.KM, NH, n), 0, 2, 1, 3));   // [KM, n, NH]
        ggml_tensor * q_abs = ggml_mul_mat(gc, b.BL(il, "attn_k_b.weight"), q);       // [KVL, n, NH]
        ggml_tensor * kv = b.rms_w(ggml_mul_mat(gc, b.BL(il, "attn_kv_a_mqa.weight"), xn), b.BL(il, "attn_kv_a_norm.weight"));
        ggml_tensor * kvc2 = ggml_set_rows(gc, L.kvc, kv, ggml_view_1d(gc, im.i_slot, n, 0));
        ggml_tensor * kview = ggml_view_3d(gc, kvc2, im.KVL, cap, 1, kvc2->nb[1], kvc2->nb[1] * cap, 0);
        ggml_tensor * mask = ggml_reshape_2d(gc, ggml_view_1d(gc, im.i_mask, cap * n, 0), cap, n);
        ggml_tensor * out = ggml_flash_attn_ext(gc, q_abs, kview, kview, mask, 1.0f / std::sqrt((float) im.KM), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);                              // [KVL, NH, n]
        out = ggml_cont(gc, ggml_permute(gc, out, 0, 2, 1, 3));                        // [KVL, n, NH]
        out = ggml_mul_mat(gc, b.BL(il, "attn_v_b.weight"), out);                      // [VM, n, NH]
        out = ggml_cont_2d(gc, ggml_permute(gc, out, 0, 2, 1, 3), im.VM * NH, n);      // [VM*NH, n]
        attn_out = ggml_mul_mat(gc, b.BL(il, "attn_output.weight"), out);
    }

    ggml_tensor * hap = b.hc_post(attn_out, xin, post_a, comb_a);
    ggml_tensor * post_f = nullptr, * comb_f = nullptr;
    ggml_tensor * fpre = b.hc_pre(hap, b.BL(il, "hc_ffn_fn.weight"), b.BL(il, "hc_ffn_scale.weight"),
                                  b.BL(il, "hc_ffn_base.weight"), &post_f, &comb_f);
    ggml_tensor * fn = b.rms_w(fpre, b.BL(il, "ffn_norm.weight"));                    // [D, n]

    const size_t ob = (size_t) im.o_blk;
    const float lim_sh = (float) im.g.swiglu_clamp_shexp[(size_t) il];
    ggml_tensor * ffo = nullptr;
    if (L.routed) {
        ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fn);
        ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
        ggml_tensor * probs = ggml_sigmoid(gc, rlog);                                  // [NEXP, n]
        ggml_tensor * sel = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
        sel = ggml_add(gc, sel, ggml_reshape_2d(gc, L.rbias, im.NEXP, 1));             // cache-aware bias (0 = model)
        ggml_tensor * selected = ggml_argsort_top_k(gc, sel, (int) im.NUSED);         // [NUSED, n]
        ggml_tensor * wts = ggml_get_rows(gc, ggml_reshape_3d(gc, probs, 1, im.NEXP, n), selected);
        wts = ggml_reshape_2d(gc, wts, im.NUSED, n);
        if (im.g.expert_weights_norm) {
            ggml_tensor * wsum = ggml_clamp(gc, ggml_sum_rows(gc, wts), 6.103515625e-5f, INFINITY);
            wts = ggml_div(gc, wts, wsum);
        }
        wts = ggml_scale(gc, wts, (float) im.g.expert_weights_scale);
        ggml_build_forward_expand(gf, ggml_cpy(gc, selected, ggml_view_2d(gc, L.o_i, im.NUSED, n, ob, (size_t) im.o_ids_off)));
        ggml_build_forward_expand(gf, ggml_cpy(gc, wts, ggml_view_2d(gc, L.o_f, im.NUSED, n, ob, (size_t) im.o_wts_off)));
        ffo = b.ffn(fn, b.BL(il, "ffn_up_shexp.weight"), b.BL(il, "ffn_gate_shexp.weight"),
                    b.BL(il, "ffn_down_shexp.weight"), lim_sh);
    } else {
        ffo = b.ffn(fn, b.BL(il, "ffn_up.weight"), b.BL(il, "ffn_gate.weight"), b.BL(il, "ffn_down.weight"), lim_sh);
    }
    auto cols2 = [&](ggml_tensor * t, int64_t n0) { return ggml_view_2d(gc, t, n0, n, t->nb[1], 0); };
    auto cols3 = [&](ggml_tensor * t, int64_t n0, int64_t n1) { return ggml_view_3d(gc, t, n0, n1, n, t->nb[1], t->nb[2], 0); };
    ggml_build_forward_expand(gf, ggml_cpy(gc, fn, ggml_view_2d(gc, L.o_f, D, n, ob, 0)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_3d(gc, hap, D, HC, n), cols3(L.hap, D, HC)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, post_f, cols2(L.post_f, HC)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_3d(gc, comb_f, HC, HC, n), cols3(L.comb_f, HC, HC)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ffo, cols2(L.ffo, D)));

    GlmDense::Impl::Var v;
    v.cap = cap;
    v.n = n;
    v.gf = gf;
    v.allo = im.new_allo();
    L.vars.push_back(v);
    return gf;
}

// ffn_out = ffo (+ routed sum on MoE layers); l_out = hc_post(ffn_out, hap, post_f, comb_f)
static ggml_cgraph * build_finish(GlmDense::Impl & im, int il, int64_t n) {
    GlmDense::Impl::Layer & L = im.ly[(size_t) il];
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * ffo = ggml_view_2d(gc, L.ffo, D, n, L.ffo->nb[1], 0);
    if (L.routed) ffo = ggml_add(gc, ffo, ggml_view_2d(gc, im.routed_sum, D, n, im.routed_sum->nb[1], 0));
    ggml_tensor * hap = ggml_view_3d(gc, L.hap, D, HC, n, L.hap->nb[1], L.hap->nb[2], 0);
    ggml_tensor * pf = ggml_view_2d(gc, L.post_f, HC, n, L.post_f->nb[1], 0);
    ggml_tensor * cf = ggml_view_3d(gc, L.comb_f, HC, HC, n, L.comb_f->nb[1], L.comb_f->nb[2], 0);
    ggml_tensor * lout = ggml_reshape_3d(gc, b.hc_post(ffo, hap, pf, cf), D, HC, n);
    ggml_build_forward_expand(gf, ggml_cpy(gc, lout, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1],
                                                                  im.x_state->nb[2], 0)));
    if (im.gate_taps)
        ggml_build_forward_expand(gf, ggml_cpy(gc, lout, ggml_view_3d(gc, L.t_lout, D, HC, n, L.t_lout->nb[1],
                                                                      L.t_lout->nb[2], 0)));
    L.gf_finish[n] = gf;
    L.allo_finish[n] = im.new_allo();
    return gf;
}

// predict: the layer's router on rms_norm(hc_attn_pre) * ffn_norm.weight for token 0 (selection scores)
static ggml_cgraph * build_predict(GlmDense::Impl & im, int il) {
    GlmDense::Impl::Layer & L = im.ly[(size_t) il];
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, im.D, im.HC, 1, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post = nullptr, * comb = nullptr;
    ggml_tensor * pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                 b.BL(il, "hc_attn_base.weight"), &post, &comb);
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), b.rms_w(pre, b.BL(il, "ffn_norm.weight")));
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * out = ggml_add(gc, ggml_sigmoid(gc, rlog), b.BL(il, "exp_probs_b.bias"));
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    L.predict_out = out;
    L.gf_predict = gf;
    L.allo_predict = im.new_allo();
    return gf;
}

static ggml_cgraph * build_init(GlmDense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    ggml_tensor * x = ggml_view_3d(gc, im.i_emb, im.D, 1, n, im.i_emb->nb[1], im.i_emb->nb[1], 0);
    x = ggml_repeat_4d(gc, x, im.D, im.HC, n, 1);
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, ggml_view_3d(gc, im.x_state, im.D, im.HC, n, im.x_state->nb[1],
                                                               im.x_state->nb[2], 0)));
    return gf;
}

// head: mean of the hc streams -> output_norm -> output (glm5-next.cpp: build_hc_mean, no hc_head weights)
static ggml_cgraph * build_head(GlmDense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    B b { &im, gc, &im.g };
    ggml_tensor * x = ggml_view_3d(gc, im.x_state, im.D, im.HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * acc = ggml_view_2d(gc, x, im.D, n, x->nb[2], 0);
    for (int64_t s = 1; s < im.HC; ++s) acc = ggml_add(gc, acc, ggml_view_2d(gc, x, im.D, n, x->nb[2], (size_t) s * x->nb[1]));
    acc = ggml_scale(gc, acc, 1.0f / (float) im.HC);
    ggml_tensor * lg = ggml_mul_mat(gc, b.W("output.weight"), b.rms_w(acc, b.W("output_norm.weight")));
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);
    im.logits_t[(size_t) n] = lg;
    return gf;
}

bool GlmDense::Impl::ensure_n(int64_t n) {
    if (n < 1 || n > NT) { err = "pass size " + std::to_string(n) + " out of range 1.." + std::to_string(NT); return false; }
    if (!gf_init[(size_t) n]) {
        gf_init[(size_t) n] = build_init(*this, n);
        allo_init[(size_t) n] = new_allo();
        gf_head[(size_t) n] = build_head(*this, n);
        allo_head[(size_t) n] = new_allo();
    }
    for (int il = 0; il < (int) ly.size(); ++il)
        if (!ly[(size_t) il].gf_finish[n]) build_finish(*this, il, n);
    return true;
}

// ================================================================== public API

GlmDense::GlmDense() : p_(new Impl()) {}
GlmDense::~GlmDense() = default;

const GlmGeometry & GlmDense::geom() const { return p_->g; }
int64_t GlmDense::n_layer() const { return p_->g.n_layer; }
int GlmDense::max_tokens() const { return (int) p_->NT; }
ggml_backend_t GlmDense::backend() const { return p_->backend; }
const std::string & GlmDense::last_error() const { return p_->err; }

bool GlmDense::init(const std::string & model_path, const GlmDenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.n_threads = cfg.n_threads;
    im.gate_taps = cfg.gate_taps;
    im.allow_long = cfg.allow_long_ctx;
    im.backend = cfg.backend;
    if (!im.backend) {
        im.backend = ggml_backend_cpu_init();
        im.own_backend = true;
        if (!im.backend) { err = "cannot create a CPU backend"; return false; }
    }
    im.cpu = ggml_backend_is_cpu(im.backend);
    if (im.cpu) ggml_backend_cpu_set_n_threads(im.backend, im.n_threads);

    im.model = std::make_unique<GgufModel>(GgufModel::open(model_path));
    if (im.model->size() == 0) { err = "cannot open " + model_path; return false; }
    if (!(err = read_geometry(im.model->shard(0), im.g)).empty()) return false;
    const GlmGeometry & g = im.g;
    if (!load_weights(*im.model, im.backend, im.w, cfg.skip_routed_experts, g.n_layer, err)) return false;
    {
        size_t sh = 0;
        const TensorInfo * te = im.model->find("token_embd.weight", &sh);
        if (!te) { err = "no token_embd.weight"; return false; }
        im.embd_data = im.model->shard(sh).tensor_data(*te);
        im.embd_type = (int) te->type;
        im.embd_row = ggml_row_size((ggml_type) te->type, (int64_t) te->shape.at(0));
    }
    im.D = g.n_embd; im.HC = g.hc; im.NH = g.n_head; im.SK = g.kda_head_dim; im.DI = g.kda_head_dim * g.n_head;
    im.DC = g.d_conv; im.KVL = g.kv_lora; im.KM = g.k_mla; im.VM = g.v_mla;
    im.NEXP = g.n_expert; im.NUSED = g.n_expert_used;
    im.NT = std::max(1, cfg.max_tokens);
    im.CAPMAX = std::max<int64_t>(im.cpu ? 16 : 256, next_pow2(std::max<int64_t>(1, cfg.ctx)));
    if (!im.allow_long && im.CAPMAX > g.dense_attn_ctx()) im.CAPMAX = std::max<int64_t>(im.cpu ? 16 : 256, next_pow2(g.dense_attn_ctx()));
    if (im.NT > 64) { err = "max_tokens > 64"; return false; }

    // ---- persistent state
    ggml_init_params ip = { 64ull * 1024 * 1024, nullptr, true };
    im.sctx = ggml_init(ip);
    im.ictx = ggml_init(ip);
    if (!im.sctx || !im.ictx) { err = "ggml_init(state) failed"; return false; }
    const int64_t NT = im.NT;
    im.x_state = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, NT);
    im.routed_sum = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, NT);
    im.i_emb = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, NT);
    im.ly.resize((size_t) g.n_layer);
    for (int64_t il = 0; il < g.n_layer; ++il) {
        Impl::Layer & L = im.ly[(size_t) il];
        L.kda = g.kda(il);
        L.routed = g.routed(il);
        if (L.kda) {
            for (auto & cs : L.conv) cs = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.DC - 1, im.DI);
            L.S = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.SK, im.SK, im.NH);
        } else {
            L.kvc = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F16, im.KVL, im.CAPMAX);
        }
        L.hap = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, NT);
        L.post_f = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.HC, NT);
        L.comb_f = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.HC, im.HC, NT);
        L.ffo = ggml_new_tensor_2d(im.sctx, GGML_TYPE_F32, im.D, NT);
        if (im.gate_taps) {
            L.t_lout = ggml_new_tensor_3d(im.sctx, GGML_TYPE_F32, im.D, im.HC, NT);
            L.host_lout.resize((size_t) (im.D * im.HC));
        }
        L.host_fn.resize((size_t) (im.D * NT));
    }
    im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.backend));
    if (!im.sbuf) { err = "cannot allocate the decode state"; return false; }
    ggml_backend_buffer_clear(im.sbuf, 0);

    // ---- input span: i_slot, i_mask, every layer's rbias, back to back
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(im.backend);
    const size_t al = std::max<size_t>(64, ggml_backend_buft_get_alignment(buft));
    auto up = [&](size_t n) { return (n + al - 1) / al * al; };
    std::vector<ggml_tensor *> in_list;
    im.i_slot = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, NT);
    im.i_mask = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F16, im.CAPMAX * NT);
    in_list = { im.i_slot, im.i_mask };
    for (Impl::Layer & L : im.ly) { L.rbias = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, im.NEXP); in_list.push_back(L.rbias); }
    size_t in_bytes = 0;
    for (ggml_tensor * t : in_list) in_bytes += up(ggml_nbytes(t));
    im.ibuf = ggml_backend_buft_alloc_buffer(buft, in_bytes + al);
    if (!im.ibuf) { err = "cannot allocate the input span"; return false; }
    im.in_base = (uint8_t *) ggml_backend_buffer_get_base(im.ibuf);
    {
        size_t off = 0;
        for (ggml_tensor * t : in_list) {
            if (ggml_backend_tensor_alloc(im.ibuf, t, im.in_base + off) != GGML_STATUS_SUCCESS) { err = "cannot place an input"; return false; }
            off += up(ggml_nbytes(t));
        }
        im.i_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) in_bytes);
        if (ggml_backend_tensor_alloc(im.ibuf, im.i_span, im.in_base) != GGML_STATUS_SUCCESS) { err = "cannot place the input span"; return false; }
        im.in_host.assign(in_bytes, 0);
    }
    // ---- router output blocks: per layer NT x [fn | ids | wts], one readback per layer
    const size_t o1 = up(ggml_row_size(GGML_TYPE_F32, im.D)), o2 = o1 + up(4 * im.NUSED), blk = o2 + up(4 * im.NUSED);
    im.o_blk = (int64_t) blk; im.o_ids_off = (int64_t) o1; im.o_wts_off = (int64_t) o2;
    const size_t lblk = blk * (size_t) NT;
    im.obuf = ggml_backend_buft_alloc_buffer(buft, lblk * (size_t) g.n_layer + al);
    if (!im.obuf) { err = "cannot allocate the router output blocks"; return false; }
    {
        uint8_t * ob = (uint8_t *) ggml_backend_buffer_get_base(im.obuf);
        for (int64_t il = 0; il < g.n_layer; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            uint8_t * b0 = ob + (size_t) il * lblk;
            L.o_f = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, (int64_t) (lblk / 4));
            L.o_i = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, (int64_t) (lblk / 4));
            L.o_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) lblk);
            for (ggml_tensor * t : { L.o_f, L.o_i, L.o_span })
                if (ggml_backend_tensor_alloc(im.obuf, t, b0) != GGML_STATUS_SUCCESS) { err = "cannot place a router block"; return false; }
            L.host_out.assign(lblk, 0);
        }
    }

    ggml_init_params gp = { 512ull * 1024 * 1024, nullptr, true };
    im.gctx = ggml_init(gp);
    if (!im.gctx) { err = "ggml_init(graph) failed"; return false; }
    im.gf_init.assign((size_t) NT + 1, nullptr);
    im.gf_head.assign((size_t) NT + 1, nullptr);
    im.allo_init.assign((size_t) NT + 1, nullptr);
    im.allo_head.assign((size_t) NT + 1, nullptr);
    im.logits_t.assign((size_t) NT + 1, nullptr);
    if (!im.ensure_n(1)) { err = im.err; return false; }
    for (int64_t il = 0; il < g.n_layer; ++il)
        if (im.ly[(size_t) il].routed) build_predict(im, (int) il);
    im.host_logits.resize((size_t) (g.vocab * NT));
    return true;
}

void GlmDense::reset() {
    Impl & im = *p_;
    ggml_backend_buffer_clear(im.sbuf, 0);
    im.next_pos = 0;
    im.in_pos = -1;
}

bool GlmDense::begin_tokens(const int * tids, int n) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_init[(size_t) n], im.gf_init[(size_t) n])) { im.err = "gallocr(init) failed"; return false; }
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        if (tids[t] < 0 || tids[t] >= im.g.vocab) { im.err = "token id out of range"; return false; }
        const uint8_t * row = im.embd_data + (size_t) tids[t] * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (ggml_backend_graph_compute(im.backend, im.gf_init[(size_t) n]) != GGML_STATUS_SUCCESS) {
        im.err = "init graph compute failed";
        return false;
    }
    return true;
}

bool GlmDense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || n_top <= 0) { im.err = "predict: bad args"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    for (int i = 0; i < n_top; ++i) top_ids[i] = -1;
    if (!L.routed) return false;
    if (!alloc_graph(L.allo_predict, L.gf_predict)) { im.err = "gallocr(predict) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, L.gf_predict) != GGML_STATUS_SUCCESS) { im.err = "predict compute failed"; return false; }
    std::vector<float> sel((size_t) ggml_nelements(L.predict_out));
    ggml_backend_tensor_get(L.predict_out, sel.data(), 0, sel.size() * 4);
    std::vector<int> idx(sel.size());
    for (size_t i = 0; i < sel.size(); ++i) idx[i] = (int) i;
    const int k = std::min<int>(n_top, (int) sel.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    for (int i = 0; i < k; ++i) top_ids[i] = idx[(size_t) i];
    return true;
}

void GlmDense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return;
    Impl::Layer & L = im.ly[(size_t) il];
    std::memcpy(im.in_host.data() + ((uint8_t *) L.rbias->data - im.in_base), bias, (size_t) im.NEXP * 4);
    im.in_dirty = true;
}

bool GlmDense::attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w, const float ** ffn_norm_host) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "attn_router: bad layer"; return false; }
    if (n < 1 || n > im.NT) { im.err = "attn_router: pass size out of range"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    const int64_t pos_last = (int64_t) pos0 + n - 1;
    if (il == 0) {
        if (pos0 != im.next_pos) {
            im.err = "attn_router: position " + std::to_string(pos0) + " but the recurrent state is at " +
                     std::to_string(im.next_pos) + " (KDA state cannot rewind; reset() to start over)";
            return false;
        }
        if (pos_last >= im.CAPMAX) {
            im.err = pos_last >= im.g.dense_attn_ctx() && !im.allow_long
                ? "context beyond " + std::to_string(im.g.dense_attn_ctx()) + " tokens needs the lightning indexer (phase 2)"
                : "context beyond the MLA cache (" + std::to_string(im.CAPMAX) + " rows; raise ctx)";
            return false;
        }
    }
    const int64_t cap = L.kda ? 0 : im.cap_for(pos_last);
    Impl::Var * var = nullptr;
    for (Impl::Var & v : L.vars) if (v.cap == cap && v.n == n) { var = &v; break; }
    if (!var) { build_attn(im, il, cap, n); var = &L.vars.back(); }
    if (!alloc_graph(var->allo, var->gf)) { im.err = "gallocr(attn) failed"; return false; }

    // inputs: one upload per pass (positions or capacity changed, or a route bias was set)
    const int64_t mcap = im.cap_for(pos_last);
    if (im.in_dirty || pos0 != im.in_pos || n != im.in_n || mcap != im.in_cap) {
        int32_t * slot = (int32_t *) (im.in_host.data() + ((uint8_t *) im.i_slot->data - im.in_base));
        ggml_fp16_t * mask = (ggml_fp16_t *) (im.in_host.data() + ((uint8_t *) im.i_mask->data - im.in_base));
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(NEG_INF);
        for (int t = 0; t < n; ++t) {
            slot[t] = pos0 + t;
            for (int64_t j = 0; j < mcap; ++j) mask[(size_t) t * mcap + j] = j <= pos0 + t ? z : ninf;
        }
        ggml_backend_tensor_set(im.i_span, im.in_host.data(), 0, im.in_host.size());
        im.in_pos = pos0; im.in_n = n; im.in_cap = mcap; im.in_dirty = false;
    }
    if (ggml_backend_graph_compute(im.backend, var->gf) != GGML_STATUS_SUCCESS) { im.err = "attn graph compute failed"; return false; }
    if (il == (int) im.ly.size() - 1) im.next_pos = pos0 + n;

    // one readback: n blocks [fn | ids | wts]
    const size_t ob = (size_t) im.o_blk;
    ggml_backend_tensor_get(L.o_span, L.host_out.data(), 0, ob * (size_t) n);
    for (int t = 0; t < n; ++t) {
        const uint8_t * blk = L.host_out.data() + (size_t) t * ob;
        std::memcpy(L.host_fn.data() + (size_t) t * im.D, blk, (size_t) im.D * 4);
        if (L.routed) {
            const int32_t * ids = (const int32_t *) (blk + im.o_ids_off);
            const float * w = (const float *) (blk + im.o_wts_off);
            for (int k = 0; k < im.NUSED; ++k) {
                routed_ids[t * im.NUSED + k] = ids[k];
                routed_w[t * im.NUSED + k] = w[k];
            }
        }
    }
    if (ffn_norm_host) *ffn_norm_host = L.host_fn.data();
    return true;
}

bool GlmDense::finish_layer_n(int il, int n, const float * routed_sum) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || n < 1 || n > im.NT) { im.err = "finish_layer: bad args"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    if (L.routed) {
        if (!routed_sum) { im.err = "finish_layer: routed sum missing"; return false; }
        ggml_backend_tensor_set(im.routed_sum, routed_sum, 0, (size_t) (im.D * n) * 4);
    }
    if (!alloc_graph(L.allo_finish[n], L.gf_finish[n])) { im.err = "gallocr(finish) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, L.gf_finish[n]) != GGML_STATUS_SUCCESS) { im.err = "finish compute failed"; return false; }
    if (im.gate_taps)
        ggml_backend_tensor_get(L.t_lout, L.host_lout.data(), (size_t) (n - 1) * L.t_lout->nb[2], L.host_lout.size() * 4);
    return true;
}

bool GlmDense::logits_n(int n, const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_head[(size_t) n], im.gf_head[(size_t) n])) { im.err = "gallocr(head) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_head[(size_t) n]) != GGML_STATUS_SUCCESS) { im.err = "head compute failed"; return false; }
    ggml_backend_tensor_get(im.logits_t[(size_t) n], im.host_logits.data(), 0, (size_t) (im.g.vocab * n) * 4);
    *out = im.host_logits.data();
    *n_vocab = (int) im.g.vocab;
    return true;
}

const float * GlmDense::tap_l_out(int il) const {
    const Impl & im = *p_;
    if (!im.gate_taps || il < 0 || il >= (int) im.ly.size()) return nullptr;
    return im.ly[(size_t) il].host_lout.data();
}
