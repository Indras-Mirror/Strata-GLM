// tools/glm/glm_generate.cpp - the Strata GLM-5.3-Flash engine end to end: GlmDense (attention + router + shared
// expert/dense FFN, CUDA or CPU ggml backend) + Ds4MoeTier (routed experts: VRAM cache, prefetch, CPU pool, PCIe share,
// host arena) + greedy/temperature sampling.  Token ids in, token ids out (one per line on stdout).
//
//   glm_generate -m model.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu]
//                [--slots auto|N] [--arena-gib 60] [--profile routes.bin] [--vram-lru] [--route-bias 0.05]
//                [--ctx 2048] [--temp 0] [--stop id,id] [--ppl] [--dump-logits f.f32] [--dump-routes f.bin]
//
// Per token, per layer: predict(l) -> tier.prefetch(l); attn_router(l); tier.run(l) (MoE layers only); finish_layer(l).
// The prompt goes through the same one-token path (phase 1: no prompt chunks yet).
// Load the real model ONLY through tools/ds4/memguard.sh.
//
// --dump-routes FILE: every token's routed ids, [512-token block][routed layer][token][top_k] u16 (the format
// Ds4MoeTier::seed_from_routes(file, 512) ranks) - a GLM routing profile for --profile.  Without --profile the
// arena and VRAM cache are filled in (layer, expert) index order.
#include "glm_dense.hpp"
#include "ds4_moe.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model, ids_csv, ids_file, profile, dump_logits, dump_routes;
    int n_predict = 64;
    std::string backend = "cuda", experts = "gpu";
    int64_t slots = 0;   // 0 = auto
    double vram_margin_gib = 1.0, pcie = 0.25, pf_b = 1.43, arena_gib = 60.0;
    int threads = 0;
    int64_t ctx = 0;
    float temp = 0.0f, route_bias = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool ppl = false, vram_lru = false, arena_adapt = false, arena_skip = false;
};

std::vector<int> parse_csv(const std::string & s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

bool parse(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--ids") a.ids_csv = next();
        else if (k == "--ids-file") a.ids_file = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::atoi(next().c_str());
        else if (k == "--backend") a.backend = next();
        else if (k == "--experts") a.experts = next();
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--pcie") a.pcie = std::atof(next().c_str());
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--profile") a.profile = next();
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--dump-routes") a.dump_routes = next();
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--arena-skip-resident") a.arena_skip = true;
        else { std::fprintf(stderr, "glm_generate: unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && (!a.ids_csv.empty() || !a.ids_file.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int sample(const float * logits, int n_vocab, float temp, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const float mx = *std::max_element(logits, logits + n_vocab);
    std::vector<double> p((size_t) n_vocab);
    double z = 0;
    for (int i = 0; i < n_vocab; ++i) z += (p[(size_t) i] = std::exp((double) (logits[i] - mx) / temp));
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n_vocab; ++i) if ((r -= p[(size_t) i]) <= 0) return i;
    return n_vocab - 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("GLM_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) { if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr); }, nullptr);
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: glm_generate -m model.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu]\n"
                             "       [--experts gpu|cpu] [--slots auto|N] [--arena-gib 60] [--profile routes.bin] [--vram-lru]\n"
                             "       [--route-bias X] [--ctx N] [--temp 0] [--stop id,id] [--ppl] [--dump-logits f] [--dump-routes f]\n");
        return 2;
    }
    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "glm_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        const size_t n = (size_t) f.tellg() / 4;
        f.seekg(0);
        prompt.resize(n);
        f.read((char *) prompt.data(), (std::streamsize) (n * 4));
    }
    if (prompt.empty()) { std::fprintf(stderr, "glm_generate: empty prompt\n"); return 2; }

    ggml_backend_t be = ggml_backend_init_by_type(a.backend == "cuda" ? GGML_BACKEND_DEVICE_TYPE_GPU
                                                                      : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::fprintf(stderr, "glm_generate: no %s backend in this build\n", a.backend.c_str()); return 2; }
    std::fprintf(stderr, "dense backend: %s\n", ggml_backend_name(be));

    const double t_load0 = now_ms();
    GlmDense dense;
    GlmDenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 8;
    dc.max_tokens = 1;
    std::string err;
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "glm_generate: dense init: %s\n", err.c_str()); return 1; }
    const GlmGeometry & G = dense.geom();
    std::fprintf(stderr, "glm5-next: %lld layers (%lld KDA, %lld MLA), %lld experts top-%lld, vocab %lld; dense half loaded "
                         "in %.1f s\n", (long long) G.n_layer,
                 (long long) std::count(G.is_kda.begin(), G.is_kda.end(), true),
                 (long long) std::count(G.is_kda.begin(), G.is_kda.end(), false), (long long) G.n_expert,
                 (long long) G.n_expert_used, (long long) G.vocab, (now_ms() - t_load0) / 1000.0);
    if (a.slots <= 0 && a.experts != "cpu") {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        // one expert: gate+up+down of the largest layer type (Q4_K ~ 14.2 MB, Q2_K ~ 8.3 MB): size by the tier's
        // blob_bytes after init would be exact; use the Q4_K size here (conservative), the tier caps at what fits
        const double blob = 3.0 * (double) G.n_embd * G.n_ff_exp * 0.5625;
        a.slots = std::max<int64_t>(0, (int64_t) (((double) fr - a.vram_margin_gib * 1073741824.0) / blob));
        std::fprintf(stderr, "slots auto: %.2f GiB free after the dense half -> %lld slots (margin %.2f GiB)\n",
                     fr / 1073741824.0, (long long) a.slots, a.vram_margin_gib);
    }

    namespace ds4 = strata::ds4;
    ds4::Ds4MoeTier tier;
    ds4::Ds4MoeConfig mc;
    mc.slots = a.slots;
    mc.pcie_frac = a.pcie;
    mc.pf_b = a.pf_b;
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;
    mc.cpu_only = a.experts == "cpu";
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.arena_skip_resident = a.arena_skip;
    if (!tier.init(a.model, mc, err)) { std::fprintf(stderr, "glm_generate: tier init: %s\n", err.c_str()); return 1; }
    const ds4::Ds4MoeGeom & TG = tier.geom();
    if (!mc.cpu_only) {
        bool ok;
        if (!a.profile.empty()) ok = tier.seed_from_routes(a.profile, 512, err);
        else {   // no profile: index order (a GLM profile comes from --dump-routes)
            std::vector<std::pair<int32_t, int32_t>> ranked;
            for (int64_t l = 0; l < TG.n_layers; ++l)
                if (TG.routed(l))
                    for (int64_t e = 0; e < TG.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
            ok = tier.seed_from_ranked(ranked, err);
        }
        if (!ok) { std::fprintf(stderr, "glm_generate: seed: %s\n", err.c_str()); return 1; }
    }
    std::fprintf(stderr, "tier: %s, %lld resident slots, arena %.1f GiB (%lld experts), file tier %lld; load %.1f s\n",
                 tier.mode(), (long long) tier.resident(), tier.arena_gib(), (long long) tier.arena_experts(),
                 (long long) tier.file_tier(), (now_ms() - t_load0) / 1000.0);

    const int n_layer = (int) G.n_layer;
    const int top_k = (int) TG.top_k;
    const int kPredW = ds4::Ds4MoeConfig::kPredW;
    const int64_t n_embd = G.n_embd;
    std::vector<int> ids_i((size_t) top_k), pred_i((size_t) kPredW);
    std::vector<int32_t> ids32((size_t) top_k), pred32((size_t) kPredW);
    std::vector<float> w((size_t) top_k), routed((size_t) n_embd), rb((size_t) G.n_expert);
    std::mt19937_64 rng(a.seed);

    // --dump-routes: [512-token block][routed layer][token][top_k] u16
    std::vector<int> routed_layers;
    for (int l = 0; l < n_layer; ++l) if (G.routed(l)) routed_layers.push_back(l);
    const int RB = 512;
    std::vector<uint16_t> rblock;
    int rblock_n = 0;
    std::FILE * rfile = a.dump_routes.empty() ? nullptr : std::fopen(a.dump_routes.c_str(), "wb");
    auto flush_routes = [&](bool partial) {
        if (!rfile || rblock_n == 0 || (rblock_n < RB && !partial)) return;
        if (rblock_n < RB) return;   // the ranker reads whole blocks only
        std::fwrite(rblock.data(), 2, rblock.size(), rfile);
        rblock_n = 0;
        std::fill(rblock.begin(), rblock.end(), 0);
    };
    if (rfile) rblock.assign((size_t) RB * routed_layers.size() * top_k, 0);

    double t_ph[5] = { 0, 0, 0, 0, 0 };
    bool timing = false;
    auto step = [&](int tid, int pos) -> bool {
        if (a.route_bias != 0.0f && !mc.cpu_only)
            for (int l = 0; l < n_layer; ++l) {
                if (!G.routed(l)) continue;
                for (int64_t e = 0; e < G.n_expert; ++e) rb[(size_t) e] = tier.resident(l, e) ? a.route_bias : 0.0f;
                dense.set_route_bias(l, rb.data());
            }
        if (!dense.begin_token(tid)) return false;
        int ri = 0;
        for (int l = 0; l < n_layer; ++l) {
            const bool moe = G.routed(l);
            double t0 = timing ? now_ms() : 0, t1;
            if (moe && !mc.cpu_only && mc.pf_b > 0 && dense.predict(l, pred_i.data(), kPredW)) {
                int n = 0;
                for (int i = 0; i < kPredW && pred_i[(size_t) i] >= 0; ++i) pred32[(size_t) n++] = pred_i[(size_t) i];
                if (n > 0) tier.prefetch(l, pred32.data(), n);
            }
            if (timing) { t1 = now_ms(); t_ph[0] += t1 - t0; t0 = t1; }
            const float * x = nullptr;
            if (!dense.attn_router_n(l, pos, 1, ids_i.data(), w.data(), &x)) return false;
            if (timing) { t1 = now_ms(); t_ph[1] += t1 - t0; t0 = t1; }
            if (moe) {
                for (int k = 0; k < top_k; ++k) ids32[(size_t) k] = ids_i[(size_t) k];
                if (rfile) {
                    uint16_t * dst = rblock.data() + ((size_t) ri * RB + rblock_n) * top_k;
                    for (int k = 0; k < top_k; ++k) dst[k] = (uint16_t) ids_i[(size_t) k];
                }
                if (!tier.run(l, ids32.data(), w.data(), x, routed.data())) {
                    std::fprintf(stderr, "glm_generate: tier.run refused at layer %d\n", l);
                    return false;
                }
                ++ri;
            }
            if (timing) { t1 = now_ms(); t_ph[2] += t1 - t0; t0 = t1; }
            if (!dense.finish_layer_n(l, 1, moe ? routed.data() : nullptr)) return false;
            if (timing) { t1 = now_ms(); t_ph[3] += t1 - t0; }
        }
        if (rfile && ++rblock_n == RB) flush_routes(false);
        return true;
    };

    // ---- prefill (decode loop over the prompt)
    tier.reset_stats();
    const double t_pf0 = now_ms();
    double nll = 0;
    int n_scored = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!step(prompt[i], (int) i)) {
            std::fprintf(stderr, "glm_generate: prefill failed at %zu: %s\n", i, dense.last_error().c_str());
            return 1;
        }
        if (a.ppl && i + 1 < prompt.size()) {
            const float * l = nullptr;
            int nv = 0;
            if (!dense.logits_n(1, &l, &nv)) return 1;
            const float mx = *std::max_element(l, l + nv);
            double z = 0;
            for (int v = 0; v < nv; ++v) z += std::exp((double) (l[v] - mx));
            nll += (std::log(z) + mx) - l[prompt[i + 1]];
            ++n_scored;
        }
    }
    if (a.ppl && n_scored)
        std::fprintf(stderr, "ppl: %d positions, mean NLL %.5f, perplexity %.4f\n", n_scored, nll / n_scored, std::exp(nll / n_scored));
    const double pf_ms = now_ms() - t_pf0;

    // ---- decode
    tier.reset_stats();
    int pos = (int) prompt.size(), n_gen = 0;
    const double t_dec0 = now_ms();
    timing = true;
    for (; n_gen < a.n_predict; ++n_gen) {
        const double th0 = now_ms();
        const float * lg = nullptr;
        int n_vocab = 0;
        if (!dense.logits_n(1, &lg, &n_vocab)) { std::fprintf(stderr, "glm_generate: logits: %s\n", dense.last_error().c_str()); return 1; }
        t_ph[4] += now_ms() - th0;
        if (n_gen == 0 && !a.dump_logits.empty())
            if (std::FILE * f = std::fopen(a.dump_logits.c_str(), "wb")) { std::fwrite(lg, 4, (size_t) n_vocab, f); std::fclose(f); }
        const int tok = sample(lg, n_vocab, a.temp, rng);
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end()) { ++n_gen; break; }
        if (n_gen + 1 == a.n_predict) { ++n_gen; break; }
        if (!step(tok, pos++)) {
            std::fprintf(stderr, "glm_generate: decode failed at pos %d: %s\n", pos - 1, dense.last_error().c_str());
            return 1;
        }
    }
    if (rfile) { flush_routes(true); std::fclose(rfile); }
    const double dec_ms = now_ms() - t_dec0;
    const ds4::Ds4MoeStats st = tier.stats();
    const int dec_steps = std::max(1, n_gen - 1);
    std::fprintf(stderr, "\nprefill: %zu tokens in %.2f s = %.2f tok/s (decode-loop prefill)\n", prompt.size(),
                 pf_ms / 1000.0, 1000.0 * (double) prompt.size() / pf_ms);
    std::fprintf(stderr, "decode : %d tokens, %d forward passes in %.2f s = %.2f tok/s\n", n_gen, dec_steps,
                 dec_ms / 1000.0, 1000.0 * dec_steps / dec_ms);
    const double look = (double) std::max<int64_t>(1, st.lookups());
    std::fprintf(stderr, "experts/token: hit %.1f%% (prefetched-useful %.1f/token of %.1f issued), cpu %.1f%%, pcie %.1f%%, "
                         "file tier %lld\n", 100.0 * (double) st.hits / look, (double) st.prefetched_useful / dec_steps,
                 (double) st.prefetch_issued / dec_steps, 100.0 * (double) st.cpu / look, 100.0 * (double) st.pcie / look,
                 (long long) st.file_tier);
    if (a.vram_lru || st.file_tier > 0)
        std::fprintf(stderr, "tier moves: vram_lru swaps %.2f/pass (demoted %lld), arena swaps %lld, file reads %.2f ms/pass\n",
                     (double) st.vram_swaps / dec_steps, (long long) st.vram_demotes, (long long) st.arena_swaps,
                     st.file_ms / dec_steps);
    std::fprintf(stderr, "decode ms/token: predict+prefetch %.2f, attention+router %.2f, experts %.2f, finish %.2f, head %.2f\n",
                 t_ph[0] / dec_steps, t_ph[1] / dec_steps, t_ph[2] / dec_steps, t_ph[3] / dec_steps, t_ph[4] / dec_steps);
    std::fprintf(stderr, "tier ms/token: wall %.2f (gpu hits %.2f, pcie %.2f, cpu pool %.2f)\n", st.wall_ms / dec_steps,
                 st.hit_ms / dec_steps, st.pcie_ms / dec_steps, st.cpu_ms / dec_steps);
    {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        std::fprintf(stderr, "VRAM free at the end: %.2f GiB\n", fr / 1073741824.0);
    }
    tier.close();
    ggml_backend_free(be);
    return 0;
}
