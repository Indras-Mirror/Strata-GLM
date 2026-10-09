// tools/glm/lora_bake.cpp - bake the rank-1 GLM abliteration LoRA into a GGUF's weights.
//
// Semantics MATCH tools/glm/glm_lora.hpp: for a linear W (ne0=in, ne1=out[, ne2=n_expert]) the adapter
// applies y += b * dot(a, x), i.e. delta[j][i] = b[j]*a[i].  We dequantize the base tensor, add the delta
// in f32, and requantize back to the SAME ggml type (so the file stays byte-identical in structure).
//
//   lora_bake --base B.gguf --lora L.gguf --test [--only SUBSTR]   # metrics only, writes nothing
//   lora_bake --base B.gguf --lora L.gguf --out O.gguf [--only SUBSTR]   # O must be a pre-made copy of B
#define _FILE_OFFSET_BITS 64
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

static inline float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, o;
    if (e == 0) {
        if (m == 0) { o = s << 31; }
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; o = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 31) { o = (s << 31) | 0x7f800000u | (m << 13); }
    else { o = (s << 31) | ((e - 15 + 127) << 23) | (m << 13); }
    float f; std::memcpy(&f, &o, 4); return f;
}

static std::vector<uint8_t> read_at(FILE * f, size_t off, size_t n) {
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) { perror("fseeko"); std::exit(1); }
    if (fread(b.data(), 1, n, f) != n) { fprintf(stderr, "short read at %zu (%zu)\n", off, n); std::exit(1); }
    return b;
}

struct Args { std::string base, lora, out, only; bool test = false; };

int main(int argc, char ** argv) {
    Args A;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", s.c_str()); std::exit(2); } return argv[++i]; };
        if (s == "--base") A.base = next();
        else if (s == "--lora") A.lora = next();
        else if (s == "--out") A.out = next();
        else if (s == "--only") A.only = next();
        else if (s == "--test") A.test = true;
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (A.base.empty() || A.lora.empty() || (!A.test && A.out.empty())) {
        fprintf(stderr, "usage: lora_bake --base B --lora L (--test | --out O) [--only SUBSTR]\n");
        return 2;
    }

    gguf_init_params np = { true, nullptr };
    gguf_context * gb = gguf_init_from_file(A.base.c_str(), np);
    gguf_context * gl = gguf_init_from_file(A.lora.c_str(), np);
    if (!gb || !gl) { fprintf(stderr, "failed to open gguf\n"); return 1; }

    const size_t data_off = gguf_get_data_offset(gb);
    const int64_t nb = gguf_get_n_tensors(gb);

    // lora id by name
    std::unordered_map<std::string, int64_t> lid;
    for (int64_t i = 0; i < gguf_get_n_tensors(gl); ++i) lid[gguf_get_tensor_name(gl, i)] = i;

    FILE * fb = fopen(A.base.c_str(), "rb");
    FILE * fl = fopen(A.lora.c_str(), "rb");
    const size_t lora_data_off = gguf_get_data_offset(gl);

    // write mode: open the output copy r+
    FILE * fo = nullptr;
    if (!A.test) { fo = fopen(A.out.c_str(), "r+b"); if (!fo) { perror("open out"); return 1; } }

    int n_edit = 0;
    double g_dnorm = 0, g_surv = 0, g_qn = 0, g_wn = 0;  // aggregated over edited tensors
    for (int64_t bi = 0; bi < nb; ++bi) {
        const char * name = gguf_get_tensor_name(gb, bi);
        if (!A.only.empty() && std::string(name).find(A.only) == std::string::npos) continue;
        std::string stem = name, la = stem + ".lora_a", lb = stem + ".lora_b";
        auto ia = lid.find(la), ib = lid.find(lb);
        if (ia == lid.end() || ib == lid.end()) continue;

        const ggml_type T = gguf_get_tensor_type(gb, bi);
        const ggml_type_traits * tr = ggml_get_type_traits(T);
        if (!tr || !tr->to_float) { fprintf(stderr, "%s: no dequant for type %d\n", name, (int) T); return 1; }
        const int64_t * ne = gguf_get_tensor_ne(gb, bi);
        const int64_t n0 = ne[0], n1 = ne[1], E = ne[2] > 0 ? ne[2] : 1;
        const size_t tbytes = gguf_get_tensor_size(gb, bi);
        const size_t toff = gguf_get_tensor_offset(gb, bi);
        const int64_t slice_el = n0 * n1;
        const size_t slice_b = (size_t) (slice_el / tr->blck_size) * tr->type_size;
        if ((int64_t) slice_b * E != (int64_t) tbytes) {
            fprintf(stderr, "%s: size mismatch slice %zu * E %lld != %zu\n", name, slice_b, (long long) E, tbytes); return 1;
        }

        // lora a/b (F16) sizes
        const int64_t * nea = gguf_get_tensor_ne(gl, ia->second);
        const int64_t * neb = gguf_get_tensor_ne(gl, ib->second);
        const int64_t a_len = nea[0], b_len = neb[1];
        if (a_len != n0 || b_len != n1) { fprintf(stderr, "%s: dim mismatch a %lld/%lld b %lld/%lld\n", name, (long long) a_len, (long long) n0, (long long) b_len, (long long) n1); return 1; }
        std::vector<uint8_t> ab = read_at(fl, lora_data_off + gguf_get_tensor_offset(gl, ia->second), gguf_get_tensor_size(gl, ia->second));
        std::vector<uint8_t> bb = read_at(fl, lora_data_off + gguf_get_tensor_offset(gl, ib->second), gguf_get_tensor_size(gl, ib->second));
        std::vector<float> Av((size_t) E * a_len), Bv((size_t) E * b_len);
        for (size_t k = 0; k < Av.size(); ++k) { uint16_t h; std::memcpy(&h, &ab[k * 2], 2); Av[k] = h2f(h); }
        for (size_t k = 0; k < Bv.size(); ++k) { uint16_t h; std::memcpy(&h, &bb[k * 2], 2); Bv[k] = h2f(h); }

        std::vector<uint8_t> raw = read_at(fb, data_off + toff, tbytes);
        std::vector<uint8_t> outb(A.test ? 0 : tbytes);
        std::vector<float> w(slice_el), wt(slice_el), wq(slice_el);

        double dnorm = 0, surv = 0, qn = 0, wn = 0;
        for (int64_t e = 0; e < E; ++e) {
            tr->to_float(raw.data() + (size_t) e * slice_b, w.data(), slice_el);
            const float * ae = Av.data() + (size_t) e * a_len;
            const float * be = Bv.data() + (size_t) e * b_len;
            for (int64_t j = 0; j < n1; ++j) {
                const float bj = be[j];
                const int64_t row = j * n0;
                for (int64_t i = 0; i < n0; ++i) wt[row + i] = w[row + i] + bj * ae[i];
            }
            uint8_t * dst = A.test ? nullptr : outb.data() + (size_t) e * slice_b;
            if (!A.test) {
                ggml_quantize_init(T);
                size_t got = ggml_quantize_chunk(T, wt.data(), dst, 0, n1, n0, nullptr);
                if (got != slice_b) { fprintf(stderr, "%s: quantize returned %zu != %zu\n", name, got, slice_b); return 1; }
                tr->to_float(dst, wq.data(), slice_el);
            } else {
                std::vector<uint8_t> tmp(slice_b);
                ggml_quantize_init(T);
                ggml_quantize_chunk(T, wt.data(), tmp.data(), 0, n1, n0, nullptr);
                tr->to_float(tmp.data(), wq.data(), slice_el);
            }
            for (int64_t k = 0; k < slice_el; ++k) {
                const double d = wt[k] - w[k], s = wq[k] - w[k], q = wq[k] - wt[k];
                dnorm += d * d; surv += s * s; qn += q * q; wn += (double) w[k] * w[k];
            }
        }
        n_edit++;
        const double rel = std::sqrt(qn / wn);
        const double ret = dnorm > 0 ? std::sqrt(surv / dnorm) : 0.0;
        const double delta_rel = std::sqrt(dnorm / wn);   // ||delta|| / ||W||
        const double bake_rel  = std::sqrt(surv / wn);    // ||W_baked - W|| / ||W||
        printf("%-42s %-7s E=%-4lld  rel_quant_err=%.4f  delta_retained=%.4f  delta/W=%.4f  bake-drift/W=%.4f\n",
               name, ggml_type_name(T), (long long) E, rel, ret, delta_rel, bake_rel);
        g_dnorm += dnorm; g_surv += surv; g_qn += qn; g_wn += wn;

        if (!A.test) {
            if (fseeko(fo, (off_t) (data_off + toff), SEEK_SET) != 0) { perror("fseeko out"); return 1; }
            if (fwrite(outb.data(), 1, tbytes, fo) != tbytes) { perror("write out"); return 1; }
        }
    }
    printf("---\nedited %d tensors  overall rel_quant_err=%.4f  overall delta_retained=%.4f\n",
           n_edit, g_wn > 0 ? std::sqrt(g_qn / g_wn) : 0.0, g_dnorm > 0 ? std::sqrt(g_surv / g_dnorm) : 0.0);

    if (fo) fclose(fo);
    fclose(fb); fclose(fl);
    gguf_free(gb); gguf_free(gl);
    return 0;
}
