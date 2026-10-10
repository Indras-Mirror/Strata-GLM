# Fast start + residency: why the draft block didn't pay, and how to make it

Two measured facts (2026-10-10, our box: 4090 24 GB + 90 GB DDR4):

| engine | file | decode | prefill | **load** | resident |
| --- | --- | ---: | ---: | ---: | --- |
| **ours** | ours (113.6 GB) | **16-20 t/s** | ~166 t/s | **796 s** | 69.2 GiB arena → **8341 experts (~69%)** |
| PolyStrata, `--no-draft` | ours | 5.05 t/s | 46.9 | **13.0 s** | 1413 experts in VRAM (10.9 GiB), rest on demand |
| PolyStrata, `--draft-model` | ours | 5.17 t/s (**+2%**) | 55.1 | 9.7 s | 1379 in VRAM; 70/82 drafts right (85%) |

The draft block is accurate (85% accepted) and buys **2%**. Our engine is **3-4x faster** than theirs here because
of the 69 GiB RAM arena - but it costs **796 s** before the first token.

## Why the draft doesn't pay yet - and the lever that would make it

Spec decode trades fewer forward passes for **more distinct experts**: a k-token verify runs the router on all k
positions, so it touches up to `distinct_ratio(k)` (our `CostModel`: 1.70 at k=2 up to 5.2 at k=9) times the experts of
one token. On a box where most experts are **resident**, the extra ones cost compute only, and spec wins (maya/DeepSeek
show 2x). On our box ~31% of experts are **not** resident, so the verify multiplies *fetches*, and the fetches cancel
the saved passes - hence +2%, and our s21 result.

**So the draft pays IFF residency is high.** That is the whole game, and it is actionable:

1. **Residency is currently low in VRAM.** Our ablated run got **731 slots / 5.63 GiB** because `--vram-margin 5.5`
   holds 5.5 GiB back. PolyStrata got **10.9 GiB / 1413 experts** on the same 24 GB card. Cutting our margin roughly
   **doubles the VRAM expert cache** (more hits, fewer fetches, and a cheaper verify).
2. **Then re-test the draft block** at small windows (k=1..3). Our `CostModel` will pick k=0 while fetches dominate;
   with a bigger resident fraction it should pick k>0 - and *that* is the experiment that decides whether spec helps
   us. The block is already fetched (`draft-block.gguf`, 2.79 GB) and the spec stack + `mtp_probe` are in the tree.

## Fast start: their model vs ours

PolyStrata starts in ~10 s because it **never builds an arena**: it mmaps the file, keeps the top experts in VRAM, and
reads the rest on demand, warming an `--expert-profile` (it swapped 1164/880 experts in *during* the run). Ours builds
the whole 69 GiB arena up front, then serves - fast decode, brutal startup.

**Proposed mode (fast start):**
- **Start serving immediately** off the mmap'd file + a VRAM cache (PolyStrata's path). First token in ~10 s.
- **Build the RAM arena in the background**, in ranked (profile) order, and swap in behind the live requests.
  Full-decode speed once the arena warms; no restart.
- **Optional: snapshot the warm arena** to disk on a clean stop and restore it on the next start, so the 796 s is paid
  once (this also survives the "which experts a conversation uses" re-learning).

This is additive: the current blocking-arena path stays the default until the fast-start path is proven.

## Risks / honest notes

- Pure "no arena" (PolyStrata's exact model) is a **decode regression for us** (5 vs 16-20 t/s) - so we take the
  *startup* shape, not the whole design. The background arena build is the part that keeps our speed.
- Our arena is near the RAM cap (69 GiB of 90 GB, with the 56.6 GB bad-lane caveat), so "more experts resident" is
  mostly a **VRAM** story: raise the cache, cut the margin, add the adaptive/demote path we already have
  (`--arena-adapt`, `--vram-lru`, `arena_skip_resident`).
- The `CostModel` defaults are the 2026-09-23 DS4/Qwen numbers; a GLM calibration on our box is needed before its k
  choice is trustworthy.

## Next steps

1. **Cut `--vram-margin`** and re-run the ablated decode A/B → measure the VRAM-cache gain (cheap, no code).
2. **Port the draft block into our engine** (GLM NextN forward + wire the `Controller`), test k=1..3 with the raised
   residency → the real "does spec help us" answer.
3. **Fast-start mode**: background arena build behind live serving; optional warm-arena snapshot.
