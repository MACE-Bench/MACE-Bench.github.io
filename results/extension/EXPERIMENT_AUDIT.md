# MACE experiment audit

Audit date: 2026-11-05. Paper: Overleaf commit ac74e06. Code: code/agmem-bench at e1df176 plus the two small changes listed at the end. Numbers below were recomputed from raw rows in this audit unless marked otherwise.

## Verified existing experiments

Every reported number was recomputed from `mace10/out/<PKG>/predictions.jsonl` and `local_baselines.csv` through `mace10/results.py`. All match the paper.

| Method | IID | Held-out | Source rows |
|---|---:|---:|---|
| SAGE | 42.8 | 50.7 | P1, condition sage, 1,600 rows |
| MACE retrieval | 62.2 | 57.0 | P1, condition mace, 1,600 rows |
| Labels only | 55.8 | 54.8 | P4labels (400) plus P8 (1,200) |
| DINOv2-retrieved labels | 55.2 | 55.8 | P9, 1,600 rows |
| DINOv2 1-NN | 52.5 | 38.2 | local_baselines.csv, no model call |

All rows use `claude-sonnet-5-5` through the CLI alias `sonnet`, equal-crop accuracy, seed 0, K=4. The benchmark holds 1,000 IID and 600 held-out queries. All 1,600 images exist on disk. One P1 row returned UNKNOWN and counts as wrong. The ledger stood at USD 657.35 before this extension.

A code-path check rebuilt 400 sampled Sonnet requests through the extension runner with a zero budget: 400 of 400 returned from the content-addressed cache with the same prediction (`raw_results/identity_check.json`).

Interval discrepancy: RESULTS.md prints the held-out MACE minus SAGE interval as [3.0, 9.5], while results.json and the paper print [3.2, 9.7]. Resampling the same rows in sorted order gives [3.2, 9.3]. The bootstrap depends on row order, so this difference is Monte Carlo noise at 2,000 resamples and not an error in the paper. Interval ends carry about 0.3 points of this noise.

## Existing experiments that can be reanalyzed at no charge

- Fallback routing on hero queries: the query caption feeds SAGE and MACE alike, and both rows exist, so routing composes saved rows exactly.
- Same- versus other-collection memory on 199 paired IID queries (P5).
- Hit and miss strata, per-crop tables, retrieved-collection analysis, dedup sensitivity, and all DINOv2 and OpenCLIP nearest-neighbor controls.

## Missing experiments that need inference

| Package | Purpose | Calls | Forecast USD |
|---|---|---:|---:|
| E1_haiku | second model: SAGE, MACE, labels | 4,800 | 4.2 |
| E2 | DINOv2 retrieval with captions | 1,600 | 20 |
| E3cal | calibration partition, 600 queries, three banks and conditions | 1,800 plus captions | 24 |
| E4other | other-collection memory on all IID queries | 801 new | 10 |
| E4family | suspected mirror collections removed | 153 | 2 |

Forecasts use smoke rates and earlier ledger rates.

## Missing data or unavailable resources

- No separate validation partition exists across ten crops. The 40-item dev panel covers five crops and was spent on image-reference calibration. E3cal builds one from unused IID images.
- `ANTHROPIC_API_KEY` is unset. All calls go through the logged-in Claude Code CLI, as in the original run. Cost is the CLI's list-price estimate, not a bill.
- The CLI exposes no sampling seed, so model sampling noise is not measurable from existing rows. Repeated calls hit the cache.
- No expert biological review exists for any output.

## Reusable artifacts

| Artifact | Location |
|---|---|
| Raw model replies, keyed by request hash | `.cache/llm/` (21,000+ files) |
| Cost and latency per call | `runs/cost_ledger.jsonl` |
| Frozen benchmark, splits, memory banks, KB | `../sage-seqground/memlock/release` (v2.1) |
| Locked query panels | `mace10/locked/queries.csv` |
| Sonnet captions | `mace10/out/captions.json`, 5,601 images |
| DINOv2 embeddings | `mace10/out/dinov2_embeddings.npz` |
| OpenCLIP embeddings | `release/audit/clip_vitb32_embeddings.npz` |
| P9 DINOv2 rankings | rebuilt exactly by `mace10/amend.py` |
| Bootstrap and tables | `mace10/results.py`, `mace10/figures.py` |

## Scientific and implementation risks

1. **Disease and collection are confounded.** With the query's collection removed, the true disease remains in the bank for only 55.4% of hero IID queries (Corn 19%, Cucumber 38%, Soybean 39%, Mango 100%). Many diseases come from one collection. An unstratified other-collection accuracy would mix source shift with absent classes, so every E4 result is split by coverage.
2. **Held-out rows are already cross-collection.** No held-out bank holds an image from the query's collection. A new "cross-collection held-out" run would repeat the existing rows, so E4 runs on IID queries instead.
3. **Collections are not independent sources.** Cross-collection neighbors at OpenCLIP cosine 0.88 or above reach 24.8% of images for Cucumber_Zen and LeafNet, 18.4% for AvinashPlant and New_Plant_Di, 18.3% for LeafNet and RadyPlantDis, and 13.1% for PlantVillage and VQAPlantDise. These pairs suggest shared origin but do not prove it.
4. **Dedup leaves a cross-label gap.** Filtering happens before splitting, so near-duplicates cannot straddle a split within a class: same-class pairs at cosine 0.90 or above are removed, and the release check confirms a maximum of 0.90. Pairs with different labels are removed only at 0.95 or above. Between 0.90 and 0.95, different-label near-duplicates remain: 28 IID and 8 held-out queries have one in their bank. MACE scores 21.4% on those 28 IID queries and 63.4% on the other 972. This pattern fits mislabeled or ambiguous source images, and it penalizes MACE rather than inflating it.
5. **BM25 relevance is normalized.** Logged `memory_relevance` divides by the maximum, so top-1 equals 1 for every query. Raw scores must be recomputed for any confidence signal.
6. **Caption and retrieval factors overlap.** BM25 selects by caption text, so its captions were chosen for matching the query caption. DINOv2 captions were not. The E2 caption effect therefore differs in meaning across retrievers.
7. **Captions come from one model.** Query and memory captions are Sonnet outputs. E1 with frozen captions tests a second diagnosis model and leaves Sonnet's perception in the packet. Haiku's MACE rows benefit from Sonnet's captions, while its SAGE rows also use a Sonnet caption to rank KB claims.
8. **The 3-point margin and P8, P9 are post hoc**, as AMENDMENT_2026-11-05.md states. E2 builds on those rows, and the BM25 caption effect has already been seen.
9. **Held-out collections repeat.** Seven of ten crops hold out Bugwood, so "held-out" crop results are not independent views of source shift.
10. **Selection of ten crops** used metadata only, which is sound, but crop-level correlations have n=10 and only weak power.
11. **CLI drift.** The CLI version is 2.1.318 now; the alias resolved to the same model id in the smoke rows. Each package manifest records the version.

## Code changes made for this extension

Both changes are backward compatible; the existing 17 tests pass and the identity check above confirms unchanged cache keys.

- `agmem/llm.py`: `use_model()`, a per-thread model override. Unset, the cache key and CLI command match the earlier code.
- `mace10/conditions.py`: a `dino_mace` header. Existing variants do not change.
