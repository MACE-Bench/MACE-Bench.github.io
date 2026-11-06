# MACE extension protocol

Fixed on 2026-11-05, before any full extension run. Only smoke rows existed at that point (24 Haiku, 24 Opus, 12 DINOv2-caption rows), and those rows are not used in any result. These analyses were added after the main results were known. They are **not preregistered** and the paper must not call them preregistered. Nothing here changes LOCKED_ANALYSIS.md or AMENDMENT_2026-11-05.md.

## Fixed for every experiment

Ten crops, 1,600 hero queries, frozen release v2.1, seed 0, K=4 images per disease, four retrieved items, equal-crop accuracy. Candidate lists, KB, system prompt, output schema, 512-pixel JPEG preprocessing and Sonnet captions do not change.

Statistics follow LOCKED_ANALYSIS.md: paired bootstrap over unique query images, stratified by crop, 2,000 resamples, seed 0, 95% percentile intervals, Holm within each family below. Repeated calls are not independent observations, and these intervals do not include model sampling noise. A nonsignificant result is not evidence of equivalence. An interval inside [-3, +3] is read as "no practically meaningful difference"; that 3-point margin comes from the earlier amendment and was chosen after a first look.

Budget: established limit. Stop optional work at ledger USD 822.82; hard stop USD 872.82. Cost is the CLI's list-price estimate.

## E1. Second Claude model

Model: claude-haiku-5-5, chosen after smoke tests on cost alone. Claude Opus 5.5 measured USD 0.027 per call, so a full run (about USD 130) plus the other packages would pass the USD 822.82 limit. Haiku measured USD 0.00088.
Conditions: SAGE, MACE, labels only, on all 1,600 queries. Captions stay frozen Sonnet captions, so retrieval, KB selection and prompts are byte-identical to the Sonnet rows. This tests a second **diagnosis** model, not a second captioner. The system uses each model's CLI default effort, which may differ across tiers.

Family F1 (Holm, 4 tests): Haiku MACE minus SAGE; Haiku MACE minus labels; each on IID and held-out.
Conclusions cover two Claude tiers only, not unrelated model families.

## E2. Retrieval by evidence factorial

New cell: DINOv2 + labels + captions. It uses the exact P9 top-4 ids and order, MACE's reference line format and header, and the same KB part as P9.

Family F2 (Holm, 8 tests; each on IID and held-out):
1. Caption effect under BM25: MACE minus labels. Already seen, reported as post hoc.
2. Caption effect under DINOv2: dino_mace minus dino_labels. New.
3. Retrieval effect with captions: dino_mace minus MACE. New.
4. Interaction: (dino_mace - dino_labels) - (MACE - labels). New.
Retrieval effect without captions (dino_labels minus labels) is reported from the amendment and not retested.

Caveat fixed in advance: BM25 ranks by caption text, so captions shown under BM25 were also selected for matching the query caption. DINOv2 captions were not. The factors are not fully separable in meaning.

## E3. Retrieval-confidence fallback

Rule form: use MACE if signal >= threshold, otherwise SAGE. One signal, one threshold. No trained model.

Signals, all available before diagnosis and free of ground truth:
- BM25 raw top-1, top-1 minus top-2 margin, and concentration (top-1 over sum of top-4). Raw scores are recomputed, because logged `memory_relevance` is divided by the max and is always 1 at top-1.
- BM25 label agreement: share of the top-4 labels held by the modal label.
- DINOv2 top-1 cosine, margin and label agreement.
- Cross-retriever agreement: share of BM25 top-4 labels also among DINOv2 top-4 labels.

Continuous scores are not compared across retrievers or crops: thresholds are per-crop quantiles of the calibration score, q in {0.05, ..., 0.50}. Agreement signals use global thresholds in {0.5, 0.75, 1.0} (cross-agreement {0.25, 0.5, 0.75, 1.0}).

Calibration partition: the next 60 IID images per crop in the locked order, never hero or dev, with SAGE, MACE on the mixed bank, and MACE on a bank without the query's collection, which stands in for collection shift using training-side data. No held-out-collection image is used.

Selection: maximize the mean of equal-crop calibration accuracy on the two calibration banks. The rule is written to configurations/fallback_rule.json before hero rows are scored with it, and is evaluated once. If no rule beats always-MACE on calibration, the best rule is still evaluated and reported.

The routing decision needs only the query caption and retrieval, both needed by SAGE and MACE alike, so a deployed system makes one diagnosis call per query. Hero evaluation composes saved SAGE and MACE rows. Added cost is zero calls, plus one local DINOv2 embedding if a DINOv2 signal is selected.

Family F3 (Holm, 2 tests): adaptive minus MACE, IID and held-out. Oracle fallback (MACE on a post hoc retrieval hit, otherwise SAGE) is a diagnostic upper bound only.

## E4. Cross-collection robustness

Existing facts: held-out banks contain no image from the query's collection, so held-out rows already are cross-collection memory. Mirrors of the same source may still hide under different collection names.

New runs on all 1,000 hero IID queries:
- other_source: the release's bank without the query's collection. The 199 P5 rows return from cache.
- family_disjoint: also removes suspected mirror collections (configurations/collection_families.json, set from embeddings only). No replacement images; the shrinkage is recorded.

K stays at most 4 per disease. If a disease has no eligible image it is absent from the bank; the candidate list keeps it. Every result is split into **covered** (true disease in that bank) and **not covered**.

Family F4 (Holm, 3 tests): other_source minus standard on all IID; other_source minus standard on covered IID; other_source minus SAGE on covered IID.
Secondary: P5 reconstruction; DINOv2 1-NN on the same banks; top-4 true-disease coverage per crop; label-distribution distance; failure categories for Apple, Grape, Potato and Wheat; retrieved-collection analysis.

Dedup sensitivity, offline: repeat main contrasts after removing queries with a different-label bank image at OpenCLIP cosine >= 0.90, and at >= 0.85. The official test set does not change.

Shortcut language: correlation with collection supports "possible confounding" only. "Confirmed shortcut" needs an intervention that changes the source and leaves the disease fixed, which here means the covered-stratum contrast. No biological causation is inferred.

## Evidence classes in the report

- Existing primary: LOCKED_ANALYSIS.md.
- Existing post hoc: AMENDMENT_2026-11-05.md.
- New analyses: F1 to F4, fixed here, after the main results were known.
- Exploratory: everything else, including per-crop readings.

## Additions after the full runs

The text above is unchanged. The following items came after the runs finished, and none changes a family, rule or contrast:

- `scripts/robustness.py` adds crop-level intervals, Bugwood and Mango-Rice groupings, nearest-bank checks, coverage under the family bank, an offline bank-size check and calibration isolation checks. All of these were chosen after seeing results and are exploratory.
- `analyze.py` `ledger()` no longer assumes a `failed` column. No result changed.
- `configurations/fallback_rule.sha256` records the rule hash before hero scoring. The rule file was not edited afterward.

## E4 size-matched control

Added 2026-11-05 on the user's approval, after all E4 results were known and before any E4matched call. It is a new analysis and is not preregistered.

Purpose: E4 banks are smaller than standard banks, so the covered-stratum contrast mixes collection removal with bank size and disease mix.

Bank: for each of the 1,000 hero IID queries, draw from that query's standard bank a random subset with the same per-disease counts as its E4other bank, seeded by image id and disease, without regard to collection. An offline check found all 1,000 banks match those counts, with mean size 33.16. Same-collection images make up 21.4% of matched banks and 34.2% of their BM25 top-4, against 0% for E4other. E4matched and E4other therefore share size, disease mix and coverage, and differ in which images they hold. One draw per query. Expected: about 884 new Sonnet calls, about USD 11.60.

Family F5 (Holm, 3 tests):
1. Matched minus other-collection, covered queries. Tests collection composition at fixed size and disease mix.
2. Matched minus other-collection, all queries.
3. Standard minus matched, covered queries. Estimates the effect of the larger bank.

Reading, fixed now: if test 1 excludes zero, collection dependence survives size matching. If it includes zero, no such dependence was detected, which does not show absence. Because matched banks keep only part of the same-collection share, test 1 understates the effect of the full standard bank. Matched and other banks also come from different image pools, since E4other redraws images the standard bank does not hold.
