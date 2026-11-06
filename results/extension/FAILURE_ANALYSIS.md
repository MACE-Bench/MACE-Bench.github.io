# MACE failure analysis

All numbers come from saved rows through `scripts/analyze.py` and `scripts/robustness.py`. No new model call was made. Everything in this file is exploratory unless a Holm family is named. Accuracy is the percent correct, and gains are in points.

## Run failures

No job failed. All five final packages reached their planned row counts with no `failures.jsonl` file. E1_haiku returned no UNKNOWN prediction. E3cal returned one, which is scored as wrong. Smoke rows (60 in total) are excluded from every result.

During analysis, `ledger()` in `analyze.py` assumed a `failed` column that the ledger does not have. The fix changed only that filter and touched no result. The first version of `tex()` in `build_outputs.py` double-escaped percent signs. The `tex()` fix also changed no number.

## Where the held-out gain comes from

MACE turned 74 SAGE errors into correct answers and 36 SAGE successes into errors on the 600 held-out queries, for a net of +38 queries (+6.3 points).

| Group | Crops | Sonnet MACE minus SAGE | Haiku MACE minus SAGE | Same crops, Sonnet IID |
|---|---|---:|---:|---:|
| All | 10 | +6.3 [+3.2, +9.7] | +11.0 [+7.3, +14.8] | +19.4 |
| Held-out collection is Bugwood | 7 | 0.0 [-3.8, +3.8] | +6.4 [+1.7, +11.0] | +18.7 |
| Other held-out collections | 3 | +21.1 [+13.9, +27.8] | +21.7 [+15.0, +28.9] | +21.0 |
| Without Mango and Rice | 8 | +0.6 [-2.9, +4.0] | +7.9 [+3.3, +12.5] | +17.9 |

Mango and Rice supply 36 of the 74 corrections and 1 of the 36 harms, so they account for 35 of the net 38 queries. The groups were formed after seeing the results. Each crop has one held-out collection, so this table cannot separate a Bugwood effect from a crop effect.

Treating each crop as one observation widens the interval. The Sonnet held-out gain becomes +6.3 [-3.2, +15.9] (sign-flip p = 0.22). The Haiku gain stays above zero at +11.0 [+4.2, +17.8] (p = 0.008). The IID gains stay well above zero for both models.

## Apple, Grape, Potato and Wheat

Apple, Grape, Potato and Wheat lost accuracy on held-out images under Sonnet. All four hold out Bugwood.

| Crop | SAGE to MACE | Hit IID to held-out | Embedding gap | Label JS (bits) | Harms / rescues | Errors with miss / with hit | Fallback rate, accuracy |
|---|---|---|---:|---:|---|---|---|
| Apple | 66.7 to 61.7 | 85.0 to 66.7 | 0.30 | 0.17 | 5 / 2 | 10 / 13 | 50.0%, 66.7 |
| Grape | 70.0 to 66.7 | 90.0 to 68.3 | 0.28 | 0.07 | 5 / 3 | 12 / 8 | 45.0%, 71.7 |
| Potato | 68.3 to 61.7 | 100.0 to 86.7 | 0.20 | 0.02 | 7 / 3 | 8 / 15 | 10.0%, 61.7 |
| Wheat | 63.3 to 61.7 | 93.0 to 86.7 | 0.21 | 0.01 | 2 / 1 | 6 / 17 | 46.7%, 65.0 |

Evidence: Apple and Grape lost about 20 points of hit rate, and misses make up about half of their errors. Potato and Wheat kept a hit rate of 86.7, yet most errors occur with the true disease among the retrieved items. Apple alone shows a sizable label shift. All four have large embedding gaps, and SAGE is already above 63% on all four.

Interpretation: Apple and Grape look like a mix of source shift and retrieval failure. Potato and Wheat do not look like retrieval failure, because the right label was usually present and the model still chose wrongly. A high SAGE baseline leaves less room for memory, which may explain part of all four losses. No biological cause is inferred.

Haiku gains on three of these four crops (all but Grape), so the loss is not a fixed property of these images. Part of the reason is a weaker Haiku SAGE baseline, which leaves more errors to fix.

## Error types on held-out queries (Sonnet MACE, 258 errors)

| Category | Count |
|---|---:|
| True disease not retrieved, answer followed a retrieved label | 82 |
| True disease not retrieved, answer elsewhere | 39 |
| True disease retrieved, answer followed another retrieved label | 82 |
| True disease retrieved, answer elsewhere | 55 |

Fallback on retrieval confidence can reach only part of this. More than half of the errors (137 of 258) occur on retrieval hits, where no retrieval-failure signal applies. Per-crop counts are in `tables/failure_categories_heldout.csv`.

## Harmful memory and retrieval confidence

Memory hurt on retrieval misses: SAGE beat MACE 23.1 to 16.0 IID and 33.1 to 24.4 held-out. Confidence signals separate these cases weakly, with rank AUC between 0.42 and 0.65 for harm. BM25 top-1 did best (0.65 IID, 0.62 held-out). The selected DINOv2 top-1 signal did not separate harm IID (0.48) and only barely did on held-out queries (0.53). These AUCs were computed after scoring, so they cannot be used to swap the rule.

## Fallback rate shift

The rule falls back on 4.1% of IID and 19.2% of held-out queries, against 10% on calibration. The thresholds pool two calibration banks, which fall back at 3.7% (mixed) and 16.3% (other-collection), and those rates match the hero rates closely. The rule therefore behaves as designed. Similarity to the bank falls under collection shift, which raises the fallback rate.

## Haiku on Potato

Haiku SAGE reached 35.0% IID and 21.7% held-out on Potato, against 66.0% and 68.3% for Sonnet. The rows parse correctly. Haiku over-predicts Alternaria (36 of 100 IID answers against 25 true) and rarely predicts Bacteria Wilt And Soft Rot. Without Potato, the Haiku held-out gain stays at +11.1 [+6.9, +15.4] and the IID gain falls to +15.8 [+12.8, +18.8].

## Shortcut audit

Confirmed by intervention (family F4, Holm): removing same-collection images from the bank lowers IID accuracy even when the true disease stays in the bank, by -5.0 points [-8.6, -1.6]. A size-matched bank recovers that loss, +5.5 [+1.9, +9.1] (family F5), so bank size does not explain it.

Possible confounding, not confirmed:

- BM25 retrieves same-collection items more often than the bank contains them (57.3% against 39.4%). When the top item shares the query's collection, MACE scores 66.6 against 54.5, while SAGE scores 44.1 against 40.5. The comparison shows association only.
- Removing suspected mirrors lowered accuracy further, by -2.1 [-3.0, -1.2] on all IID queries. Rows that kept the disease in the bank lost -3.5 [-6.2, -1.0]. Only 153 rows in six crops changed, and 73 of those are Cucumber. The bank also shrank, and the mirrors themselves are suspected from embeddings, not proven.
- Mango held-out images (LeafNet) find a Mango_Leaf nearest neighbor for 40 of 60 queries, though Mango_Leaf is 6 of 20 bank images in the run inspected. The Mango embedding gap is 0.01. Whether LeafNet contains Mango_Leaf images is unknown. The near-copy screen found no same-label pair at OpenCLIP 0.90 or above, but the release removes those pairs by design, so this result says nothing about pairs just below 0.90.
- Mango, Rice and Tomato have embedding gaps near zero (0.01, 0.02, -0.06), so their held-out collections do not look shifted relative to the bank. Spearman correlation between the gap and held-out gain is -0.75 over ten crops, uncorrected.

## Bank size in E4

Other-collection banks hold 33.2 images on average against 49.2 for the standard bank. The paid size-matched control (family F5) separates size from collection. Matched banks have the same per-disease counts as other-collection banks and keep some same-collection images.

| Covered queries (554) | Accuracy | Hit rate |
|---|---:|---:|
| Standard bank | 60.7 | 82.0 |
| Size-matched bank | 61.2 | 83.7 |
| Other-collection bank | 55.7 | 73.3 |

Matched minus other is +5.5 [+1.9, +9.1], Holm p = 0.008. Standard minus matched is -0.5 [-2.8, +2.0]. Bank size therefore does not explain the -5.0 contrast. An earlier offline check, which shrank banks at random without keeping the disease mix, suggested a possible size effect, and this control supersedes that reading.

By crop, matched exceeds other on eight of ten crops, led by Corn (+15.8), Grape (+10.4) and Apple (+7.8). Cucumber (-5.3) and Rice (-1.6) go the other way. Corn has only 19 covered queries.

The cause remains open. Same-collection images may help through shared background or camera, which would be a shortcut, or through closer disease appearance, which would not. This design cannot separate the two.

## Near-copy sensitivity

| Removal | Queries left IID / held-out | MACE minus SAGE | MACE minus labels |
|---|---|---|---|
| None | 1,000 / 600 | +19.4, +6.3 | +6.4, +2.2 |
| Different-label bank image at OpenCLIP 0.90 or above | 972 / 592 | +19.9, +6.4 | +6.9, +2.4 |
| The same at 0.85 or above | 695 / 508 | +19.6, +8.0 | +6.0, +3.6 |

The main contrasts do not depend on these queries. The 28 flagged IID queries are hard for every method (MACE 21.4, SAGE 21.4, DINOv2 1-NN 17.9), which fits ambiguous or conflicting labels better than leakage. Nobody has reviewed those labels.
