# MACE experimental extension report

Date: 2026-11-05. Protocol: `EXPERIMENT_PROTOCOL.md`, fixed before the full runs and after the main MACE results were known. **None of these analyses is preregistered.** Accuracy is the equal-crop mean with a paired, crop-stratified bootstrap (B = 2,000, seed 0) unless stated otherwise.

## 1. Summary

| Experiment | Result | Reading |
|---|---|---|
| E1 second Claude model | Haiku MACE exceeds Haiku SAGE by +18.5 IID and +11.0 held-out; the method order matches Sonnet | Holds across two Claude models, not across model families |
| E2 retrieval by evidence | Captions add +6.4 under BM25 and +6.3 under DINOv2 IID; held-out caption effects are inconclusive | The IID caption effect does not depend on BM25 selecting the captions |
| E3 fallback | -0.4 [-1.0, +0.1] IID, +0.8 [-0.5, +2.2] held-out against always-MACE | No demonstrated improvement |
| E4 collection shift | Other-collection memory lowers accuracy by 12.8 overall and by 5.0 when the disease stays in the bank, and still beats SAGE by +12.5 on those queries | Lost disease coverage explains most of the loss; the remaining loss survives a size-matched control |
| Post hoc | Sonnet held-out gain is 0.0 [-3.8, +3.8] on the seven Bugwood crops | The strongest new concern for the paper |

## 2. Research questions

1. E1: Do the MACE gains hold when a second Claude model makes the diagnosis?
2. E2: Do caption effects and retrieval effects hold across the 2 by 2 design?
3. E3: Can a label-free confidence rule avoid harmful memory?
4. E4: How much of MACE depends on memory from the query's own collection?

## 3. Configurations

Fixed for all runs: frozen release v2.1, ten crops, 1,600 hero queries (1,000 IID, 600 held-out), seed 0, K = 4 images per disease, four retrieved items, the same KB, candidates, schema and 512-pixel preprocessing. Access runs through the Claude Code CLI, which does not expose temperature.

| Package | Model | Conditions | Rows | New calls | USD |
|---|---|---|---:|---:|---:|
| E1_haiku | claude-haiku-5-5 | SAGE, MACE, labels | 4,800 | 4,776 | 4.14 |
| E2 | claude-sonnet-5-5 | DINOv2 + labels + captions | 1,600 | 1,585 | 20.76 |
| E3cal | claude-sonnet-5-5 | SAGE, MACE, MACE other-collection | 1,800 | 1,682 | 21.92 |
| E4other | claude-sonnet-5-5 | other-collection bank | 1,000 | 769 | 10.11 |
| E4family | claude-sonnet-5-5 | suspected mirrors also removed | 153 | 119 | 1.60 |
| E4matched | claude-sonnet-5-5 | standard-bank draw with the same per-disease counts as E4other | 1,000 | 510 | 7.14 |
| Smokes | Haiku, Opus, Sonnet | excluded from results | 60 | 60 | 0.82 |

The fallback rule, frozen on calibration rows before hero scoring (sha256 `dd96bb8e...`): use MACE if DINOv2 top-1 cosine is at or above a per-crop threshold set at the 10th percentile of calibration scores, and otherwise use SAGE.

Haiku was chosen on cost before any full accuracy was seen. A full Opus run would have cost about $130 at smoke rates. Haiku receives prompts identical to Sonnet's, including Sonnet-written captions, so E1 tests a second diagnosis model and not a second captioner.

## 4. Reused and new artifacts

Reused: P1 (SAGE, MACE), P8 (labels), P9 (DINOv2 labels), P5 (same and other collection, 199 rows), local 1-NN baselines, frozen captions, OpenCLIP and DINOv2 embeddings, `mace10/results.py` bootstrap and Holm code.

New: the packages above, 358 additional DINOv2 embeddings, the calibration partition, the frozen rule, and everything in this folder. Exact hashes are in `RUN_MANIFEST.json`.

Evidence classes: F1 to F4 are new analyses fixed in the protocol. F5 was fixed before its own run but after the E4 results. Everything else in `statistics/robustness_posthoc.json`, including all Bugwood, crop-level and offline bank-size results, is exploratory and was chosen after seeing the results.

## 5. Results

### E1 (Table A, Figure 1)

| Model | Method | IID | Held-out |
|---|---|---|---|
| Sonnet 5.5 | SAGE | 42.8 [39.8, 46.0] | 50.7 [46.8, 54.3] |
| | Labels | 55.8 [52.9, 58.8] | 54.8 [51.0, 58.7] |
| | MACE | 62.2 [59.3, 65.0] | 57.0 [53.2, 60.7] |
| Haiku 5.5 | SAGE | 37.0 [34.0, 39.9] | 42.5 [38.8, 46.2] |
| | Labels | 49.1 [46.2, 52.3] | 50.3 [46.5, 54.0] |
| | MACE | 55.5 [52.5, 58.5] | 53.5 [49.8, 57.2] |

Family F1 (Haiku, Holm):

| Contrast | Split | Estimate | Holm p |
|---|---|---|---:|
| MACE minus SAGE | IID | +18.5 [+15.7, +21.4] | 0.002 |
| MACE minus SAGE | Held-out | +11.0 [+7.3, +14.8] | 0.002 |
| MACE minus labels | IID | +6.4 [+3.9, +8.8] | 0.002 |
| MACE minus labels | Held-out | +3.2 [-0.5, +6.8] | 0.082 |

Method ranking did not change on either split. Haiku MACE fixed 22.2% and broke 3.7% of SAGE answers IID, and 18.5% and 7.5% held-out. Sonnet's rates are 22.6% and 3.2% IID, and 12.3% and 6.0% held-out. MACE beat SAGE on 10 of 10 crops IID for both models, and held-out on 9 of 10 for Haiku against 5 of 10 for Sonnet (one tied, four lower).

The held-out gain was larger for Haiku than for Sonnet by 4.7 points [0.5, 8.8] (uncorrected). A weaker Haiku SAGE baseline is one possible reason. The IID difference was 0.9 points [-2.4, +4.0] in the other direction.

### E2 (Table B, Figure 2)

| Retrieval | Evidence | IID | Held-out |
|---|---|---|---|
| BM25 | Labels | 55.8 [52.9, 58.8] | 54.8 [51.0, 58.7] |
| BM25 | Labels + captions | 62.2 [59.3, 65.0] | 57.0 [53.2, 60.7] |
| DINOv2 | Labels | 55.2 [52.2, 58.2] | 55.8 [51.8, 59.7] |
| DINOv2 | Labels + captions (new) | 61.5 [58.6, 64.3] | 56.3 [52.5, 60.0] |

Family F2 (Holm over eight tests):

| Contrast | IID | Holm p | Held-out | Holm p |
|---|---|---:|---|---:|
| Captions under BM25 | +6.4 [+4.1, +8.9] | 0.004 | +2.2 [-0.8, +5.2] | 1.0 |
| Captions under DINOv2 | +6.3 [+3.8, +8.7] | 0.004 | +0.5 [-2.7, +3.5] | 1.0 |
| DINOv2 minus BM25 with captions | -0.7 [-3.1, +1.8] | 1.0 | -0.7 [-3.5, +2.2] | 1.0 |
| Interaction | -0.1 [-3.0, +2.7] | 1.0 | -1.7 [-5.7, +2.0] | 1.0 |

Captions helped on 8 of 10 crops IID under each retriever. Held-out, they helped on 7 of 10 crops under BM25 and on 3 of 10 under DINOv2. The absence of a detected interaction does not show that retrieval and evidence add independently, and the intervals allow an interaction of several points.

### E3 (Table C, Figure 3)

Sixty candidate rules were tested on calibration rows, and four beat always-MACE. The best one gained +0.42 points over always-MACE (51.25 against 50.83), a margin that selection noise alone could produce.

| Method | IID | Held-out |
|---|---|---|
| SAGE | 42.8 | 50.7 |
| MACE | 62.2 [59.3, 65.0] | 57.0 [53.2, 60.7] |
| Fallback rule | 61.8 [58.9, 64.6] | 57.8 [54.2, 61.5] |
| Labeled oracle (bound only) | 63.4 [60.5, 66.2] | 59.3 [55.5, 63.2] |

Family F3: adaptive minus MACE is -0.4 [-1.0, +0.1] IID and +0.8 [-0.5, +2.2] held-out, Holm p = 0.31 for both. The rule fell back on 4.1% of IID and 19.2% of held-out queries. The rule avoided 2 of 32 IID harms and 11 of 36 held-out harms, and lost 6 correct MACE answers on each split. Even an oracle that uses ground truth adds only +1.2 and +2.3 points, so this form of fallback has little room to help.

Per-crop effects and calibration sensitivity are in `processed_results/hero_fallback.csv` and `tables/fallback_calibration_grid.csv`.

### E4 (Table D, Figure 4)

The true disease stays in the other-collection bank for 554 of 1,000 IID queries, from 19% on Corn to 100% on Mango.

| Queries | SAGE | Standard bank | Other-collection | Mirrors removed | DINOv2 1-NN, other |
|---|---:|---:|---:|---:|---:|
| All (1,000) | 42.8 | 62.2 | 49.4 | 47.3 | 24.8 |
| Covered (554) | 43.2 | 60.7 | 55.7 | 50.4 | 40.9 |
| Not covered (446) | 40.6 | 58.5 | 37.3 | 37.5 | 0.0 |

Family F4 (Holm):

| Contrast | Estimate | Holm p |
|---|---|---:|
| Other minus standard, all | -12.8 [-15.3, -10.3] | 0.002 |
| Other minus standard, covered | -5.0 [-8.6, -1.6] | 0.005 |
| Other minus SAGE, covered | +12.5 [+8.6, +16.4] | 0.002 |

Secondary: other minus SAGE is +6.6 [+4.2, +9.2] on all queries and -3.3 [-6.8, 0.0] on uncovered queries. The P5 result reproduces at +6.7 [+0.6, +12.8]. The not-covered stratum has nine crops because Mango has no such query.

**Size-matched control (family F5).** The user approved this run after seeing the E4 results. Each query got a random draw from its standard bank with the same per-disease counts as its other-collection bank, so size, disease mix and coverage match. Same-collection images make up 21.4% of matched banks and 34.2% of retrieved items, against 0% for other-collection banks.

| Contrast | Estimate | Holm p |
|---|---|---:|
| Matched minus other, covered | +5.5 [+1.9, +9.1] | 0.008 |
| Matched minus other, all | +3.0 [+0.9, +5.0] | 0.008 |
| Standard minus matched, covered | -0.5 [-2.8, +2.0] | 0.70 |

Covered accuracy is 61.2 for matched banks, 60.7 for standard banks and 55.7 for other-collection banks. Retrieval hit rate on covered queries is 83.7, 82.0 and 73.3. With crops as the unit, matched minus other on covered queries is +5.5 [+1.2, +9.8], and it is positive on eight of ten crops.

By the reading fixed in the protocol, collection dependence survives size matching. The larger standard bank showed no detectable benefit once the disease was present. Of 1,000 matched prompts, 490 repeated earlier rows, many because the top four items matched the standard bank's, which narrows the standard-minus-matched interval.

Per-crop results and the shortcut audit are in `FAILURE_ANALYSIS.md`.

### Post hoc concentration of the held-out gain (exploratory)

Sonnet's held-out gain over SAGE is 0.0 [-3.8, +3.8] on the seven crops that hold out Bugwood and +21.1 [+13.9, +27.8] on the other three. The same seven crops gain +18.7 IID. Haiku gains +6.4 [+1.7, +11.0] on those seven. With crops as the unit, the Sonnet held-out interval becomes [-3.2, +15.9].

## 6. Cost and latency

| Run | Mean USD per call | Median s | Mean input / output tokens |
|---|---:|---:|---|
| Sonnet SAGE (P1) | 0.0115 | 4.7 | 4,301 / 292 |
| Sonnet MACE (P1) | 0.0128 | 4.7 | 4,974 / 283 |
| Sonnet DINOv2 + captions (E2) | 0.0131 | 3.1 | 5,131 / 283 |
| Haiku SAGE | 0.00084 | 4.4 | 4,555 / 772 |
| Haiku MACE | 0.00092 | 4.6 | 5,153 / 808 |
| Haiku labels | 0.00084 | 4.8 | 4,372 / 799 |

All figures are list-price estimates for one diagnosis call and exclude the Sonnet caption ($0.0043 mean). P1 figures use the 1,440 uncached calls per condition, so they differ slightly from the paper's $0.0014 per-query difference. Latencies come from different concurrency levels and should not be compared across runs. Fallback makes no extra model call and needs one local DINOv2 embedding, whose time was not measured.

Total: $66.48 against a $71.80 forecast, with ledger $723.83. Forecast by package: E1 4.2, E2 20, E3cal 24, E4other 10, E4family 2, E4matched 11.60. Actual E4matched was $7.14, because 490 prompts repeated earlier rows.

## 7. Negative and inconclusive findings

- Fallback did not improve accuracy, and the selected signal barely separates harmful memory.
- Caption effects on held-out images are inconclusive under both retrievers.
- MACE minus labels is inconclusive held-out for both models.
- The Sonnet held-out gain is absent on the seven Bugwood crops.
- Accuracy falls when same-collection images leave the bank, even with the disease kept and bank size matched.
- Memory that lacks the true disease tends to lower accuracy below SAGE.

## 8. Limitations

- Both models are Claude models, and both use Sonnet captions. Each tier uses the CLI default effort, which may differ.
- Each new cell is a single run, and the CLI does not fix sampling. Repeat-run variance of new cells is unmeasured.
- Query-level intervals treat the ten crops as fixed. Crop and collection are confounded, and seven crops share one held-out collection.
- Calibration uses IID images, with an other-collection bank standing in for shift. The calibration set contains no held-out collection image.
- The size-matched control is one random draw per query, and matched and other-collection banks come from different image pools. It shows that collection composition matters, not whether the cause is a shortcut or a legitimate visual match.
- Family mirrors are suspected from embeddings, not proven.
- No plant pathologist reviewed labels, flagged queries or explanations.

## 9. Adversarial review

**1. Does MACE consistently improve over SAGE?**
Evidence: IID, yes for both models on all ten crops. Held-out, the crop-mean gain is positive (+6.3 and +11.0), but Sonnet gains on only 5 of 10 crops, and its crop-level interval includes zero.
Interpretation: consistent IID, not consistent under collection shift.

**2. Is the benefit mostly driven by retrieved labels?**
Evidence: labels alone recover 13.0 of 19.4 IID points and 4.2 of 6.3 held-out points for Sonnet, and 12.1 of 18.5 and 7.8 of 11.0 for Haiku. Existing controls agree: masked labels stay near SAGE, and shuffled labels fall below it.
Interpretation: yes, about two thirds of the gain.

**3. Is caption text independently useful?**
Evidence: +6.4 IID under BM25, +6.3 IID under DINOv2 selection, and +6.4 IID with Haiku. Held-out effects are +2.2, +0.5 and +3.2, all with intervals that include zero.
Interpretation: useful IID, and the effect survives selection that never reads caption text. Under collection shift the data cannot say.

**4. Does the benefit generalize to another Claude model?**
Evidence: same method order, similar IID gains, and a larger held-out gain for Haiku.
Interpretation: it holds for a second diagnosis model from the same family, reading the same Sonnet captions. Nothing here tests unrelated families or a different captioner.

**5. Does cross-collection accuracy remain meaningful?**
Evidence: other-collection memory beats SAGE by +6.6 overall and +12.5 when the disease stays in the bank, but loses -12.8 and -5.0 against standard memory. A size-matched bank recovers the -5.0 (+5.5 [+1.9, +9.1]).
Interpretation: yes, when the bank covers the disease. Lost coverage explains most of the drop, and the rest reflects collection composition and not bank size.

**6. Can retrieval confidence predict harmful memory?**
Evidence: rank AUC for harm is at most 0.65.
Interpretation: only weakly, and not well enough to act on.

**7. Does adaptive fallback genuinely improve accuracy?**
Evidence: no contrast differs from zero, and the oracle bound is small. Held-out gains on Apple, Grape and Wheat are offset by losses on Rice and Mango.
Interpretation: no demonstrated improvement. The intervals do not show equivalence either.

**8. Are there remaining leakage or shortcut concerns?**
Evidence: main contrasts survive near-copy removal. Same-collection memory helps when the disease and bank size are held fixed. Removing suspected mirrors lowers accuracy further. Three held-out collections show no embedding shift, and those three crops account for the held-out gain.
Interpretation: yes. Near-copy leakage looks unlikely as the main driver, but collection dependence is confirmed at about 5 points on covered queries. The cause remains unidentified: shared background or camera would be a shortcut, while closer disease appearance within a collection would not. Some held-out collections may not test much shift. Mirror overlap remains a possibility, not a finding.

**9. Are reported statistical claims defensible?**
Evidence: Holm-corrected IID claims hold under query-level and crop-level inference. The paper's held-out gain holds at query level only.
Interpretation: defensible if the held-out claim is limited to these crops and queries. The extension families are new analyses chosen after the main results. The 3-point reading rule is post hoc. Bugwood subgroup results should stay labeled exploratory.

**10. Does the evidence justify workshop-level contributions?**
Evidence: a cleaned ten-crop benchmark, a strong IID effect that holds across two models, a completed factorial, and clear negative results on fallback and collection dependence.
Interpretation: yes, for a workshop paper that presents MACE as an IID gain with a measured, uneven response to collection shift. A claim of robustness to collection shift would not survive review.

No expert biological validation was completed, and no claim in this report depends on one.

## 10. Reproduction

From `code/agmem-bench`, with `PY=../sage-seqground/.venv/bin/python`:

```sh
$PY -m experiments.mace_extension.scripts.run identity
bash experiments/mace_extension/scripts/queue.sh          # returns cached rows at no charge if .cache/llm is present
$PY -m experiments.mace_extension.scripts.analyze select  # rewrites the rule: same result only with the same rows
$PY -m experiments.mace_extension.scripts.run E4matched
$PY -m experiments.mace_extension.scripts.analyze all
$PY -m experiments.mace_extension.scripts.robustness
$PY -m experiments.mace_extension.scripts.build_outputs
$PY -m experiments.mace_extension.scripts.manifest
$PY -m pytest -q tests
```
