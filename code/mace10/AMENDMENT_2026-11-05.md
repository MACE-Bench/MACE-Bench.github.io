# Amendment to the locked plan, 2026-11-05

Added by the user's decision after the 400-query labels-only result was seen (labels minus MACE: -3.0 [-8.5, +2.5] IID,
+0.5 [-4.5, +5.5] held-out). These arms are therefore **post hoc**, and the paper must say so. They were fixed here before
either arm produced any data. The locked analysis does not change: the 400-query contrast stays the locked result, and the
Holm family of LOCKED_ANALYSIS.md stays as it was.

## New packages (run after P7, inside the existing USD 400 cap)

| Package | What | Calls | About |
|---|---|---|---|
| P8 | Labels only on the 1,200 hero queries outside the ablation panel; with P4 labels this covers all 1,600 | 1,200 | $14 |
| P9 | DINOv2 labels: the four labels of the DINOv2 top-4 of the same seed 0, K=4 bank, on all 1,600 hero queries | 1,600 | $19 |

P9 differs from labels only in the four labels and ids shown, and in nothing else: same query, KB text (claims selected for the
MACE packet of that query), candidates, header and schema. Order: cosine descending, ties by image id. Embeddings: the
existing out/dinov2_embeddings.npz, normalized.

## Analysis (fixed now)

Equal-crop accuracy, paired bootstrap as in LOCKED_ANALYSIS.md, IID and held-out separately, all 1,600 hero queries.
Family A, Holm across these four:
- A1 labels (all 1,600) minus MACE retrieval
- A2 DINOv2 labels minus labels (BM25 over captions)

Reading rule, margin 3 points (chosen after the first look, so disclosed as such):
- A1 interval entirely below 0: caption text adds accuracy beyond labels.
- A1 interval inside [-3, +3]: no practically meaningful difference.
- A2 interval entirely below 0: caption retrieval beats image retrieval.
- A2 interval inside [-3, +3]: captions are not needed for retrieval on this bank.
- Anything else: inconclusive, reported as such.

Secondary: DINOv2 labels minus DINOv2 1-NN; true-disease hit rate of DINOv2 top-4 against BM25 top-4.
