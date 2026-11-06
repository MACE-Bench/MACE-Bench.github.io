# MACE

Memory-Augmented Crop-Disease Evaluation of Frozen Vision-Language Models.

**Project page:** https://mace-bench.github.io/

MACE adds a caption memory to a frozen vision-language model. Each labeled reference image is described once as text, and the four most relevant labeled captions are retrieved for each test image.

## Results

On ten crops and 1,600 queries, equal-crop accuracy (%):

| Method | In distribution | Held-out collections |
|---|---:|---:|
| SAGE | 42.8 | 50.7 |
| DINOv2 1-NN | 52.5 | 38.2 |
| MACE retrieval | 62.2 | 57.0 |

Follow-up tests refine this picture:

- A second Claude model keeps the method order, with gains of +18.5 and +11.0 points over SAGE. Both models are Claude models.
- Caption text adds +6.3 points in distribution under DINOv2 retrieval. Held-out caption effects are inconclusive.
- A retrieval-confidence fallback showed no detectable gain.
- Memory from the query's collection still helps at equal bank size and disease mix, by +5.5 points. The cause is unknown.
- In a post hoc grouping, Sonnet gains 0.0 points on the seven crops that hold out Bugwood.

No plant pathologist reviewed the labels or recorded cases.

## Check the numbers

Needs Python 3 with numpy and pandas, and no images or model calls.

```bash
pip install numpy pandas
python results/verify.py             # main results
python results/verify_extension.py   # follow-up tests
```

## Contents

| Path | Contents |
|---|---|
| `index.html`, `static/` | Project page |
| `code/` | Method, benchmark build, ten-crop study, follow-up tests and offline tests |
| `results/` | Every prediction, caption, statistic and call record for the main study |
| `results/extension/` | Follow-up predictions, statistics, tables, protocol and report |
| `REPRODUCE.md` | How to verify the numbers and rerun the study |
| `LICENSE.md` | Licenses: code MIT, results CC BY 4.0, images under their source licenses |

Benchmark images are not public yet, so model calls cannot be rerun from this repository alone. See `REPRODUCE.md`.
