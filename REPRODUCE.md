# Reproducing MACE

This repository holds the MACE project page, the code and every number in the paper.

| Folder | Contents |
|---|---|
| `code/agmem/` | Evidence packets, prompts, retrieval, the agent loop and the model interface |
| `code/mace10/` | Ten-crop study: locked queries, job lists, dispatcher, analysis, tables and figures |
| `code/benchmark/` | Scripts that built the benchmark from SAGE images |
| `code/tests/` | Offline tests with a fake model |
| `results/results_wide.csv` | One row per query and one column per method (1 = correct) |
| `results/results.json` | Every reported accuracy, interval and contrast |
| `results/predictions/` | Every model answer, by package (gzip JSON lines) |
| `results/captions.json.gz` | Every label-free caption |
| `results/budget_ledger.csv` | Planned and actual calls and prices |

## 1. Check the reported numbers (under a minute)

This step needs Python 3 with numpy and pandas. It does not need images or model calls.

```bash
pip install numpy pandas
python results/verify.py
```

The expected output includes these equal-crop accuracies (%):

| Method | In distribution | Held-out |
|---|---|---|
| SAGE | 42.8 | 50.7 |
| DINOv2 1-NN | 52.5 | 38.2 |
| MACE retrieval | 62.2 | 57.0 |

## 2. Rerun the analysis

`code/mace10/results.py` computes the intervals and Holm-adjusted contrasts. It reads the files in `results/` once they
are placed in `code/mace10/out/` under the package folders that `code/mace10/run.py` writes. The latency and price rows
also require the private call ledger, so only accuracy results can be rebuilt from this repository.

## 3. Rerun the model calls

This step cannot be completed from this repository yet, because the benchmark images are not public. It requires:

- The frozen benchmark, tag `v2.1-frozen-2026-11-03`, with `AGMEM_DATA` set to its location. **Release is pending.**
- The Claude Code command-line interface, signed in. The model alias `sonnet` resolved to `claude-sonnet-5-5` for every
  recorded call.
- DINOv2 embeddings for packages P9 and the 1-NN baselines, built with `code/mace10/local_baselines.py` and
  `code/mace10/embed_dinov2.py`.
- About USD 185 at list prices for all packages.

```bash
cd code
pip install -r requirements.txt
python -m pytest tests/test_mace10_run.py -q                # offline, no model calls
python -m mace10.run P1 --cap-additional 400                # repeat for each package below
```

Run the packages in this order: `P7dev P1 P4labels P2 P3 P4rest P5 P6 P7 P8 P9`.

## What will differ

- The command-line interface does not fix sampling, so a rerun will not match every answer. Repeat-run variance is
  unmeasured, and each reported result is one run.
- Prices are list-price estimates reported by the interface.
- Packages P8 and P9 are post hoc. `code/mace10/AMENDMENT_2026-11-05.md` records why they were added and how they were
  analyzed. `code/mace10/LOCKED_ANALYSIS.md` records the planned analysis.

## Not in this repository

Benchmark images, raw call traces and the response cache are not included. Raw traces hold third-party knowledge-base
passages. The trace exporter is `code/mace10/export_traces.py`.

## License

Code: MIT. Results: CC BY 4.0. See [LICENSE.md](LICENSE.md).
