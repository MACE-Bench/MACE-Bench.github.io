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
| `results/cost_ledger.jsonl.gz` | Every model call: stage, tokens, list price, latency, cache status (no prompts or replies) |
| `results/extension/` | Follow-up tests: predictions, statistics, tables, protocol, report and failure analysis |
| `code/experiments/mace_extension/` | Scripts and configurations for the follow-up tests |
| `code/mace10/out/` | DINOv2 embeddings of benchmark images, which the job lists need (vectors only, no images) |

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

To check the follow-up tests the same way, run `python results/verify_extension.py`. It recomputes 15 contrasts from
`results/extension/extension_wide.csv`, checks each against the saved statistics and prints a fresh bootstrap interval.

## 2. Rerun the analysis

`code/mace10/results.py` computes the intervals and Holm-adjusted contrasts. It reads the files in `results/` once they
are placed in `code/mace10/out/` under the package folders that `code/mace10/run.py` writes. For the latency and
price rows, decompress `results/cost_ledger.jsonl.gz` to `code/runs/cost_ledger.jsonl`.

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
AGMEM_DATA=/path/to/benchmark python -m pytest tests -q     # no model calls, but needs the benchmark metadata
python -m mace10.run P1 --cap-additional 400                # repeat for each package below
```

Run the packages in this order: `P7dev P1 P4labels P2 P3 P4rest P5 P6 P7 P8 P9`.

## 4. Rerun the follow-up tests

These tests need the same benchmark and interface as step 3, plus the response cache for the fallback signals. The
protocol is `results/extension/EXPERIMENT_PROTOCOL.md`, and it records which analyses were fixed before each run.

```bash
cd code
python -m experiments.mace_extension.scripts.run E2
python -m experiments.mace_extension.scripts.run E4other
python -m experiments.mace_extension.scripts.run E4family
python -m experiments.mace_extension.scripts.run E4matched
python -m experiments.mace_extension.scripts.run E3cal
python -m experiments.mace_extension.scripts.run E1_haiku --model claude-haiku-5-5
python -m experiments.mace_extension.scripts.analyze all
python -m experiments.mace_extension.scripts.robustness
```

The fallback rule in `code/experiments/mace_extension/configurations/fallback_rule.json` is frozen, and its hash is
recorded beside it. Do not rerun `analyze select` unless you intend to replace it. These runs cost USD 66.48 at list prices.

## What will differ

- The command-line interface does not fix sampling, so a rerun will not match every answer. Repeat-run variance is
  unmeasured, and each reported result is one run.
- Prices are list-price estimates reported by the interface.
- All follow-up tests were added after the main results and are not preregistered. Packages P8 and P9 are post hoc. `code/mace10/AMENDMENT_2026-11-05.md` records why they were added and how they were
  analyzed. `code/mace10/LOCKED_ANALYSIS.md` records the planned analysis.

## Not in this repository

Benchmark images, the paper PDF, raw call traces and the response cache are not included. Benchmark images await
release, and raw traces hold third-party knowledge-base passages. The trace exporter is `code/mace10/export_traces.py`.

## License

Code: MIT. Results: CC BY 4.0. See [LICENSE.md](LICENSE.md).
