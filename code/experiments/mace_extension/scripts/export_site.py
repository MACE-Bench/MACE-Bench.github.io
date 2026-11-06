"""Copy the extension into the public site repository and patch the page. No model call.

  python -m experiments.mace_extension.scripts.export_site <site_dir>

Left out on purpose: the paper revision guide and status notes (internal), smoke rows, benchmark images, the response cache.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from .analyze import PR, RAW, ROOT, ST, TB, rows

REPO = ROOT.parent.parent
FINAL = ("E1_haiku", "E2", "E3cal", "E4other", "E4family", "E4matched")
FIG = {"fig1_model_robustness": "ext_models", "fig2_retrieval_caption": "ext_factorial", "fig3_fallback": "ext_fallback", "fig4_collection_shift": "ext_collections"}

SECTION = """
<section class="section is-tint" id="followup">
  <div class="container"><div class="res-wrap">
    <div class="has-text-centered" style="margin-bottom:1.2rem"><span class="sec-kicker">Follow-up tests</span><h2 class="title is-3">Stress tests of the main result</h2></div>
    <div class="table-container"><table class="table res is-fullwidth">
      <thead><tr><th>Test</th><th class="has-text-right">In distribution</th><th class="has-text-right">Held-out</th></tr></thead>
      <tbody>
<tr class="grp"><td colspan="3">Second diagnosis model: Claude Haiku 5.5, same Sonnet captions</td></tr>
<tr><td class="model">SAGE</td><td class="num">37.0 <small>[34.0, 39.9]</small></td><td class="num">42.5 <small>[38.8, 46.2]</small></td></tr>
<tr><td class="model">Labels only</td><td class="num">49.1 <small>[46.2, 52.3]</small></td><td class="num">50.3 <small>[46.5, 54.0]</small></td></tr>
<tr class="best"><td class="model">MACE</td><td class="num">55.5 <small>[52.5, 58.5]</small></td><td class="num">53.5 <small>[49.8, 57.2]</small></td></tr>
<tr class="grp"><td colspan="3">Retrieval and evidence, Sonnet 5.5</td></tr>
<tr><td class="model">DINOv2 retrieval with captions</td><td class="num">61.5 <small>[58.6, 64.3]</small></td><td class="num">56.3 <small>[52.5, 60.0]</small></td></tr>
<tr class="grp"><td colspan="3">Fallback to SAGE at low retrieval confidence, Sonnet 5.5</td></tr>
<tr><td class="model">Frozen fallback rule</td><td class="num">61.8 <small>[58.9, 64.6]</small></td><td class="num">57.8 <small>[54.2, 61.5]</small></td></tr>
<tr><td class="model">Labeled oracle (upper bound only)</td><td class="num">63.4 <small>[60.5, 66.2]</small></td><td class="num">59.3 <small>[55.5, 63.2]</small></td></tr>
<tr class="grp"><td colspan="3">Memory source, 554 in-distribution queries with the true disease in memory, Sonnet 5.5</td></tr>
<tr><td class="model">Standard memory</td><td class="num">60.7 <small>[56.5, 64.9]</small></td><td class="num">not run</td></tr>
<tr><td class="model">Same size and disease mix</td><td class="num">61.2 <small>[56.9, 65.3]</small></td><td class="num">not run</td></tr>
<tr><td class="model">Other collections only</td><td class="num">55.7 <small>[51.6, 59.6]</small></td><td class="num">not run</td></tr>
      </tbody></table></div>
    <p class="table-legend">Equal-crop accuracy (%) with 95% bootstrap intervals. All tests were added after the main results and are not preregistered.</p>
    <div class="disc" style="margin-top:1.4rem">
      <div class="box"><h4><span class="icon"><i class="fas fa-exchange-alt"></i></span>Second model</h4><p>Haiku keeps the method order. MACE exceeds SAGE by +18.5 points in distribution and +11.0 held-out. Both models are Claude models that read Sonnet captions, so this says nothing about other model families.</p></div>
      <div class="box"><h4><span class="icon"><i class="fas fa-th-large"></i></span>Captions as evidence</h4><p>Captions add +6.3 points in distribution when DINOv2 selects the memory, close to the +6.4 under BM25. Held-out caption effects are inconclusive under both retrievers.</p></div>
      <div class="box"><h4><span class="icon"><i class="fas fa-undo"></i></span>Fallback did not help</h4><p>A rule frozen on 600 separate images changed accuracy by &minus;0.4 [&minus;1.0, +0.1] in distribution and +0.8 [&minus;0.5, +2.2] held-out. Even an oracle would add only 1.2 and 2.3 points.</p></div>
      <div class="box"><h4><span class="icon"><i class="fas fa-layer-group"></i></span>Memory source matters</h4><p>Removing same-collection images costs 12.8 points, mostly because 45% of queries lose their disease from memory. With the disease kept, a bank of equal size recovers the 5.0-point loss (+5.5 [+1.9, +9.1]). The cause is unknown: it may be a shortcut such as shared background, or closer disease appearance.</p></div>
      <div class="box"><h4><span class="icon"><i class="fas fa-exclamation-circle"></i></span>Uneven held-out gain (post hoc)</h4><p>Sonnet gains 0.0 [&minus;3.8, +3.8] on the seven crops that hold out Bugwood and +21.1 [+13.9, +27.8] on the other three. With crops as the unit, the +6.3 held-out interval is [&minus;3.2, +15.9].</p></div>
    </div>
    <div class="fig-grid" style="margin-top:1.4rem">
      <div class="fig-card"><a href="static/images/figures/ext_models.png" target="_blank"><img src="static/images/figures/ext_models.png" alt="Accuracy by Claude model, and per-crop gains" loading="lazy"></a><p class="description has-text-centered">Two Claude models, with per-crop gains.</p></div>
      <div class="fig-card"><a href="static/images/figures/ext_factorial.png" target="_blank"><img src="static/images/figures/ext_factorial.png" alt="Retrieval by evidence accuracy, and caption effect by crop" loading="lazy"></a><p class="description has-text-centered">Retrieval by evidence, with caption effects by crop.</p></div>
      <div class="fig-card"><a href="static/images/figures/ext_fallback.png" target="_blank"><img src="static/images/figures/ext_fallback.png" alt="Fallback calibration and final accuracy" loading="lazy"></a><p class="description has-text-centered">Fallback threshold sensitivity and final accuracy.</p></div>
      <div class="fig-card"><a href="static/images/figures/ext_collections.png" target="_blank"><img src="static/images/figures/ext_collections.png" alt="Accuracy by memory source, and held-out gain against image shift" loading="lazy"></a><p class="description has-text-centered">Memory source by crop, and held-out gain against image shift.</p></div>
    </div>
    <p class="table-legend" style="margin-top:1rem">Data, statistics and the full report are in <a href="https://github.com/MACE-Bench/MACE-Bench.github.io/tree/main/results/extension" target="_blank" rel="noopener">results/extension</a>.</p>
  </div></div>
</section>
"""

NEW_ABS = "Retrieved labels account for about two thirds of the gain, and in post hoc tests caption text adds 6.4 points in distribution, and 6.3 when an image encoder selects the captions. A second Claude model reproduces the method order. The held-out gain is uneven: Sonnet shows no gain on the seven crops that hold out one collection. MACE falls below SAGE when the true disease is not retrieved, and a retrieval-confidence fallback did not improve accuracy. "


def sub(p, pat, new, count=1):
    s, n = re.subn(pat, lambda _: new, p, count=count, flags=re.S)
    assert n, pat
    return s


def patch(s, abstract):
    if 'id="followup"' in s:
        return s
    s = sub(s, r"Retrieved labels account for most of the gain, caption text adds .{1,40}? points in distribution in a post hoc test, and accuracy falls below SAGE when the true disease is not retrieved\.", "Retrieved labels account for about two thirds of the gain, caption text adds 6.4 points in distribution in a post hoc test, and accuracy falls below SAGE when the true disease is not retrieved. A second Claude model keeps the method order, a confidence fallback did not help, and the held-out gain is uneven: Sonnet gains nothing on the seven crops that hold out one collection.")
    if abstract:
        s = sub(s, r"The retrieved labels account for most of the gain, and a post hoc test.*?(?=MACE adds)", NEW_ABS)
    s = sub(s, r"equal on one and lower on four\.", "equal on one and lower on four. In a post hoc grouping, the gain is 0.0 points on the seven crops that hold out Bugwood and +21.1 on the other three.")
    s = sub(s, r"the held-out difference is inconclusive\.</p>", "the held-out difference is inconclusive. Captions add +6.3 points when DINOv2 selects the memory.</p>")
    s = sub(s, r'<span class="ans good">Stable</span>', '<span class="ans mixed">Depends on memory</span>')
    s = sub(s, r"Memory from the same collection still helps by .{1,40}? points, so shared acquisition conditions contribute\.", "A second Claude model keeps the method order. Memory from the same collection still helps, by +5.5 points at equal bank size and disease mix; the cause is unknown.")
    s = sub(s, r"Deployment would require a signal for retrieval failure, which we did not test\.", "A label-free fallback to SAGE at low retrieval confidence did not detectably change accuracy, and even an oracle would add at most 2.3 points.")
    s = sub(s, r"DINOv2 retrieval matched caption retrieval in distribution, so captions were not required for retrieval there\. Later work should separate the caption as an index from the caption as evidence\.", "Captions added as much after DINOv2 retrieval as after BM25, so in distribution the caption acts mainly as evidence and not as an index.")
    s = sub(s, r"<li>The tests of caption text and DINOv2 retrieval are post hoc\.</li><li>Each result is one run of one VLM\.</li>", "<li>The caption, DINOv2, second-model, fallback and collection tests are post hoc.</li><li>Each result is one run, and both VLMs are Claude models sharing Sonnet captions.</li><li>Seven crops share one held-out collection, so crop and collection shift remain confounded.</li>")
    s = sub(s, r'<a href="#results">Results</a>', '<a href="#results">Results</a><a href="#followup">Follow-up</a>')
    s = sub(s, r'(?=<section class="section is-tint" id="discussion">)', SECTION.lstrip("\n") + "\n")
    return s


def wide():
    H = pd.read_csv(PR / "hero_all_methods.csv"); F = pd.read_csv(PR / "hero_fallback.csv")[["image_id", "adaptive", "oracle", "fallback"]]; X = pd.read_csv(PR / "hero_iid_cross_collection.csv")[["image_id", "covered", "other_hit", "dino_other"]]
    M = rows("E4matched").drop_duplicates("image_id")[["image_id", "correct"]].rename(columns={"correct": "matched"}); M["matched"] = M.matched.astype(float)
    c = ["image_id", "crop", "split", "label", "collection", "hit", "sage", "mace", "labels_all", "dino_labels", "dino_mace", "dino_s0", "h_sage", "h_mace", "h_labels", "other_full", "family"]
    return H[c].merge(F, on="image_id").merge(X, on="image_id", how="left").merge(M, on="image_id", how="left")


def main(site):
    site = Path(site); ext = site / "results/extension"; code = site / "code"
    for d in ("predictions", "statistics", "tables", "processed"):
        (ext / d).mkdir(parents=True, exist_ok=True)
    for n in FINAL:
        with (RAW / n / "predictions.jsonl").open("rb") as f, gzip.GzipFile(ext / f"predictions/{n}.jsonl.gz", "wb", mtime=0) as g:
            shutil.copyfileobj(f, g)
    wide().to_csv(ext / "extension_wide.csv", index=False)
    for p in ST.glob("*.json"):
        shutil.copy(p, ext / "statistics" / p.name)
    for p in TB.iterdir():
        shutil.copy(p, ext / "tables" / p.name)
    for n in ("calibration_signals.csv", "hero_fallback.csv", "hero_iid_cross_collection.csv", "hero_all_methods.csv"):
        shutil.copy(PR / n, ext / "processed" / n)
    for n in ("EXPERIMENT_AUDIT.md", "EXPERIMENT_PROTOCOL.md", "FAILURE_ANALYSIS.md", "BUDGET_LEDGER.csv", "RUN_MANIFEST.json"):
        shutil.copy(ROOT / n, ext / n)
    r = (ROOT / "EXPERIMENT_REPORT.md").read_text(); r = r[:r.index("## 11. Recommended paper changes")].rstrip() + "\n"
    (ext / "EXPERIMENT_REPORT.md").write_text(r)
    with (REPO / "runs/cost_ledger.jsonl").open("rb") as f, gzip.GzipFile(site / "results/cost_ledger.jsonl.gz", "wb", mtime=0) as g:
        shutil.copyfileobj(f, g)
    shutil.copy(ROOT / "scripts/verify_extension.py", site / "results/verify_extension.py")
    for p in (site / "code").rglob("*"):
        src = REPO / p.relative_to(code)
        if p.is_file() and src.is_file():
            shutil.copy(src, p)
    e = code / "experiments/mace_extension"; (e / "scripts").mkdir(parents=True, exist_ok=True); (e / "configurations").mkdir(exist_ok=True)
    shutil.copy(REPO / "experiments/__init__.py", code / "experiments/__init__.py"); shutil.copy(ROOT / "__init__.py", e / "__init__.py")
    for p in (ROOT / "scripts").iterdir():
        if p.suffix in (".py", ".sh") and p.name != "verify_extension.py":
            shutil.copy(p, e / "scripts" / p.name)
    for p in (ROOT / "configurations").iterdir():
        shutil.copy(p, e / "configurations" / p.name)
    (code / "mace10/out").mkdir(exist_ok=True); (e / "processed_results").mkdir(exist_ok=True)   # job lists need these derived vectors
    for n in ("dinov2_embeddings.npz", "dinov2_index.json"):
        shutil.copy(REPO / "mace10/out" / n, code / "mace10/out" / n)
    for n in ("dinov2_extra.npz", "dinov2_extra_index.json"):
        shutil.copy(PR / n, e / "processed_results" / n)
    for a, b in FIG.items():
        shutil.copy(ROOT / f"figures/{a}.png", site / f"static/images/figures/{b}.png")
    (site / "index.html").write_text(patch((site / "index.html").read_text(), True))
    t = REPO / "mace10/site_template.html"; t.write_text(patch(t.read_text(), False))
    print("exported")


if __name__ == "__main__":
    main(sys.argv[1])
