"""Make memlock/release/ viewer-ready and presentable (no paid calls). .venv/bin/python -m memlock.finalize
1. images/<split>/<crop>/<id>.jpg with images/<split>/metadata.jsonl, so the HF viewer previews every protocol split
2. assets/: logo, pipeline flowchart, example grid, per-crop overview
3. README.md dataset card; LOCK.json refreshed"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Circle
from matplotlib.path import Path as MPath
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent; REL = HERE / "release"; DATA = HERE / "data"; A = REL / "assets"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURF, LINE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SPLITS = ["iid", "ood_in_scope", "ood_out_of_scope", "probe_out_of_scope"]
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "text.color": INK, "axes.edgecolor": LINE,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def slug(s): return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def restructure():
    M = pq.read_table(REL / "metadata/images.parquet").to_pylist()
    meta_lines = defaultdict(list)
    for m in M:
        new = f"images/{m['role']}/{slug(m['crop'])}/{m['image_id']}.jpg"
        if m["file"] != new:
            (REL / new).parent.mkdir(parents=True, exist_ok=True); shutil.move(str(REL / m["file"]), str(REL / new)); m["file"] = new
        meta_lines[m["role"]].append({"file_name": new.split("/", 2)[2], "image_id": m["image_id"], "crop": m["crop"], "disease": m["class"],
                                      "split_role": m["role"], "collection": m["collection"], "held_out_collection_of_crop": m["ood_collection_of_crop"],
                                      "sage_raw_label": m["raw_label"], "budget_rank": m["budget_rank"]})
    for d in list((REL / "images").iterdir()):
        if d.is_dir() and d.name not in SPLITS:
            shutil.rmtree(d)
    for role, rows in meta_lines.items():
        with (REL / "images" / role / "metadata.jsonl").open("w") as f:
            for r in sorted(rows, key=lambda r: (r["crop"], r["disease"], r["image_id"])): f.write(json.dumps(r) + "\n")
    pq.write_table(pa.Table.from_pylist(M), REL / "metadata/images.parquet")
    return M


# ------------------------------------------------------------------ assets
def logo():
    fig = plt.figure(figsize=(7.2, 1.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 7.2); ax.set_ylim(0, 1.9); ax.axis("off")
    verts = [(0.35, 0.3), (0.45, 1.35), (1.25, 1.65), (1.75, 1.6), (1.6, 0.95), (1.15, 0.3), (0.35, 0.3)]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * 6
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=AQUA, edgecolor="none"))
    ax.plot([0.42, 1.62], [0.37, 1.52], color=SURF, lw=2.4, solid_capstyle="round")
    for t in (0.3, 0.5, 0.7):                       # side veins
        x, y = 0.42 + 1.2 * t, 0.37 + 1.15 * t
        ax.plot([x, x - 0.22], [y, y + 0.2], color=SURF, lw=1.5, solid_capstyle="round")
        ax.plot([x, x + 0.2], [y, y - 0.17], color=SURF, lw=1.5, solid_capstyle="round")
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    t1 = ax.text(2.15, 1.08, "AgMEM", fontsize=42, fontweight="bold", color=INK, va="center")
    bb = t1.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    ax.text(bb.x1 + 0.02, 1.08, "-Bench", fontsize=42, fontweight="bold", color=BLUE, va="center")
    ax.text(2.18, 0.42, "Caption memory vs agentic few-shot for crop disease diagnosis", fontsize=11.5, color=INK2, va="center")
    fig.savefig(A / "logo.png", dpi=220); plt.close(fig)


def box(ax, x, y, w, h, title, sub, color, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fill or "white", edgecolor=color, lw=1.6))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, shrinkA=2, shrinkB=2))


def pipeline(st):
    fig = plt.figure(figsize=(11, 6.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(0.35, 6.25, "How the benchmark was built", fontsize=15, fontweight="bold")
    ax.text(0.35, 5.93, "Every step is unsupervised or rule-based and was fixed before any model saw the data.", fontsize=9.5, color=INK2)
    W, H, y = 2.35, 1.05, 4.45
    steps = [("SAGE source", f"{st['sage_rows']:,} images\n{st['sage_crops']} crops, {st['collections']} collections", BLUE),
             ("Seeded selection", f"<= 300 per class, round-robin\nover collections: {st['selected']:,}", BLUE),
             ("Copy and conflict removal", f"SHA-256 + pHash <= 8\n-{st['copies']:,} copies, -{st['conflicts']:,} conflicts", ORANGE),
             ("Text and label noise", f"OCR text -{st['text']:,}; cross-label\ncos >= 0.95, cross-crop: -{st['ambiguous']:,}", ORANGE)]
    xs = [0.35 + i * 2.62 for i in range(4)]
    for (t, s, c), x in zip(steps, xs): box(ax, x, y, W, H, t, s, c)
    for x in xs[:-1]: arrow(ax, x + W, y + H / 2, x + 2.62, y + H / 2)
    y2 = 2.75
    steps2 = [("SemDeDup within class", f"OpenCLIP ViT-B-32, keep only if\ncos < 0.90 to every kept image: -{st['semdedup']:,}", ORANGE),
              ("Held-out collection", f"one per crop by fixed rule,\nnever in memory ({st['crops_ood']} crops)", AQUA),
              ("Eligibility", ">= 15 distinct images per class\n>= 2 classes per crop", AQUA),
              ("AgMEM-Bench", f"{st['crops']} crops, {st['classes']} diseases\n{st['images']:,} images", BLUE)]
    for (t, s, c), x in zip(steps2, xs): box(ax, x, y2, W, H, t, s, c, fill="#eef4fc" if t == "AgMEM-Bench" else None)
    ym = (y + y2 + H) / 2
    ax.plot([xs[3] + W / 2, xs[3] + W / 2, xs[0] + W / 2], [y, ym, ym], color=MUTED, lw=1.3)
    arrow(ax, xs[0] + W / 2, ym, xs[0] + W / 2, y2 + H)
    for x in xs[:-1]: arrow(ax, x + W, y2 + H / 2, x + 2.62, y2 + H / 2)
    # protocol fan-out
    prots = [("IID", "3 seeds x 5 runs"), ("OOD in-scope", "held-out collection"), ("OOD out-of-scope", "no memory for class"),
             ("Probes", "rare classes"), ("Same disease,\nother crop", f"{st['xcrop']} diseases"), ("Same vs other\ncollection", "paired items"),
             ("Memory label\nshuffle", "seeded permutation")]
    px = [0.35 + i * 1.51 for i in range(7)]; py = 0.45
    for (t, s), x in zip(prots, px):
        ax.add_patch(FancyBboxPatch((x, py), 1.36, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor="white", edgecolor=LINE, lw=1.2))
        ax.text(x + 0.68, py + 0.6, t, ha="center", va="center", fontsize=8.8, fontweight="bold", linespacing=1.05)
        ax.text(x + 0.68, py + 0.2, s, ha="center", va="center", fontsize=7.8, color=INK2)
        arrow(ax, x + 0.68, 1.85, x + 0.68, py + 0.95)
    ax.plot([xs[3] + W / 2, xs[3] + W / 2], [y2, 1.85], color=MUTED, lw=1.3)
    ax.plot([px[0] + 0.68, px[-1] + 0.68], [1.85, 1.85], color=MUTED, lw=1.3)
    ax.text(0.35, 2.05, "Every protocol uses the same memory per (seed, run)", fontsize=9.5, color=INK2, fontweight="bold")
    for c, l, xx in ((BLUE, "data", 7.0), (ORANGE, "removal", 8.0), (AQUA, "split rule", 9.25)):
        ax.add_patch(FancyBboxPatch((xx, 6.17), 0.22, 0.16, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor="white", edgecolor=c, lw=1.6))
        ax.text(xx + 0.32, 6.25, l, va="center", fontsize=9, color=INK2)
    fig.savefig(A / "pipeline.png", dpi=180); plt.close(fig)


def examples(M):
    import re; re_stock = re.compile(r"shutterstock|istock|alamy|getty|dreamstime", re.I)
    rng = random.Random(7); fig, axs = plt.subplots(3, 6, figsize=(11, 6.3))
    rows = [("iid", "IID"), ("ood_in_scope", "OOD in-scope"), ("ood_out_of_scope", "OOD out-of-scope")]
    for r, (role, name) in enumerate(rows):
        pool = [m for m in M if m["role"] == role]; seen, pick = set(), []
        for m in sorted(pool, key=lambda m: hashlib.sha256(f"7{m['image_id']}".encode()).hexdigest()):
            if re_stock.search(m["orig_filename"] or "") or m["collection"] in ("VQAPlantDise", "PlantWild"): continue
            if m["crop"] not in seen: pick.append(m); seen.add(m["crop"])
            if len(pick) == 6: break
        for c, m in enumerate(pick):
            ax = axs[r, c]; im = Image.open(REL / m["file"]).convert("RGB"); w, h = im.size; s = min(w, h)
            ax.imshow(im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            dis = m["class"] if len(m["class"]) <= 24 else m["class"][:22] + "..."
            ax.set_title(f"{m['crop']}\n{dis}", fontsize=8.2, color=INK, pad=3, linespacing=1.1)
        axs[r, 0].set_ylabel(name, fontsize=10.5, fontweight="bold", color=[BLUE, ORANGE, ORANGE][r], labelpad=8)
        for sp in ("left",): axs[r, 0].spines[sp].set_visible(False)
    fig.tight_layout(h_pad=0.9, w_pad=0.6); fig.savefig(A / "examples.png", dpi=150); plt.close(fig)


def crops_chart(M):
    iid = Counter(m["crop"] for m in M if m["role"] == "iid")
    ncls = Counter(); seen = set()
    for m in M:
        if m["role"] == "iid" and (m["crop"], m["class"]) not in seen: seen.add((m["crop"], m["class"])); ncls[m["crop"]] += 1
    ood = Counter(m["crop"] for m in M if m["role"].startswith("ood"))
    order = sorted(iid, key=lambda c: -iid[c]); n = len(order)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.17 * n + 1.4), sharey=True, gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.06})
    y = list(range(n))[::-1]
    a1.barh(y, [iid[c] for c in order], color=BLUE, height=0.72, label="IID")
    a1.barh(y, [ood[c] for c in order], left=[iid[c] for c in order], color=ORANGE, height=0.72, label="OOD (held-out collection)")
    a1.set_yticks(y); a1.set_yticklabels(order, fontsize=7.6); a1.set_xlabel("Images"); a1.grid(axis="x", color=LINE, lw=0.6); a1.set_axisbelow(True)
    a1.legend(frameon=False, loc="lower right", fontsize=8.5); a1.tick_params(axis="y", length=0)
    a1.set_title("Images per crop", loc="left", fontsize=10.5, fontweight="bold")
    a2.scatter([ncls[c] for c in order], y, s=22, color=AQUA, zorder=3)
    a2.hlines(y, 0, [ncls[c] for c in order], color=AQUA, lw=1.2); a2.set_xlim(0, max(ncls.values()) + 2)
    a2.set_xlabel("Disease classes"); a2.grid(axis="x", color=LINE, lw=0.6); a2.set_axisbelow(True); a2.tick_params(axis="y", length=0)
    a2.set_title("Classes per crop", loc="left", fontsize=10.5, fontweight="bold")
    a1.set_ylim(-0.7, n - 0.3)
    fig.savefig(A / "crops.png", dpi=160, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


# ------------------------------------------------------------------ card
def stats(M):
    sel = json.loads((DATA / "selection.json").read_text())
    meta = json.loads((DATA / "sage_meta_all.json").read_text())["rows"]
    aud = json.loads((REL / "audit/leakage_guarantee.json").read_text())
    X = json.loads((REL / "splits/same_disease_other_crop.json").read_text())
    oc = json.loads((REL / "splits/ood_collection_per_crop.json").read_text())
    S = json.loads((REL / "metadata/summary.json").read_text())
    rc = Counter(m["role"] for m in M)
    import re
    coll = {re.sub(r"[_\- ]*\(?\d+\)?$", "", re.sub(r"\.[A-Za-z0-9]+$", "", r["filename"] or "")) for r in meta}
    return {"sage_rows": len(meta), "sage_crops": len({r["crop"].strip().title() for r in meta}), "collections": len(coll),
            "selected": len(sel["rows"]), "copies": aud["exact_or_phash_copies_collapsed"], "conflicts": aud["conflict_groups_dropped"],
            "ambiguous": aud["label_ambiguous_or_cross_crop_dropped"], "semdedup": aud["semdedup_removed"], "text": aud.get("text_excluded", 0),
            "crops": len(S), "classes": sum(len(v["in_scope_classes"]) for v in S.values()), "images": len(M), "rc": rc,
            "crops_ood": sum(1 for c in S if S[c]["ood_collection"]), "xcrop": len(X["classes"]), "aud": aud, "S": S, "oc": oc}


def card(st, M):
    L = json.loads((REL / "LOCK.json").read_text()); rc = st["rc"]; aud = st["aud"]; S = st["S"]
    F = pq.read_table(REL / "splits/iid_5fold_3seed.parquet").to_pylist()
    n = Counter((f["crop"], f["class"]) for f in F if f["seed"] == 0); t = Counter((f["crop"], f["class"]) for f in F if f["seed"] == 0 and f["fold"] not in (0, 1))
    v = sorted(n.values()); te = sorted(t[k] for k in n); med = lambda a: a[len(a) // 2]
    C = pq.read_table(REL / "splits/collection_same_vs_other.parquet").to_pylist(); paired = sum(c["paired_eligible"] for c in C if c["seed"] == 0 and c["run"] == 0)
    held = Counter(st["oc"][c] for c in S if S[c]["ood_collection"])
    rows = "\n".join(f"| {c} | {len(x['in_scope_classes'])} | {sum(x['in_scope_classes'].values()):,} | {x['ood_collection'] or '-'} | "
                     f"{x['ood_in_scope'] or '-'} | {(x['ood_out_of_scope'] + sum(x['out_of_scope_classes'].values())) or '-'} |"
                     for c, x in sorted(S.items(), key=lambda kv: -sum(kv[1]["in_scope_classes"].values())))
    badge = lambda l, m, c: f'<img alt="{l}" src="https://img.shields.io/badge/{l.replace("-", "--").replace(" ", "_")}-{m.replace("-", "--").replace(" ", "_")}-{c}">'
    yaml = """---
license: cc-by-4.0
pretty_name: AgMEM-Bench
language: [en]
task_categories: [image-classification, zero-shot-image-classification]
tags: [agriculture, plant-disease, crop-disease, vision-language-model, retrieval-augmented, few-shot, benchmark, ood]
size_categories: [10K<n<100K]
configs:
  - config_name: images
    default: true
    drop_labels: true
    data_files:
      - split: iid
        path: images/iid/**
      - split: ood_in_scope
        path: images/ood_in_scope/**
      - split: ood_out_of_scope
        path: images/ood_out_of_scope/**
      - split: probe_out_of_scope
        path: images/probe_out_of_scope/**
  - config_name: assignments
    data_files: splits/assignments.parquet
  - config_name: memory
    data_files: splits/memory.parquet
  - config_name: iid_folds
    data_files: splits/iid_5fold_3seed.parquet
  - config_name: ood_heldout_collection
    data_files: splits/ood_heldout_collection.parquet
  - config_name: collection_same_vs_other
    data_files: splits/collection_same_vs_other.parquet
  - config_name: metadata
    data_files: metadata/images.parquet
---
"""
    body = f"""
<p align="center"><img src="assets/logo.png" alt="AgMEM-Bench" width="640"></p>

<p align="center">
{badge("crops", str(st['crops']), "2a78d6")} {badge("diseases", str(st['classes']), "2a78d6")} {badge("images", f"{st['images']:,}".replace(',', '%2C'), "2a78d6")}
{badge("memory K", "1 2 4 8 16", "1baf7a")} {badge("near-copies", "removed", "1baf7a")} {badge("splits", f"locked {L['locked']}", "eb6834")} {badge("license", "CC BY 4.0", "52514e")}
</p>

**AgMEM-Bench** tests one claim: a frozen vision-language model that reads short text captions of labelled reference
images (caption memory) can diagnose crop disease as well as an agent that opens the reference images themselves
(agentic few-shot), at a fraction of the cost. A memory gain means little if the memory holds a near-copy of the test
image, so the benchmark removes near-copies before any split and checks the guarantee on every fold.

## At a glance

| | |
|---|---|
| Crops | **{st['crops']}**, each with at least 2 disease classes |
| Disease classes | **{st['classes']}**, each with at least 15 distinct images |
| IID images | {rc['iid']:,} (per class: min {v[0]}, median {med(v)}, max {v[-1]}) |
| IID test images per class per run | min {te[0]}, median {med(te)} (3 of 5 folds) |
| OOD in-scope images | {rc['ood_in_scope']:,} from a held-out collection in {st['crops_ood']} crops |
| OOD out-of-scope images | {rc['ood_out_of_scope'] + rc['probe_out_of_scope']:,} ({rc['ood_out_of_scope']:,} held-out collection, {rc['probe_out_of_scope']:,} rare-class probes) |
| Same disease on two or more crops | {st['xcrop']} diseases |
| Knowledge base | every SAGE source-grounded entry for every class, with URL and verbatim quote |

<p align="center"><img src="assets/examples.png" alt="Example images per split" width="900"></p>

## How it was built

<p align="center"><img src="assets/pipeline.png" alt="Build pipeline" width="900"></p>

1. **Selection.** Up to 300 images per (crop, disease) from [tirtho149/SAGE](https://huggingface.co/datasets/tirtho149/SAGE),
   drawn with a fixed seed and round-robin over source collections so no single collection dominates a class.
2. **Copies and conflicts.** Exact SHA-256 copies and pHash pairs within Hamming distance 8 collapse to one image.
   When copies carry different labels or crops, every copy is dropped ({st['conflicts']:,} groups).
3. **Text and label noise.** Apple Vision OCR reads every image. Images whose text names or describes a disease,
   pathogen or symptom, carries a caption, slide or document text, or that OpenCLIP rates as a slide or collage, are
   dropped ({st['text']:,} images). Images whose only text is a watermark, credit or ID stamp stay, flagged
   `has_watermark`. Near-identical images (OpenCLIP cosine >= 0.95) filed under different diseases of one crop, and
   same-disease images filed under two crops (cosine >= 0.90), are both dropped ({st['ambiguous']:,} images).
4. **Near-copy removal.** SemDeDup (Abbas et al., 2023) on OpenCLIP ViT-B-32 (laion2b_s34b_b79k) embeddings, run
   exhaustively within each crop and disease: an image stays only if its cosine to every kept image is below 0.90
   ({st['semdedup']:,} images removed).
5. **Held-out collection.** For each crop with several collections, one collection is held out by a fixed rule (the one
   covering the most diseases that keep at least 15 images elsewhere). It never enters memory.
6. **Eligibility.** A disease needs at least 15 distinct in-distribution images; a crop needs at least 2 such diseases.

## Guarantees (checked, see `audit/leakage_guarantee.json`)

| Check | Result |
|---|---|
| Highest same-disease cosine between a test image and any memory image, over all seeds and folds | {aud['max_same_class_cos_overall']} (bound: below 0.90) |
| Highest cosine to any memory image, median / 99th percentile | {aud['max_any_class_cos_quantiles']['0.5']} / {aud['max_any_class_cos_quantiles']['0.99']} |
| Near-identical images with different labels in one crop | none (all pairs >= 0.95 removed) |
| Images whose text names or describes a disease | none (OCR on every image, {st['text']:,} removed) |
| Same-disease near-copies across crops | {aud['cross_crop_same_class_near_duplicates']} |
| Duplicate image IDs / duplicate file hashes | 0 / 0 |

## Train and test, explicitly

`splits/assignments.parquet` has one row per (seed, run, crop, image) and a `split` column. **train** is the labelled
pool from which memory is chosen. Every other value is a test set. Per (seed, run):

| split | images per (seed, run) | changes with run? |
|---|---|---|
| `train` | {st['assign']['train']:,} | yes, 2 of the 5 folds (memory is chosen from these) |
| `test_iid` | {st['assign']['test_iid']:,} | yes, the other 3 folds |
| `test_ood_in_scope` | {st['assign']['test_ood_in_scope']:,} | no, the held-out collection |
| `test_ood_out_of_scope` | {st['assign']['test_ood_out_of_scope']:,} | no |
| `test_probe` | {st['assign']['test_probe']:,} | no |

Run *r* of seed *s* uses folds *r* and *r*+1 (mod 5) as train and the other three as test, so every IID image is train in
2 runs and test in 3 runs of each seed. Seeds 0, 1, 2 times runs 0 to 4 give 15 runs. Train always comes from the same crop as the test image and never contains
a test image or the crop's held-out collection.

### Memory: K = 1, 2, 4, 8, 16 representatives per disease

Memory holds K labelled images per disease, K in {{1, 2, 4, 8, 16}}. In each (seed, run, crop, disease) the images are
ordered by greedy facility location on OpenCLIP embeddings of that disease's train images: each step adds the image that
best covers the train images not yet represented. The first K images of that order form the size-K memory, so the sets
are nested (K = 4 contains K = 2) and the memory-size curve isolates the effect of adding memory. Ag-MEM retrieves
**4** captions per test image and the agentic baseline views **4** images, the SAGE budget; **K = 4** is the headline
point. Diseases with fewer than K train images use all of them (the smallest has 6 per run; the median has 22).

`splits/memory.parquet` (one row per memory image and K; select `K == k`):

| memory | what it holds | question it answers |
|---|---|---|
| `mixed` (default) | K per disease from all train collections of the crop | headline IID and OOD numbers |
| `same_source` | K per disease from the test image's own collection only (`test_collection`) | does memory help when it looks like the test image? |
| `other_source` | K per disease with the test image's collection removed | does memory transfer across collections? |
| `mixed` + `noise25_class` | 25% of memory labels replaced by another disease | robustness to label noise |
| `mixed` + `shuffled_class` | all memory labels permuted | does the model use the labels at all? |

Memory images per (seed, run), `mixed`: """ + ", ".join(f"K = {K}: {st['reps']['per_run'][K]:,}" for K in (1, 2, 4, 8, 16)) + f""".
Distinct images to caption over every variant and run: K = 4: {st['reps']['union'][4]:,}; K = 16: {st['reps']['union'][16]:,}.

**Reproduce and verify.** `python code/verify.py` re-checks every guarantee from the released files (hashes in
`LOCK.json`, no train/test overlap, held-out collections absent from train, each IID image tested 3 times per seed, the
0.90 near-copy bound from the shipped OpenCLIP embeddings, the shuffle being a permutation). The last run is in
`audit/verify_output.txt`. `code/` rebuilds the release from SAGE with fixed seeds (`meta.py`, `select_rows.py`,
`fetch.py`, `embed_clip.py`, `build.py lock 0.90`, `finalize.py`).

## Protocols

Every (seed, run) defines the memories: K representatives per disease, chosen from that run's train folds. All seven protocols use that same memory,
so their numbers compare directly. Report the mean and SD over the 15 runs, and pair conditions on identical test items.

| Protocol | Test | Memory | File |
|---|---|---|---|
| **IID** | the 3 test folds of run *r*, seed *s* | `mixed` memory, K per disease from the 2 train folds | `splits/assignments.parquet`, `splits/memory.parquet` |
| **OOD in-scope** | held-out collection, disease present in memory | same fold memory | `splits/ood_heldout_collection.parquet` (`scope == in_scope`) |
| **OOD out-of-scope** | held-out collection or rare-class probe, disease absent from memory | same fold memory | `ood_heldout_collection.parquet` (`scope == out_of_scope`), `splits/out_of_scope.json` |
| **Same disease, other crop** | IID test of a shared disease | that disease's memory from another crop | `splits/same_disease_other_crop.json` |
| **Same vs other source** | {paired:,} paired items per seed | `same_source` vs `other_source` vs `mixed` | `splits/collection_same_vs_other.parquet`, `splits/memory.parquet` |
| **Label noise / shuffle** | IID test | `mixed` with `noise25_class` or `shuffled_class` | `splits/memory.parquet` |

Out-of-scope items support two scorings: `open_set` (the disease is not in the class list) and `in_list_no_memory`
(the disease is listed and has KB text, but no memory). Held-out collections: {", ".join(f"{k} ({v} crops)" for k, v in held.most_common())}.

## Quick start

```python
from datasets import load_dataset
import pandas as pd
from huggingface_hub import hf_hub_download

iid = load_dataset("tirtho149/agmem-bench", "images", split="iid")          # image + crop, disease, collection
asg = pd.read_parquet(hf_hub_download("tirtho149/agmem-bench", "splits/assignments.parquet", repo_type="dataset"))

seed, run, crop = 0, 0, "Corn"
a = asg[(asg.seed == seed) & (asg.run == run) & (asg.crop == crop)]
test_ids = set(a[a.split == "test_iid"].image_id)  # also test_ood_in_scope, test_ood_out_of_scope, test_probe
memory = pd.read_parquet(hf_hub_download("tirtho149/agmem-bench", "splits/memory.parquet", repo_type="dataset"))
mem = memory[(memory.seed == seed) & (memory.run == run) & (memory.crop == crop) & (memory.memory == "mixed") & (memory.K == 4)]
memory_ids = set(mem.image_id)                    # 4 per disease; caption these, agentic refs view 4 of the same
```

For a smaller budget, keep images with `budget_rank < N` (a fixed seeded order inside each disease).

## Knowledge base

`kb/<crop>/sage_kb.json` lists, for every disease in the crop, each distinct SAGE knowledge entry (visual symptoms,
pathogen, disease type, affected organ, source URL, verbatim quote) with its row count. Where the crop was in the SAGE
WACV study, `sage_wacv_internet.xlsx` and `sage_wacv_part_index.md` hold that KB; soybean, corn and tomato also carry
`kb_corrected_v1.json`. `kb/all_sage_crops_kb.jsonl` covers all {st['sage_crops']} SAGE crops.

## Per crop

<p align="center"><img src="assets/crops.png" alt="Images and classes per crop" width="900"></p>

<details><summary>Table: all {st['crops']} crops</summary>

| Crop | Diseases | IID images | Held-out collection | OOD in-scope | Out-of-scope |
|---|---|---|---|---|---|
{rows}

</details>

## Layout

```
images/<split>/<crop>/<image_id>.jpg   longest side 512 px, JPEG q90; <split>/metadata.jsonl for the viewer
metadata/images.parquet                 one row per image: crop, disease, role, collection, SAGE shard/row, sha256, pHash, has_watermark
metadata/summary.json                   per-crop counts
splits/assignments.parquet              explicit train / test per (seed, run, crop, image)
splits/memory.parquet                   K = 1, 2, 4, 8, 16 per disease (nested): mixed / same_source / other_source, noise, shuffle
splits/                                 the seven protocols
kb/                                     knowledge base per crop
audit/                                  guarantees, removed images and why
code/                                   scripts that rebuild everything from SAGE
LICENSE.md                              CC BY 4.0 (data), MIT (code), third-party notes
LOCK.json                               SHA-256 of every file; splits fixed {L['locked']}
```

## Versions

`main` is v2: {st['crops']} crops, at least 15 distinct images per disease. The earlier v1 (49 crops, up to 100 images
per disease before near-copy removal) stays under the tag `v1-49crops-cap100`.

## Citation

Images and the knowledge base come from SAGE. Please cite it:

```bibtex
@article{{sage2026,
  title   = {{SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis}},
  author  = {{Arshad, Muhammad Arbab and Roy, Tirtho and Shen, Yanben and Elango, Dinakaran and
             Chiranjeevi, Shivani and Singh, Asheesh K. and Ganapathysubramanian, Baskar and
             Hegde, Chinmay and Singh, Arti and Sarkar, Soumik}},
  journal = {{arXiv preprint arXiv:2605.09768}},
  year    = {{2026}}
}}
```

## License

Data: **CC BY 4.0**, inherited from SAGE. Code in `code/`: **MIT**. Source collections and quoted knowledge-base pages
keep their own terms; `metadata/images.parquet` traces every image to its collection and original filename. Labels were
not re-verified by plant pathologists. Full terms: [LICENSE.md](LICENSE.md).
"""
    (REL / "README.md").write_text(yaml + body)


TRAIN_FOLDS = 2   # per run: train (memory pool) = 2 of 5 folds (40%), test = the other 3 (60%)


def run_train_folds(r): return {r % 5, (r + 1) % 5}


def assignments(M):
    """One row per (seed, run, crop, image). Run r of seed s: train = folds r and r+1 (mod 5), test_iid = the other 3.
    Every IID image is train in 2 runs and test in 3 runs of each seed. OOD and probe images are test in every run."""
    F = pq.read_table(REL / "splits/iid_5fold_3seed.parquet").to_pylist()
    fold = {(f["seed"], f["image_id"]): f["fold"] for f in F}
    tests = {"ood_in_scope": "test_ood_in_scope", "ood_out_of_scope": "test_ood_out_of_scope", "probe_out_of_scope": "test_probe"}
    rows = []
    for s in (0, 1, 2):
        for r in range(5):
            tf = run_train_folds(r)
            for m in M:
                sp = ("train" if fold[(s, m["image_id"])] in tf else "test_iid") if m["role"] == "iid" else tests[m["role"]]
                rows.append({"seed": s, "run": r, "crop": m["crop"], "disease": m["class"], "image_id": m["image_id"], "split": sp, "file": m["file"]})
    pq.write_table(pa.Table.from_pylist(rows), REL / "splits/assignments.parquet")
    # same vs other collection: recomputed for the new train/test definition
    coll = {m["image_id"]: m["collection"] for m in M}; out = []
    by = defaultdict(list)
    for x in rows: by[(x["seed"], x["run"], x["crop"])].append(x)
    for (s, r, c), g in by.items():
        have = defaultdict(set)
        for x in g:
            if x["split"] == "train": have[x["disease"]].add(coll[x["image_id"]])
        for x in g:
            if x["split"] != "test_iid": continue
            same = coll[x["image_id"]] in have[x["disease"]]; other = bool(have[x["disease"]] - {coll[x["image_id"]]})
            out.append({"seed": s, "run": r, "image_id": x["image_id"], "crop": c, "disease": x["disease"], "collection": coll[x["image_id"]],
                        "same_collection_memory_has_class": same, "other_collection_memory_has_class": other, "paired_eligible": same and other})
    pq.write_table(pa.Table.from_pylist(out), REL / "splits/collection_same_vs_other.parquet")
    import numpy as np
    z = np.load(DATA / "clip_embeddings.npz"); ix = {k: i for i, k in enumerate(z["keys"].tolist())}
    ids = [m["image_id"] for m in M]
    np.savez_compressed(REL / "audit/clip_vitb32_embeddings.npz", ids=np.array(ids), E=z["E"][[ix[i] for i in ids]].astype(np.float16))
    return Counter(x["split"] for x in rows if x["seed"] == 0 and x["run"] == 0)


LICENSE = (HERE / "LICENSE.md").read_text()


KS = (1, 2, 4, 8, 16)   # memory images per disease; K = 4 matches SAGE's agentic view budget
K_VIEW = 4              # Ag-MEM retrieves 4 captions per test image; the agentic baseline views 4 images
MEMORIES = ("mixed", "same_source", "other_source")


def facility_order(X, kmax):
    """Greedy facility location: repeatedly add the image that most increases sum_i max_j sim(i, chosen_j).
    The order is nested, so the first K picks form the size-K memory for every K."""
    import numpy as np
    S = X @ X.T; n = len(X); best = np.full(n, -1.0); order = []
    for _ in range(min(kmax, n)):
        gain = np.maximum(S, best[:, None]).sum(0) - best.sum()
        gain[order] = -np.inf; j = int(np.argmax(gain)); order.append(j); best = np.maximum(best, S[:, j])
    return order


def representatives(M):
    """Memory per (seed, fold, crop, variant): for every disease, the first K images of a greedy facility-location order
    over OpenCLIP embeddings of that disease's train images (K in KS, nested).
      mixed        all train collections of the crop (default)
      same_source  only the test image's own collection        (one memory per collection)
      other_source the test image's own collection removed     (one memory per collection)
    Label variants per memory and K: shuffled_class (all labels permuted) and noise25_class (25% replaced)."""
    import numpy as np
    z = np.load(DATA / "clip_embeddings.npz"); ix = {k: i for i, k in enumerate(z["keys"].tolist())}; E = z["E"]
    A = pq.read_table(REL / "splits/assignments.parquet").to_pylist(); coll = {m["image_id"]: m["collection"] for m in M}
    test_colls = defaultdict(set); train = defaultdict(list)
    for a in A:
        if a["split"] == "train": train[(a["seed"], a["run"], a["crop"], a["disease"])].append(a["image_id"])
        else: test_colls[(a["seed"], a["run"], a["crop"])].add(coll[a["image_id"]])
    rows = []
    for (s, f, c, d), ids in sorted(train.items()):
        variants = [("mixed", None, ids)]
        for col in sorted(test_colls[(s, f, c)]):
            variants += [("same_source", col, [i for i in ids if coll[i] == col]), ("other_source", col, [i for i in ids if coll[i] != col])]
        for mname, col, pool in variants:
            if not pool: continue
            pool = sorted(pool); order = [pool[j] for j in facility_order(E[[ix[i] for i in pool]], max(KS))]
            for K in KS:
                for r, i in enumerate(order[:K]):
                    rows.append({"seed": s, "run": f, "crop": c, "memory": mname, "test_collection": col, "K": K, "disease": d,
                                 "image_id": i, "collection": coll[i], "rank": r})
    by = defaultdict(list)
    for r in rows: by[(r["seed"], r["run"], r["crop"], r["memory"], r["test_collection"], r["K"])].append(r)
    for key, g in by.items():
        g.sort(key=lambda r: r["image_id"]); labs = [r["disease"] for r in g]; classes = sorted(set(labs))
        order = sorted(range(len(g)), key=lambda i: hashlib.sha256(f"shuffle|{key}|{i}".encode()).hexdigest())
        for r, j in zip(g, order): r["shuffled_class"] = labs[j]
        for r in g:
            hh = hashlib.sha256(f"noise25|{key[:5]}|{r['image_id']}".encode()).hexdigest()
            others = [x for x in classes if x != r["disease"]]
            r["noise25_class"] = others[int(hh[8:16], 16) % len(others)] if (int(hh[:8], 16) % 4 == 0 and others) else r["disease"]
    pq.write_table(pa.Table.from_pylist(rows), REL / "splits/memory.parquet")
    per = {K: sum(1 for r in rows if r["memory"] == "mixed" and r["K"] == K and r["seed"] == 0 and r["run"] == 0) for K in KS}
    union = {K: len({r["image_id"] for r in rows if r["K"] == K}) for K in KS}
    return {"per_run": per, "union": union, "n_rows": len(rows)}


def drop_legacy():
    for f in ("splits/memory_label_shuffle.parquet", "splits/memory_representatives.parquet", "splits/contamination_near_copies.parquet"):
        if (REL / f).exists(): (REL / f).unlink()
    if (REL / "contamination").exists(): shutil.rmtree(REL / "contamination")


def relock():
    L = json.loads((REL / "LOCK.json").read_text()); M = pq.read_table(REL / "metadata/images.parquet").to_pylist()
    files = {str(p.relative_to(REL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(REL.rglob("*"))
             if p.is_file() and not str(p.relative_to(REL)).startswith("images/") and p.name not in ("LOCK.json", "README.md", "verify_output.txt")}
    files["images/*"] = hashlib.sha256("".join(sorted(m["sha256"] + m["image_id"] for m in M)).encode()).hexdigest()
    L["files"] = files; (REL / "LOCK.json").write_text(json.dumps(L, indent=1))


def main():
    A.mkdir(exist_ok=True)
    M = restructure(); st = stats(M); st["assign"] = assignments(M); st["reps"] = representatives(M)
    logo(); pipeline(st); examples(M); crops_chart(M)
    drop_legacy(); (REL / "LICENSE.md").write_text(LICENSE)
    shutil.copy2(Path(__file__), REL / "code" / "finalize.py"); shutil.copy2(HERE / "verify.py", REL / "code" / "verify.py"); shutil.copy2(HERE / "LICENSE.md", REL / "code" / "LICENSE.md")
    relock(); card(st, M)
    import subprocess; r = subprocess.run([sys.executable, str(REL / "code/verify.py")], capture_output=True, text=True); print(r.stdout)
    (REL / "audit/verify_output.txt").write_text(r.stdout)
    print("finalized", st["crops"], st["classes"], st["images"])


if __name__ == "__main__":
    main()
