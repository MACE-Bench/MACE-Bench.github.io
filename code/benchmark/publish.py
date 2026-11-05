"""Write the dataset card and upload memlock/release/ to a private HF dataset repo.
.venv/bin/python -m memlock.publish card          # README.md only (local)
.venv/bin/python -m memlock.publish upload REPO   # e.g. tirtho149/agmem-bench (private)"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REL = HERE / "release"


def card():
    S = json.loads((REL / "metadata/summary.json").read_text()); A = json.loads((REL / "audit/leakage_guarantee.json").read_text())
    L = json.loads((REL / "LOCK.json").read_text()); X = json.loads((REL / "splits/same_disease_other_crop.json").read_text())
    import pyarrow.parquet as pq
    M = pq.read_table(REL / "metadata/images.parquet").to_pylist(); C = pq.read_table(REL / "splits/collection_same_vs_other.parquet").to_pylist()
    from collections import Counter
    rc = Counter(m["role"] for m in M); ins = [m for m in M if m["role"] == "iid"]; oos = [m for m in M if m["role"] == "probe_out_of_scope"]
    rows = "\n".join(f"| {c} | {len(v['in_scope_classes'])} | {sum(v['in_scope_classes'].values())} | {v['ood_collection'] or 'none'} | {v['ood_in_scope']} | "
                     f"{v['ood_out_of_scope'] + sum(v['out_of_scope_classes'].values())} | {len(v['id_collections'])} |" for c, v in sorted(S.items(), key=lambda kv: -sum(kv[1]['in_scope_classes'].values())))
    paired = sum(c["paired_eligible"] for c in C if c["seed"] == 0)
    txt = f"""---
license: cc-by-4.0
pretty_name: AgMEM-Bench (locked splits for caption memory vs agentic few-shot)
task_categories: [image-classification]
tags: [plant-disease, vision-language-model, retrieval, memory, benchmark]
configs:
  - config_name: images
    data_files: metadata/images.parquet
  - config_name: iid_5fold_3seed
    data_files: splits/iid_5fold_3seed.parquet
  - config_name: collection_same_vs_other
    data_files: splits/collection_same_vs_other.parquet
  - config_name: memory_label_shuffle
    data_files: splits/memory_label_shuffle.parquet
---

# AgMEM-Bench

A locked benchmark for one question: can a frozen VLM diagnose crop disease from a few labelled reference images stored
as text captions (caption memory) as well as an agent that looks at the reference images (agentic few-shot), and does any
gain survive when the memory holds no near-copy of the test image?

All images come from [tirtho149/SAGE](https://huggingface.co/datasets/tirtho149/SAGE) (CC-BY-4.0). The splits were fixed
on {L['locked']} before any model was run on them. This is v2 (up to 300 sampled images per class before
near-copy removal; every class has >= 15 distinct images, every crop >= 2 such classes); v1 (up to 100, 49 crops) is kept under the git tag `v1-49crops-cap100`. `LOCK.json` holds the SHA-256 of every file.

## Contents
| | |
|---|---|
| Crops | {len(S)} |
| In-scope classes | {sum(len(v['in_scope_classes']) for v in S.values())} |
| IID images (in-distribution collections, one per near-duplicate cluster) | {rc['iid']} |
| Crops with a held-out OOD collection | {sum(1 for v in S.values() if v['ood_collection'])} |
| OOD in-scope images (held-out collection, class in memory) | {rc['ood_in_scope']} |
| OOD out-of-scope images (held-out collection, class without memory) | {rc['ood_out_of_scope']} |
| Out-of-scope probe classes / images (too few distinct images to fold) | {sum(len(v['out_of_scope_classes']) for v in S.values())} / {rc['probe_out_of_scope']} |
| Source collections | {len({m['collection'] for m in M})} |
| Classes shared by two or more crops | {len(X['classes'])} |

## How near-copies were removed (memory-test variance)
1. Exact SHA-256 copies and pHash pairs with Hamming distance <= 8 were collapsed to one image. If the copies carried different
   labels or crops, all copies were dropped ({A['conflict_groups_dropped']} conflict groups).
   Near-identical images (OpenCLIP cosine >= 0.95) with different labels in one crop, and same-disease images in two crops
   with cosine >= tau, were also dropped, both members ({A['label_ambiguous_or_cross_crop_dropped']} images).
2. SemDeDup (Abbas et al., 2023) on OpenCLIP ViT-B-32 (laion2b_s34b_b79k) embeddings, run exhaustively within each crop and
   class: an image is kept only if its cosine to every kept image of its class is below tau = {A['tau']}
   ({A['semdedup_removed']} images removed).
3. Check (`audit/leakage_guarantee.json`): across every seed and fold, the highest same-class cosine between a test image and
   any memory image is {A['max_same_class_cos_overall']}. Highest cosine to any memory image, median / 99th percentile:
   {A['max_any_class_cos_quantiles']['0.5']} / {A['max_any_class_cos_quantiles']['0.99']}.

## Protocols
Each (seed, fold) defines ONE memory: the IID images of that crop outside fold f. IID test, OOD tests and every control use
that same memory, so their numbers are directly comparable. The held-out OOD collection of each crop
(`splits/ood_collection_per_crop.json`) was chosen by a fixed rule: the collection covering the most classes that keep
>= 15 distinct images elsewhere (ties: fewer images). Report each metric as mean and SD over the 15 seed-fold runs, and pair
conditions on identical test items.

| Protocol | File | Test | Memory (and agentic references) |
|---|---|---|---|
| IID | `splits/iid_5fold_3seed.parquet` | fold f of seed s | IID images of the same crop in the other 4 folds |
| OOD in-scope (collection shift) | `splits/ood_heldout_collection.parquet`, `scope == in_scope` | every image of the crop's held-out collection whose class has memory | the same fold memory as IID (the held-out collection is never in memory, in any fold or seed) |
| OOD out-of-scope | `ood_heldout_collection.parquet` `scope == out_of_scope`, plus `splits/out_of_scope.json` | held-out-collection images of classes without memory, and probe classes | the same fold memory; score as `open_set` (class not listed) or `in_list_no_memory` (class listed, KB present) |
| Same vs other collection | `splits/collection_same_vs_other.parquet` | fold f items with `paired_eligible` ({paired} items at seed 0) | (a) IID memory; (b) IID memory minus the test item's own collection |
| Memory label shuffle | `splits/memory_label_shuffle.parquet` | IID test | IID memory with labels replaced by `shuffled_class` (seeded permutation; class counts preserved) |
| Same disease, other crop | `splits/same_disease_other_crop.json` | IID test items of the shared class | that class's memory taken from the other crop (cross-crop near-copies already removed; `near_duplicates_to_exclude` is a check and should be empty) |

Use `budget_rank` in `metadata/images.parquet` to take a fixed seeded subset per class when the budget is smaller (rank < N).

## Knowledge base
`kb/<crop>/sage_kb.json` holds, for every in-scope and probe class, every distinct SAGE KB entry (visual symptoms, pathogen,
disease type, organ, source URL, verbatim quote) with its row count. `kb/<crop>/sage_wacv_internet.xlsx` and
`sage_wacv_part_index.md` are the SAGE WACV KB, where that crop was in the study; soybean, corn and tomato also have
`kb_corrected_v1.json`. `kb/all_sage_crops_kb.jsonl` covers all SAGE crops.

## Per crop
| Crop | Classes | IID images | Held-out OOD collection | OOD in-scope | Out-of-scope (OOD + probes) | ID collections |
|---|---|---|---|---|---|---|
{rows}

## Files
- `images/<crop>/<image_id>.jpg`: longest side 512 px, JPEG q90.
- `metadata/images.parquet`: image_id, file, crop, class, role, collection, raw label, SAGE shard/row, sha256, pHash, budget_rank.
- `splits/`, `kb/`, `audit/` as above; `code/` reproduces everything from SAGE.
"""
    (REL / "README.md").write_text(txt); print("card written")


def upload(repo):
    for l in open(os.path.expanduser("~/.config/secrets/.env")):
        l = l.strip()
        if l.startswith(("HF_TOKEN=", "export HF_TOKEN=")): os.environ["HF_TOKEN"] = l.split("=", 1)[1].strip().strip("\"'")
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(REL), commit_message="AgMEM-Bench v2.1 frozen: OCR text removal, 59 crops, 290 diseases", delete_patterns=["*"])
    print("uploaded", repo)


if __name__ == "__main__":
    {"card": card, "upload": lambda: upload(sys.argv[2])}[sys.argv[1]]()
