"""Locked, experiment-ready benchmark for caption memory vs agentic few-shot (no paid calls).
.venv/bin/python -m memlock.build calibrate     # kept counts per threshold (no labels of any model output used)
.venv/bin/python -m memlock.build lock TAU      # writes memlock/release/

Near-duplicate removal (unsupervised): SemDeDup criterion (Abbas et al., 2023) on OpenCLIP ViT-B-32 (laion2b_s34b_b79k)
embeddings, applied exhaustively (no k-means approximation) within each (crop, class): images are visited in seeded
order and an image is kept only if its cosine to every already-kept image is < TAU. Before that, exact SHA-256 copies
and pHash Hamming <= PHASH_MAX pairs are collapsed; if such a pair carries two different labels (same crop) or two
different crops, every copy is dropped as a conflict. Guarantee: within a class, any memory/test pair has cosine < TAU.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; REL = HERE / "release"; REPO = HERE.parent
SEED = 20261102; SEEDS = (0, 1, 2); K = 5; PHASH_MAX = 8
MIN_CLASS = 15            # kept images per in-scope class (>= 3 per fold)
TAU_XLABEL = 0.95         # near-identical images with different labels in one crop: both dropped (label noise)
MIN_PROBE = 5             # kept images for an out-of-scope probe class
MIN_CLASSES = 2
OOD_MIN_PER_CLASS = 3   # held-out collection must hold >= 3 kept images of a class to count it


def slug(s): return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
def h(*a): return hashlib.sha256("|".join(map(str, (SEED,) + a)).encode()).hexdigest()


def load():
    recs, kb = [], {}
    for p in sorted((DATA / "fetch_parts_v2").glob("shard_*.json")):
        d = json.loads(p.read_text())
        recs += [r for r in d["images"] if r.get("readable")]
        for k, e in d["kb"].items():
            t = kb.setdefault(k, {"crop": e["crop"], "class": e["class"], "n_rows": 0, "variants": {}})
            t["n_rows"] += e["n_rows"]
            for vk, v in e["variants"].items():
                t["variants"].setdefault(vk, {**v, "n_rows": 0})["n_rows"] += v["n_rows"]
    z = np.load(DATA / "clip_embeddings.npz"); idx = {k: i for i, k in enumerate(z["keys"].tolist())}
    recs = [r for r in recs if r["key"] in idx]
    return recs, kb, z["E"], idx


def conflicts(recs):
    """Exact/pHash duplicates across labels or crops -> drop all copies. Returns dropped keys and log."""
    import imagehash
    par = {r["key"]: r["key"] for r in recs}
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    by = defaultdict(list)
    for r in recs: by[r["sha256"]].append(r["key"])
    for ks in by.values():
        for k in ks[1:]: par[f(k)] = f(ks[0])
    H = np.array([int(r["phash"], 16) for r in recs], dtype=np.uint64)
    for i in range(len(recs)):
        d = np.bitwise_count(H[i + 1:] ^ H[i])
        for j in np.nonzero(d <= PHASH_MAX)[0]:
            par[f(recs[i + 1 + j]["key"])] = f(recs[i]["key"])
    comp = defaultdict(list)
    for r in recs: comp[f(r["key"])].append(r)
    drop, log, dupkeep = set(), [], {}
    for c, v in comp.items():
        if len(v) < 2: continue
        labs = {(r["crop_n"], r["class"]) for r in v}
        if len(labs) > 1:
            drop |= {r["key"] for r in v}; log.append({"members": [r["key"] for r in v], "labels": sorted(map(list, labs)), "action": "dropped (conflict)"})
        else:
            keep = min(v, key=lambda r: h(r["key"]))["key"]
            drop |= {r["key"] for r in v if r["key"] != keep}; dupkeep[keep] = [r["key"] for r in v if r["key"] != keep]
    return drop, log, dupkeep


def ambiguous(recs, E, idx, tau):
    """Drop both members of (a) pairs in one crop with different labels and cosine >= TAU_XLABEL, and (b) pairs of the
    same class name in different crops with cosine >= tau (the same picture filed under two crops)."""
    drop, log = set(), []
    by_crop, by_cls = defaultdict(list), defaultdict(list)
    for r in recs: by_crop[r["crop_n"]].append(r); by_cls[r["class"]].append(r)
    for crop, v in by_crop.items():
        X = E[[idx[r["key"]] for r in v]]; lab = np.array([r["class"] for r in v])
        for i in range(0, len(v), 2048):
            S = X[i:i + 2048] @ X.T
            ii, jj = np.nonzero(S >= TAU_XLABEL)
            for a, b in zip(ii + i, jj):
                if a < b and lab[a] != lab[b]:
                    drop |= {v[a]["key"], v[b]["key"]}; log.append({"kind": "cross_label", "crop": crop, "a": v[a]["key"], "a_class": lab[a], "b": v[b]["key"], "b_class": lab[b], "cos": round(float(S[a - i, b]), 4)})
    for cl, v in by_cls.items():
        crops_ = np.array([r["crop_n"] for r in v])
        if len(set(crops_)) < 2: continue
        X = E[[idx[r["key"]] for r in v]]; S = X @ X.T
        ii, jj = np.nonzero(S >= tau)
        for a, b in zip(ii, jj):
            if a < b and crops_[a] != crops_[b]:
                drop |= {v[a]["key"], v[b]["key"]}; log.append({"kind": "cross_crop", "class": cl, "a": v[a]["key"], "a_crop": crops_[a], "b": v[b]["key"], "b_crop": crops_[b], "cos": round(float(S[a, b]), 4)})
    return drop, log


def semdedup(recs, E, idx, tau):
    by = defaultdict(list)
    for r in recs: by[(r["crop_n"], r["class"])].append(r)
    kept, removed = {}, {}
    for cc, v in by.items():
        v = sorted(v, key=lambda r: h(r["key"])); K_ = []
        for r in v:
            e = E[idx[r["key"]]]
            if K_:
                s = E[[idx[k] for k in K_]] @ e; j = int(s.argmax())
                if s[j] >= tau: removed[r["key"]] = (K_[j], float(s[j])); continue
            K_.append(r["key"])
        kept[cc] = K_
    return kept, removed


def calibrate():
    recs, kb, E, idx = load()
    drop, _, _ = conflicts(recs); recs = [r for r in recs if r["key"] not in drop]
    ins = [r for r in recs if r["role"] == "in_scope"]
    print(len(recs), "after exact/pHash; in-scope", len(ins))
    for tau in (0.98, 0.95, 0.93, 0.92, 0.90, 0.88, 0.85):
        kept, _ = semdedup(ins, E, idx, tau)
        cls_ok = {cc for cc, v in kept.items() if len(v) >= MIN_CLASS}
        crops = Counter(c for c, _ in cls_ok); ok = {c for c, n in crops.items() if n >= MIN_CLASSES}
        print(f"tau={tau}: kept {sum(map(len, kept.values()))}, classes>={MIN_CLASS}: {len(cls_ok)}, crops>=2 classes: {len(ok)}, "
              f"crops>=4 classes: {sum(n >= 4 for n in crops.values())}, images in eligible: {sum(len(kept[cc]) for cc in cls_ok if cc[0] in ok)}")


def lock(tau):
    recs, kb, E, idx = load()
    TX = json.loads((DATA / "text_exclusions.json").read_text()); text_drop = set(TX["excluded"]); WM = set(TX["watermark_only"])
    by_key = {r["key"]: r for r in recs}
    recs = [r for r in recs if r["key"] not in text_drop]   # images whose text names or describes a disease, slides, collages
    drop, conf_log, dupkeep = conflicts(recs)
    recs = [r for r in recs if r["key"] not in drop]
    amb_drop, amb_log = ambiguous(recs, E, idx, tau)
    recs = [r for r in recs if r["key"] not in amb_drop]
    kept, removed = semdedup(recs, E, idx, tau)
    # ---------------- held-out OOD collection per crop (rule fixed before any model run):
    # among collections that hold >= OOD_MIN_PER_CLASS kept images of >= 2 in-scope classes which also keep >= MIN_CLASS
    # images outside it, pick the one covering the most such classes; ties -> fewer images, then name. Never in memory.
    coll_of = {k: by_key[k]["collection"] for v in kept.values() for k in v}
    ood_coll = {}
    for crop in sorted({c for c, _ in kept}):
        ccs = [cc for cc in kept if cc[0] == crop and by_key[kept[cc][0]]["role"] == "in_scope"]
        best = None
        for col in sorted({coll_of[k] for cc in ccs for k in kept[cc]}):
            shared = [cc for cc in ccs if sum(coll_of[k] == col for k in kept[cc]) >= OOD_MIN_PER_CLASS
                      and sum(coll_of[k] != col for k in kept[cc]) >= MIN_CLASS]
            n_img = sum(coll_of[k] == col for cc in ccs for k in kept[cc])
            cand = (len(shared), -n_img, col)
            if len(shared) >= 2 and (best is None or cand[:2] > best[:2]): best = cand
        if best: ood_coll[crop] = best[2]
    idk = {cc: [k for k in v if coll_of[k] != ood_coll.get(cc[0])] for cc, v in kept.items()}
    odk = {cc: [k for k in v if coll_of[k] == ood_coll.get(cc[0])] for cc, v in kept.items()}
    cls_ok = {cc for cc, v in idk.items() if by_key[kept[cc][0]]["role"] == "in_scope" and len(v) >= MIN_CLASS}
    ncls = Counter(c for c, _ in cls_ok); crops = sorted(c for c, n in ncls.items() if n >= MIN_CLASSES)
    single = set()                                             # v2-core: single-class crops are not included
    cls_ok = {cc for cc in cls_ok if cc[0] in crops}
    probes = {cc for cc, v in idk.items() if cc[0] in crops and cc not in cls_ok and len(v) >= MIN_PROBE}
    kept_all = kept; kept = idk          # everything below (folds, memory, audit) uses the in-distribution pool only
    if REL.exists(): shutil.rmtree(REL)
    for d in ("images", "metadata", "splits", "kb", "audit", "code"): (REL / d).mkdir(parents=True)
    # ---------------- images + metadata
    meta = []
    def add(k, crop, cl, role):
        r = by_key[k]; dst = f"images/{slug(crop)}/{k}.jpg"
        (REL / dst).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(DATA / r["file"], REL / dst)
        meta.append({"image_id": k, "file": dst, "crop": crop, "class": cl, "role": role, "ood_collection_of_crop": ood_coll.get(crop),
                     "collection": r["collection"], "parquet_source": r.get("parquet_source"), "raw_label": r["raw_label"],
                     "orig_filename": r["filename"], "sage_shard": r["shard"], "sage_row": r["row"], "sha256": r["sha256"],
                     "phash": r["phash"], "orig_size": r["orig_size"], "n_near_duplicates_removed": 0, "budget_rank": None, "has_watermark": k in WM,
})
    for cc in sorted(cls_ok | probes):
        for k in idk[cc]: add(k, cc[0], cc[1], "iid" if cc in cls_ok else "probe_out_of_scope")
    shared_names = {cl for _, cl in cls_ok}

    for cc in sorted(single):
        for k in kept_all[cc]: add(k, cc[0], cc[1], "xcrop_only")
    for cc in sorted(kept_all):
        if cc[0] in crops:
            for k in odk[cc]: add(k, cc[0], cc[1], "ood_in_scope" if cc in cls_ok else "ood_out_of_scope")
    mk = {m["image_id"]: m for m in meta}
    for k, (kk, s) in removed.items():
        if kk in mk: mk[kk]["n_near_duplicates_removed"] += 1
    for k, v in dupkeep.items():
        if k in mk: mk[k]["n_near_duplicates_removed"] += len(v)
    for cc in cls_ok | probes:          # seeded order inside class: use the first N for a smaller budget
        for i, k in enumerate(sorted(kept[cc], key=lambda k: h("budget", k))): mk[k]["budget_rank"] = i
    for cc in kept_all:
        if cc[0] in crops:
            for i, k in enumerate(sorted([k for k in odk[cc] if k in mk], key=lambda k: h("budget", k))): mk[k]["budget_rank"] = i
    for cc in single:
        for i, k in enumerate(sorted(kept_all[cc], key=lambda k: h("budget", k))): mk[k]["budget_rank"] = i
    # ---------------- IID 5-fold x 3 seeds (stratified by class; one image per near-duplicate cluster)
    folds = []
    for s in SEEDS:
        for cc in sorted(cls_ok):
            for i, k in enumerate(sorted(kept[cc], key=lambda k: h("fold", s, k))):
                folds.append({"seed": s, "image_id": k, "crop": cc[0], "class": cc[1], "fold": i % K})
    fold_of = {(f["seed"], f["image_id"]): f["fold"] for f in folds}
    # ---------------- collection analysis (same vs other collection, identical test items)
    coll = []
    ins_by_crop = defaultdict(list)
    for m in meta:
        if m["role"] == "iid": ins_by_crop[m["crop"]].append(m)
    for s in SEEDS:
        for crop, items in ins_by_crop.items():
            for f in range(K):
                mem = [m for m in items if fold_of[(s, m["image_id"])] != f]
                have = defaultdict(set)
                for m in mem: have[m["class"]].add(m["collection"])
                for t in items:
                    if fold_of[(s, t["image_id"])] != f: continue
                    same = t["collection"] in have[t["class"]]; other = bool(have[t["class"]] - {t["collection"]})
                    coll.append({"seed": s, "fold": f, "image_id": t["image_id"], "crop": crop, "class": t["class"], "collection": t["collection"],
                                 "same_collection_memory_has_class": same, "other_collection_memory_has_class": other,
                                 "paired_eligible": same and other})
    # ---------------- memory label shuffle (seeded permutation of memory labels within crop, per seed x fold)
    shuf = []
    for s in SEEDS:
        for crop, items in ins_by_crop.items():
            for f in range(K):
                mem = sorted((m for m in items if fold_of[(s, m["image_id"])] != f), key=lambda m: m["image_id"])
                labs = [m["class"] for m in mem]; order = sorted(range(len(labs)), key=lambda i: h("shuffle", s, f, crop, i))
                for m, j in zip(mem, order):
                    shuf.append({"seed": s, "fold": f, "crop": crop, "image_id": m["image_id"], "true_class": m["class"], "shuffled_class": labs[j]})
    # ---------------- out-of-scope probes
    oos = [{"image_id": m["image_id"], "crop": m["crop"], "class": m["class"], "collection": m["collection"],
            "protocols": ["open_set: class absent from class list and memory",
                          "in_list_no_memory: class in class list and KB, absent from memory"]} for m in meta if m["role"] == "probe_out_of_scope"]
    # ---------------- OOD collection test (memory = the same IID fold memory; the held-out collection is never in memory)
    ood = [{"image_id": m["image_id"], "crop": m["crop"], "class": m["class"], "collection": m["collection"],
            "scope": "in_scope" if m["role"] == "ood_in_scope" else "out_of_scope"} for m in meta if m["role"].startswith("ood_")]
    # ---------------- same disease, different crop
    cls_crops = defaultdict(list)
    for crop, cl in cls_ok | single: cls_crops[cl].append(crop)
    for cc in single: kept[cc] = kept_all[cc]
    xcrop = [{"class": cl, "crops": sorted(v), "benchmark_crops": sorted(c for c in v if c in crops),
              "pairs": [[a, b] for a in sorted(v) for b in sorted(v) if a != b and a in crops],
              "n_images": {c: len(kept[(c, cl)]) for c in sorted(v)}} for cl, v in sorted(cls_crops.items())
             if len(v) >= 2 and any(c in crops for c in v)]
    # cross-crop visual near-duplicates among same-class pools (memory from crop B for test in crop A must not be a copy)
    xdup = []
    for x in xcrop:
        for a, b in x["pairs"]:
            if a < b or b not in crops:
                Ea = E[[idx[k] for k in kept[(a, x["class"])]]]; Eb = E[[idx[k] for k in kept[(b, x["class"])]]]
                S = Ea @ Eb.T; ii, jj = np.nonzero(S >= tau)
                for i, j in zip(ii, jj):
                    xdup.append({"class": x["class"], "a": kept[(a, x["class"])][i], "b": kept[(b, x["class"])][j], "cos": round(float(S[i, j]), 4)})
    # ---------------- KB (full: every distinct SAGE KB variant for every class of every crop, plus curated KBs)
    kb_by_crop = defaultdict(dict)
    for e in kb.values(): kb_by_crop[e["crop"]][e["class"]] = {"n_sage_rows": e["n_rows"], "variants": sorted(e["variants"].values(), key=lambda v: -v["n_rows"])}
    for crop in crops:
        sc = slug(crop); d = REL / "kb" / sc; d.mkdir(parents=True)
        classes = sorted(cl for c, cl in cls_ok if c == crop); pr = sorted(cl for c, cl in probes if c == crop)
        (d / "sage_kb.json").write_text(json.dumps({"crop": crop, "in_scope_classes": classes, "out_of_scope_classes": pr,
                                                   "classes": {cl: kb_by_crop[crop].get(cl) for cl in classes + pr},
                                                   "other_sage_classes_of_this_crop": sorted(set(kb_by_crop[crop]) - set(classes) - set(pr))}, indent=1))
        wx = REPO / "seqground/data/hf_meta/sage-wacv/table2_dataset/kb" / sc / "internet.xlsx"
        if wx.exists(): shutil.copy2(wx, d / "sage_wacv_internet.xlsx")
        pi = REPO / "seqground/data/hf_meta/sage-wacv/table2_dataset/part_index" / f"{sc}.md"
        if pi.exists(): shutil.copy2(pi, d / "sage_wacv_part_index.md")
        tag = {"soybean": "soy", "corn": "corn", "tomato": "tomato"}.get(sc)
        if tag and (REPO / f"seqground/outputs_full{tag}/kb_corrected_v1.json").exists():
            shutil.copy2(REPO / f"seqground/outputs_full{tag}/kb_corrected_v1.json", d / "kb_corrected_v1.json")
    with (REL / "kb" / "all_sage_crops_kb.jsonl").open("w") as fh:
        for crop in sorted(kb_by_crop):
            for cl, v in sorted(kb_by_crop[crop].items()): fh.write(json.dumps({"crop": crop, "class": cl, **v}) + "\n")
    # ---------------- audit: test-to-memory similarity per IID fold (guarantee check)
    aud = {"tau": tau, "per_seed_fold_max_same_class_cos": [], "max_any_class_cos_quantiles": None}
    anyc = []
    for s in SEEDS:
        for crop, items in ins_by_crop.items():
            ids = [m["image_id"] for m in items]; Ec = E[[idx[k] for k in ids]]; S = Ec @ Ec.T
            fo = np.array([fold_of[(s, k)] for k in ids]); cl = np.array([m["class"] for m in items])
            for f in range(K):
                t = np.nonzero(fo == f)[0]; mm = np.nonzero(fo != f)[0]
                if not len(t) or not len(mm): continue
                sub = S[np.ix_(t, mm)]; same = cl[t][:, None] == cl[mm][None, :]
                aud["per_seed_fold_max_same_class_cos"].append({"seed": s, "crop": crop, "fold": f, "max": round(float(np.where(same, sub, -1).max()), 4)})
                if s == 0: anyc += sub.max(1).tolist()
    aud["max_same_class_cos_overall"] = max(x["max"] for x in aud["per_seed_fold_max_same_class_cos"])
    aud["max_any_class_cos_quantiles"] = {q: round(float(np.quantile(anyc, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)}
    aud["conflict_groups_dropped"] = len(conf_log); aud["exact_or_phash_copies_collapsed"] = sum(map(len, dupkeep.values()))
    aud["text_excluded"] = len(text_drop); aud["semdedup_removed"] = len(removed); aud["label_ambiguous_or_cross_crop_dropped"] = len(amb_drop); aud["cross_crop_same_class_near_duplicates"] = len(xdup)
    # ---------------- write
    import pyarrow as pa, pyarrow.parquet as pq
    def wp(rows, p): pq.write_table(pa.Table.from_pylist(rows), REL / p)
    wp(meta, "metadata/images.parquet"); wp(folds, "splits/iid_5fold_3seed.parquet"); wp(coll, "splits/collection_same_vs_other.parquet")
    wp(shuf, "splits/memory_label_shuffle.parquet")
    wp(ood, "splits/ood_heldout_collection.parquet")
    (REL / "splits/ood_collection_per_crop.json").write_text(json.dumps(ood_coll, indent=1))
    (REL / "splits/out_of_scope.json").write_text(json.dumps(oos, indent=1))
    (REL / "splits/same_disease_other_crop.json").write_text(json.dumps({"classes": xcrop, "near_duplicates_to_exclude": xdup}, indent=1))
    (REL / "audit/dedup_conflicts.json").write_text(json.dumps(conf_log, indent=1))
    (REL / "audit/ambiguous_dropped.json").write_text(json.dumps(amb_log, indent=1))
    (REL / "audit/semdedup_removed.json").write_text(json.dumps({k: {"kept_neighbour": v[0], "cos": round(v[1], 4)} for k, v in removed.items()}))
    (REL / "audit/leakage_guarantee.json").write_text(json.dumps(aud, indent=1))
    summ = {}
    for crop in crops:
        ci = sorted(cl for c, cl in cls_ok if c == crop)
        summ[crop] = {"in_scope_classes": {cl: len(kept[(crop, cl)]) for cl in ci},
                      "out_of_scope_classes": {cl: len(kept[(crop, cl)]) for c, cl in sorted(probes) if c == crop},
                      "id_collections": dict(Counter(m["collection"] for m in meta if m["crop"] == crop and m["role"] == "iid")),
                      "ood_collection": ood_coll.get(crop),
                      "ood_in_scope": sum(m["crop"] == crop and m["role"] == "ood_in_scope" for m in meta),
                      "ood_out_of_scope": sum(m["crop"] == crop and m["role"] == "ood_out_of_scope" for m in meta)}
    (REL / "metadata/summary.json").write_text(json.dumps(summ, indent=1))
    for f in ("build.py", "select_rows.py", "fetch.py", "embed_clip.py", "meta.py", "ocr_text.py", "text_flag.py", "text_rule.py"): shutil.copy2(HERE / f, REL / "code" / f)
    shutil.copy2(DATA / "text_exclusions.json", REL / "audit" / "text_exclusions.json")
    lockf = {str(p.relative_to(REL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(REL.rglob("*")) if p.is_file() and not str(p.relative_to(REL)).startswith("images/")}
    lockf["images/*"] = hashlib.sha256("".join(sorted(m["sha256"] + m["image_id"] for m in meta)).encode()).hexdigest()
    (REL / "LOCK.json").write_text(json.dumps({"locked": "2026-11-02", "tau": tau, "seed": SEED, "files": lockf}, indent=1))
    print(json.dumps({"crops": len(crops), "in_scope_classes": len(cls_ok), "probe_classes": len(probes), "images": len(meta),
                      "roles": dict(Counter(m["role"] for m in meta)), "crops_with_ood": sum(c in ood_coll for c in crops), "xcrop_classes": len(xcrop),
                      "paired_collection_items_seed0": sum(c["paired_eligible"] for c in coll if c["seed"] == 0), **{k: aud[k] for k in aud if k != "per_seed_fold_max_same_class_cos"}}, indent=1))


if __name__ == "__main__":
    {"calibrate": calibrate, "lock": lambda: lock(float(sys.argv[2]))}[sys.argv[1]]()
