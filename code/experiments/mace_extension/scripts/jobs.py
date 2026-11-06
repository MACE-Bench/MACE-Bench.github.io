"""Job lists of the MACE extension (EXPERIMENT_PROTOCOL.md). Built only from locked queries, frozen banks and saved rankings.

Packages
  E1_<tag>   second diagnosis model: sage, mace, labels on the 1,600 hero queries. Sonnet captions stay frozen, so retrieval,
             KB selection and every prompt are byte-identical to the Sonnet rows; only the diagnosis model changes.
  E2         dino_mace on the 1,600 hero queries: the exact DINOv2 top-4 of P9, with their captions (Sonnet).
  E3cal      calibration partition: 60 unseen IID images per crop, never hero or dev, with sage, mace (mixed bank) and
             mace_other (query's collection removed from the bank, a training-side stand-in for collection shift).
  E4other    mace with other_source memory on all 1,000 hero IID queries (the 199 P5 rows return from cache).
  E4family   mace with memory that also drops suspected mirror collections; only queries whose bank differs from E4other.
  E4matched  mace with a random draw from the standard bank that has the same per-disease counts as that query's E4other bank.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from agmem import data
from mace10 import amend, jobs as JB, panels

HERE = Path(__file__).resolve().parent.parent
CAL_PER_CROP = 60
FAMILIES = json.loads((HERE / "configurations/collection_families.json").read_text())["families"] if (HERE / "configurations/collection_families.json").exists() else []


def family_of(collection: str) -> set[str]:
    for f in FAMILIES:
        if collection in f:
            return set(f)
    return {collection}


@lru_cache(None)
def base() -> tuple:
    return tuple(JB.build()), tuple(amend.build())


def retag(j: dict, pkg: str, cond: str | None = None, **kw) -> dict:
    cond = cond or j["condition"]
    return {**j, **kw, "pkg": pkg, "condition": cond, "job_id": f"{pkg}|{cond}|s{j['seed']}|r{j['run']}|K{j['K']}|{j['image_id']}"}


def e1(tag: str) -> list[dict]:
    J, A = base(); out = []
    for j in J:
        if j["pkg"] == "P1" or (j["pkg"] == "P4" and j["variant"] == "labels"):
            out.append(retag(j, f"E1_{tag}"))
    out += [retag(j, f"E1_{tag}") for j in A if j["pkg"] == "P8"]
    return out


def e2() -> list[dict]:
    return [retag(j, "E2", "dino_mace", variant="dino_mace") for j in base()[1] if j["pkg"] == "P9"]


def calibration_queries() -> pd.DataFrame:
    """Next images of each crop's locked round-robin order, after every hero and dev item. IID role only."""
    Q = JB.queries(); I = data.images(); used = set(Q.image_id); rows = []
    for crop in Q.crop.unique():
        pool = I[(I.crop == crop) & (I.role == "iid")]
        picked = panels.round_robin(pool, len(pool))
        picked = picked[~picked.image_id.isin(used)].head(CAL_PER_CROP)
        for r in picked.itertuples():
            rows.append({"crop": crop, "image_id": r.image_id, "split": "cal", "label": I.loc[r.image_id, "class"], "collection": r.collection,
                         "run": panels.run_for(0, crop, r.image_id, "iid")})
    return pd.DataFrame(rows)


def e3cal() -> list[dict]:
    out = []
    for r in calibration_queries().itertuples():
        b = {"variant": "mace", "crop": r.crop, "image_id": r.image_id, "split": "cal", "label": r.label, "collection": r.collection, "seed": 0, "run": r.run, "K": 4}
        for cond, mem in (("sage", None), ("mace", JB.mem(r.crop, 0, r.run, 4)), ("mace_other", JB.mem(r.crop, 0, r.run, 4, "other_source", r.collection))):
            out.append(retag({**b, "memory": mem, "condition": cond}, "E3cal"))
    return out


def hero_iid():
    Q = JB.queries()
    return Q[Q.hero & (Q.split == "iid")]


def e4other() -> list[dict]:
    out = []
    for r in hero_iid().itertuples():
        b = {"variant": "mace", "crop": r.crop, "image_id": r.image_id, "split": r.split, "label": r.label, "collection": r.collection, "seed": 0, "run": r.run_s0, "K": 4}
        out.append(retag({**b, "memory": JB.mem(r.crop, 0, r.run_s0, 4, "other_source", r.collection), "condition": "other_source"}, "E4other"))
    return out


def e4family() -> list[dict]:
    I = data.images(); out = []
    for j in e4other():
        fam = family_of(j["collection"])
        keep = [m for m in j["memory"] if I.loc[m["image_id"], "collection"] not in fam]
        if len(keep) != len(j["memory"]):
            out.append(retag({**j, "memory": keep}, "E4family", "family_disjoint"))
    return out


def e4matched() -> list[dict]:
    """Same size and disease mix as E4other, drawn without regard to collection, so the two differ only in which images they hold."""
    out = []
    for j in e4other():
        want = Counter(m["label"] for m in j["memory"]); std = JB.mem(j["crop"], 0, j["run"], 4); keep = set()
        for d, n in sorted(want.items()):
            ids = sorted(m["image_id"] for m in std if m["label"] == d)
            keep |= set(random.Random(f"E4matched|{j['image_id']}|{d}").sample(ids, min(n, len(ids))))
        out.append(retag({**j, "memory": [m for m in std if m["image_id"] in keep]}, "E4matched", "size_matched"))
    return out


def build(pkg: str) -> list[dict]:
    if pkg.startswith("E1_"):
        return e1(pkg[3:])
    return {"E2": e2, "E3cal": e3cal, "E4other": e4other, "E4family": e4family, "E4matched": e4matched}[pkg]()


@lru_cache(None)
def bank_ids(crop: str, run: int) -> tuple[str, ...]:
    return tuple(m["image_id"] for m in JB.mem(crop, 0, int(run), 4))
