"""Read-only access to the frozen AgMEM-Bench release."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from . import config as C


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@lru_cache(None)
def images() -> pd.DataFrame:
    return pd.read_parquet(C.DATA / "metadata/images.parquet").set_index("image_id", drop=False)


@lru_cache(None)
def assignments() -> pd.DataFrame:
    return pd.read_parquet(C.DATA / "splits/assignments.parquet")


@lru_cache(None)
def memory() -> pd.DataFrame:
    return pd.read_parquet(C.DATA / "splits/memory.parquet")


@lru_cache(None)
def paired() -> pd.DataFrame:
    p = pd.read_parquet(C.DATA / "splits/collection_same_vs_other.parquet")
    return p[p.paired_eligible]


def path(image_id: str):
    return C.DATA / images().loc[image_id, "file"]


def classes(crop: str) -> list[str]:
    """Candidate list: every disease of the crop in the benchmark (IID, held-out-collection and probe diseases)."""
    return sorted(images()[images().crop == crop]["class"].unique())


def test_items(crop: str, seed: int, n_per_disease: int | None) -> pd.DataFrame:
    """Test rows of every run of one seed. With n_per_disease, keep the first N per (run, split, disease) by budget_rank."""
    a = assignments()
    t = a[(a.crop == crop) & (a.seed == seed) & (a.split != "train")].copy()
    t["budget_rank"] = t.image_id.map(images().budget_rank)
    t = t.sort_values(["run", "split", "disease", "budget_rank", "image_id"])
    if n_per_disease:
        t = t.groupby(["run", "split", "disease"], sort=False).head(n_per_disease)
    t["collection"] = t.image_id.map(images().collection)
    return t.reset_index(drop=True)


def memory_set(crop: str, seed: int, run: int, K: int, variant: str = "mixed", test_collection: str | None = None) -> pd.DataFrame:
    m = memory()
    q = m[(m.crop == crop) & (m.seed == seed) & (m.run == run) & (m.K == K) & (m.memory == variant)]
    if variant != "mixed":
        q = q[q.test_collection == test_collection]
    return q.sort_values(["disease", "rank"])


def kb_dir(crop: str):
    return C.DATA / "kb" / slug(crop)


def sage_kb(crop: str) -> dict:
    return json.loads((kb_dir(crop) / "sage_kb.json").read_text())
