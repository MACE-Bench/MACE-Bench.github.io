"""Offline tests (no paid calls): data integrity for the runner, retrieval determinism, an end-to-end run with a fake model."""
import json

import pytest

from agmem import config as C, data, kb as KB, llm, packet, prompts, run as R
from agmem.stats import holm, mcnemar_exact

CROP, SEED, N = "Soybean", 0, 2


def test_no_test_image_in_memory():
    jobs = R.build_jobs(CROP, SEED, N)
    for j in jobs:
        if j["memory"]:
            assert j["image_id"] not in {m["image_id"] for m in j["memory"]}


def test_memory_respects_K_and_variants():
    for j in R.build_jobs(CROP, SEED, N):
        if not j["memory"]:
            continue
        per = {}
        for m in j["memory"]:
            per[m["label_true"]] = per.get(m["label_true"], 0) + 1
        assert max(per.values()) <= j["K"]
        cols = {data.images().loc[m["image_id"], "collection"] for m in j["memory"]}
        if j["condition"] == "same_source":
            assert cols == {j["collection"]}
        if j["condition"] == "other_source":
            assert j["collection"] not in cols


def test_retrieval_is_deterministic():
    kbd = KB.build(CROP, data.classes(CROP))
    cap = {"organ": "leaf", "color": "brown spots with yellow halo", "texture": "dry"}
    mem = [{"id": f"m{i}", "label_shown": "X", "label_true": "X", "caption": {"organ": "leaf", "color": c}} for i, c in enumerate(["brown", "yellow halo", "white powder", "brown spot halo"])]
    assert packet.build(CROP, kbd, cap, mem) == packet.build(CROP, kbd, cap, mem)
    ids = packet.build(CROP, kbd, cap, mem)[1]["memory_ids"]
    assert ids[0] == "m1" and ids[-1] == "m2"   # "yellow halo" best, "white powder" worst for a yellow-halo query


def test_kb_has_an_entry_for_every_candidate():
    kbd = KB.build(CROP, data.classes(CROP))
    assert set(kbd) == set(data.classes(CROP))
    iid = set(data.images()[(data.images().crop == CROP) & (data.images().role == "iid")]["class"])
    missing = [c for c in iid if not kbd[c]["claims"]]
    assert len(missing) <= 2, missing   # documented gaps: SAGE has no symptom text for these diseases


def test_stats():
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(10, 0) < 0.01
    assert holm([0.01, 0.04]) == [0.02, 0.04]


def test_end_to_end_with_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CACHE", tmp_path / "cache"); monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "runs/ledger.jsonl")
    classes = data.classes(CROP)
    def fake(system, content):
        if system == prompts.CAPTION_SYS:
            return {"text": json.dumps({"organ": "leaf", "color": "brown"}), "cost_usd": 0.004}
        return {"text": json.dumps({"prediction": classes[0], "confidence": 0.5}), "cost_usd": 0.016}
    monkeypatch.setattr(llm, "RUNNER", fake)
    monkeypatch.setattr(C, "KS", (1,))
    R.run(CROP, SEED, 1, cap=500.0)
    d = R.out_dir(CROP, SEED, 1)
    recs = [json.loads(l) for l in (d / "predictions.jsonl").open()]
    assert recs and all(r["prediction"] == classes[0] for r in recs)
    assert json.loads((d / "run_status.json").read_text())["budget_stop"] is None
