"""Offline tests of the ten-crop dispatcher with a fake model: no paid calls."""
import json

import pytest

from agmem import config as C, llm
from mace10 import amend, jobs as JB, run as M


JOBS = []


def all_jobs():
    if not JOBS:
        JOBS.extend(JB.build())
    return JOBS


AM = []
real_amend = amend.build


def amend_jobs():
    if not AM:
        AM.extend(real_amend())
    return AM


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(JB, "build", all_jobs); monkeypatch.setattr(amend, "build", amend_jobs)
    monkeypatch.setattr(C, "CACHE", tmp_path / "cache"); monkeypatch.setattr(C, "PARALLEL", 2)
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "ledger.jsonl"); monkeypatch.setitem(llm._spent_cache, "off", 0); monkeypatch.setitem(llm._spent_cache, "sum", 0.0)
    monkeypatch.setattr(M, "OUT", tmp_path / "out"); monkeypatch.setattr(M, "LEDGER_CSV", tmp_path / "ledger.csv"); monkeypatch.setattr(M, "LEDGER_START", 0.0)
    monkeypatch.setattr(M.time, "sleep", lambda s: None); monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    (tmp_path / "out").mkdir(); (tmp_path / "ledger.csv").write_text(M.LEDGER_HEADER)
    return tmp_path


def fake(calls, fail_after=None):
    def runner(system, content):
        calls.append(content)
        if fail_after is not None and len(calls) > fail_after:
            return {"text": "You've hit your limit", "cost_usd": 0.0, "is_error": True}
        if system == M.prompts.CAPTION_SYS:
            text = json.dumps({"organ": "leaf", "color": f"brown {len(calls)}"})
        else:
            text = json.dumps({"prediction": "UNKNOWN", "confidence": 0.5})
        return {"text": text, "cost_usd": 0.01, "input_tokens": 1, "output_tokens": 1, "model_ids": ["fake"], "is_error": False, "elapsed_s": 0.1}
    return runner


def rows(tmp, pkg):
    return [json.loads(l) for l in (tmp / "out" / pkg / "predictions.jsonl").open()]


def test_every_package_selects_the_planned_jobs():
    J = all_jobs() + amend_jobs(); n = {p: sum(map(f, J)) for p, f in M.SELECT.items()}
    assert n == {"P7dev": 40, "P1": 3200, "P4labels": 400, "P2": 3200, "P3": 800, "P4rest": 1200, "P5": 398, "P6": 200, "P7": 200, "P8": 1200, "P9": 1600}


def test_dino_labels_changes_only_the_four_labels():
    from agmem import data, kb as KB
    from mace10 import conditions
    j = next(x for x in amend_jobs() if x["pkg"] == "P9")
    assert len(j["retrieved_ids"]) == 4 and j["retrieved_cos"] == sorted(j["retrieved_cos"], reverse=True)
    assert set(j["retrieved_ids"]) <= {m["image_id"] for m in j["memory"]} and j["image_id"] not in j["retrieved_ids"]
    classes = data.classes(j["crop"]); kbd = KB.build(j["crop"], classes); cap = {"organ": "leaf", "color": "brown"}
    mem = [{"id": "m_" + m["image_id"], "label_shown": m["label"], "label_true": m["label"], "caption": {"color": "brown spots " + m["image_id"]}} for m in j["memory"]]
    a, _, _ = conditions.build("labels", j["crop"], kbd, cap, mem, j["image_id"], classes)
    b, det, _ = conditions.build("dino_labels", j["crop"], kbd, cap, mem, j["image_id"], classes, j["retrieved_ids"])
    head = "### Labelled reference observations"
    assert a.split(head)[0] == b.split(head)[0] and a.split(head)[1].split("\n")[0] == b.split(head)[1].split("\n")[0]
    assert det["memory_ids"] == ["m_" + i for i in j["retrieved_ids"]]


def test_run_then_resume_makes_no_new_calls(sandbox, monkeypatch):
    calls = []; monkeypatch.setattr(llm, "RUNNER", fake(calls))
    s = M.run("P1", 400, limit=4)
    assert len(rows(sandbox, "P1")) == 4 and s["stop"] is None and not s["complete"]
    n = len(calls)
    M.run("P1", 400, limit=4)
    assert len(calls) == n and len(rows(sandbox, "P1")) == 4


def test_image_references_attach_four_images_after_the_test_image(sandbox, monkeypatch):
    calls = []; monkeypatch.setattr(llm, "RUNNER", fake(calls))
    M.run("P7dev", 400, limit=1)
    diag = calls[-1]
    assert sum(c["type"] == "image" for c in diag) == 5 and diag[-1]["type"] == "text"
    assert "reference image 4" in diag[-1]["text"]


@pytest.mark.parametrize("pkg", ["P4labels", "P4rest", "P5", "P8", "P9"])
def test_variants_run(sandbox, monkeypatch, pkg):
    monkeypatch.setattr(llm, "RUNNER", fake([]))
    assert len(rows(sandbox, pkg) if M.run(pkg, 400, limit=2) else []) == 2


def test_usage_limit_stops_the_queue_and_is_not_written(sandbox, monkeypatch):
    calls = []; monkeypatch.setattr(llm, "RUNNER", fake(calls, fail_after=3))
    s = M.run("P1", 400, limit=6)
    assert s["stop"] and "CallFailed" in s["stop"] and not s["complete"]
    assert not list((C.CACHE / "llm").glob("*.json")) or all("limit" not in p.read_text() for p in (C.CACHE / "llm").glob("*.json"))


def test_budget_stop(sandbox, monkeypatch):
    monkeypatch.setattr(llm, "RUNNER", fake([]))
    s = M.run("P1", 0.02, limit=6)
    assert s["stop"] and "BudgetExceeded" in s["stop"]
