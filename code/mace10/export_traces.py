"""Write the full record of every model call of a package: exact system prompt, user text, attached image ids and hashes,
raw reply, parsed JSON, tokens, cost, model id. Requests are replayed through the same code against the cache; a paid call
cannot happen here, because the runner is replaced by one that raises and agent records are read straight from their cache.

  python -m mace10.export_traces P1 [P2 ...]      writes out/<PKG>/traces.jsonl.gz and out/jobs_all.jsonl.gz
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

from agmem import config as C, data, kb as KB, llm
from . import amend, jobs as JB, run as M


def blocked(system, content):
    raise llm.CallFailed("export found a request with no cached reply")


def jobs_all(J):
    with gzip.open(M.OUT / "jobs_all.jsonl.gz", "wt") as f:
        for j in J:
            f.write(json.dumps(j) + "\n")


def export(pkg, J):
    rows = {json.loads(l)["job_id"]: json.loads(l) for l in (M.OUT / pkg / "predictions.jsonl").open()}
    caps = json.loads((M.OUT / "captions.json").read_text())
    real = llm.call; missing = 0; n = 0
    kbs = {c: (data.classes(c), KB.build(c, data.classes(c))) for c in {j["crop"] for j in J}}
    with gzip.open(M.OUT / pkg / "traces.jsonl.gz", "wt") as f:
        for j in J:
            if not M.SELECT[pkg](j) or j["job_id"] not in rows:
                continue
            rec = rows[j["job_id"]]
            if j["condition"] == "agentic":
                p = C.CACHE / "agentic_out" / f"{rec['trace_key']}.json"
                f.write(json.dumps({"job_id": j["job_id"], "memory": j["memory"], "result": rec, "agent": json.loads(p.read_text())}) + "\n"); n += 1
                continue
            seen = {}

            def spy(system, content, **kw):
                out = real(system, content, **kw)
                seen.update(system=system, text=[c["text"] for c in content if c["type"] == "text"],
                            image_sha=[llm.sha(c["source"]["data"]) for c in content if c["type"] == "image"], reply=out)
                return out
            llm.call = spy
            try:
                M.diagnose(j["crop"], *kbs[j["crop"]], caps, j)
                f.write(json.dumps({"job_id": j["job_id"], "memory": j["memory"], "result": rec, **seen}) + "\n"); n += 1
            except llm.CallFailed:
                missing += 1
            finally:
                llm.call = real
    print(f"{pkg}: {n} traces, {missing} missing")
    return missing


def main(pkgs):
    llm.RUNNER = blocked; llm.set_budget(0)
    with tempfile.TemporaryDirectory() as t:
        llm.LEDGER = Path(t) / "ledger.jsonl"     # replay lines must not touch the real ledger
        J = JB.build() + amend.build(); jobs_all(J)
        return sum(export(p, J) for p in pkgs)


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
