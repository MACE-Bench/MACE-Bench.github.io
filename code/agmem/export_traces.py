"""Write a complete trace for every model call of a run: the exact prompt (system + user text + image file), the raw
model output, the parsed JSON, tokens, cost and model id. Requests are rebuilt deterministically from the frozen code
and data and matched to the saved responses by their content hash, so nothing has to be re-run.

  python -m agmem.export_traces --crop Soybean --seed 0 --n 5 --ks 1,4,16 --conditions sage,agmem --splits test_iid,test_ood_in_scope --tag pilot
Writes <run dir>/traces/captions.jsonl and <run dir>/traces/diagnoses.jsonl (missing responses are counted, not invented).
"""
from __future__ import annotations

import argparse
import json

from . import config as C, data, kb as KB, llm, packet, prompts
from .run import build_jobs, caption_jobs, out_dir


def _response(system, content):
    key = llm.sha({"m": C.MODEL, "s": system, "c": content})
    p = C.CACHE / "llm" / f"{key}.json"
    return key, (json.loads(p.read_text()) if p.exists() else None)


def export(crop, seed, n, ks=None, conditions=None, splits=None, tag=""):
    d = out_dir(crop, seed, n, tag); t = d / "traces"; t.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(crop, seed, n, ks, conditions, splits); cap_ids = caption_jobs(jobs)
    classes = data.classes(crop); kbd = KB.build(crop, classes)
    caps, n_cap = {}, [0, 0]
    with (t / "captions.jsonl").open("w") as f:
        for i in cap_ids:
            content = [llm.image_block(data.path(i)), {"type": "text", "text": prompts.caption_text(crop)}]
            key, r = _response(prompts.CAPTION_SYS, content); n_cap[r is None] += 1
            caps[i] = (r or {}).get("parsed") if isinstance((r or {}).get("parsed"), dict) else {}
            f.write(json.dumps({"call": "caption", "image_id": i, "image_file": str(data.images().loc[i, "file"]), "request_key": key,
                                "system_prompt": prompts.CAPTION_SYS, "user_text": prompts.caption_text(crop), "response_found": r is not None,
                                **({k: r.get(k) for k in ("text", "parsed", "cost_usd", "input_tokens", "output_tokens", "model_ids")} if r else {})}) + "\n")
    n_dx = [0, 0]
    with (t / "diagnoses.jsonl").open("w") as f:
        for j in jobs:
            memory = None if j["memory"] is None else [{"id": "m_" + m["image_id"], "label_shown": m["label_shown"], "label_true": m["label_true"],
                                                        "caption": caps.get(m["image_id"], {})} for m in j["memory"]]
            text, det = packet.build(crop, kbd, caps.get(j["image_id"], {}), memory)
            user = prompts.diagnosis_user(crop, classes, text)
            key, r = _response(prompts.DIAG_SYS, [llm.image_block(data.path(j["image_id"])), {"type": "text", "text": user}])
            n_dx[r is None] += 1
            f.write(json.dumps({"call": "diagnosis", **{k: v for k, v in j.items() if k != "memory"}, "image_file": str(data.images().loc[j["image_id"], "file"]),
                                "memory_pool": [m["image_id"] for m in (j["memory"] or [])], **det, "request_key": key,
                                "system_prompt_sha": llm.sha(prompts.DIAG_SYS), "user_text": user, "response_found": r is not None,
                                **({k: r.get(k) for k in ("text", "parsed", "cost_usd", "input_tokens", "output_tokens", "model_ids")} if r else {})}) + "\n")
    (t / "system_prompts.json").write_text(json.dumps({"caption": prompts.CAPTION_SYS, "diagnosis": prompts.DIAG_SYS}, indent=1))
    s = {"captions_found": n_cap[0], "captions_missing": n_cap[1], "diagnoses_found": n_dx[0], "diagnoses_missing": n_dx[1]}
    (t / "export_status.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", required=True); ap.add_argument("--seed", default="0"); ap.add_argument("--n", type=int)
    ap.add_argument("--ks"); ap.add_argument("--conditions"); ap.add_argument("--splits"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); sd = [int(x) for x in a.seed.split(",")]
    export(a.crop, sd[0] if len(sd) == 1 else sd, a.n, [int(x) for x in a.ks.split(",")] if a.ks else None,
           a.conditions.split(",") if a.conditions else None, a.splits.split(",") if a.splits else None, a.tag)
