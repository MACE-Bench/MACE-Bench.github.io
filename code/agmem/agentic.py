"""Agentic memory: the SAGE agent loop with captions instead of reference images.

The agent sees the test image attached to its message (as in the retrieval run), the class list, the same KB text as the SAGE (KB only) condition, and a list of
memory caption files (one per labelled memory image, grouped by disease like SAGE's reference images). Its only tool
reads those text files (the working folder holds no images), and it may read up to k of them, choosing which itself (SAGE's own prompt and budget rules,
agmem/sage_prompt.py with medium="caption"). Claude Code CLI with the Read tool only; no other tools, settings or MCP.
Every call's full stream (messages, tool calls, tool results, final JSON) is stored with the cached result.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import config as C, data, llm, packet, prompts
from .sage_prompt import build_system_prompt, build_user_message

K_READS = 4


def _workdir(job, caps) -> tuple[Path, str, dict]:
    files = {}
    refs: dict[str, list[str]] = {}
    for n, m in enumerate(sorted(job["memory"], key=lambda m: (m["label_shown"], m["image_id"]))):
        cls = m["label_shown"]; rel = f"refs/{cls.replace(' ', '_')}/ref_{sum(1 for x in refs.get(cls, [])) + 1}.txt"
        files[rel] = (packet.caption_text(caps[m["image_id"]]) or "(no visible description)").encode()
        refs.setdefault(cls, []).append(rel)
    key = llm.sha({"files": {k: llm.sha(v) for k, v in sorted(files.items())}})
    wd = C.CACHE / "agentic_work" / key
    if not wd.exists():
        for rel, b in files.items():
            (wd / rel).parent.mkdir(parents=True, exist_ok=True); (wd / rel).write_bytes(b)
    return wd, key, {c: [str(wd / r) for r in v] for c, v in refs.items()}


def _parse(stdout: str):
    trace, res = [], {}
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for c in ev.get("message", {}).get("content", []):
                if c.get("type") == "text" and c.get("text", "").strip():
                    trace.append({"type": "text", "content": c["text"]})
                elif c.get("type") == "tool_use":
                    trace.append({"type": "tool_use", "tool": c.get("name"), "input": c.get("input", {})})
        elif ev.get("type") == "user":
            for c in ev.get("message", {}).get("content", []) if isinstance(ev.get("message", {}).get("content"), list) else []:
                if c.get("type") == "tool_result":
                    body = c.get("content")
                    if isinstance(body, list):
                        body = [b.get("text", "[image]") if b.get("type") == "text" else "[image]" for b in body]
                    trace.append({"type": "tool_result", "content": body})
        elif ev.get("type") == "result":
            res = ev
    return trace, res


def diagnose(crop, classes, kbd, caps, job, k=K_READS, max_turns=None, reserve_usd=0.45) -> dict:
    wd, files_key, refs = _workdir(job, caps)
    kb_text, _ = packet.build(crop, kbd, caps[job["image_id"]], None)     # identical KB to the SAGE condition
    system = build_system_prompt(k=k, medium="caption")
    user = build_user_message("TEST", classes, refs, kb_text, k, medium="caption")
    head = user.split("## Possible Classes")[0]
    user = user.replace(head, "## Test Image\n\nThe test image is attached above this message.\n\n")
    img = llm.image_block(data.path(job["image_id"]))
    kd = {"m": C.MODEL, "agentic": 2, "s": system, "u": user, "img": llm.sha(img["source"]["data"]), "files": files_key}
    if max_turns:
        kd["max_turns"] = max_turns   # hard bound on turns, so on captions exposed (one Read per turn)
    key = llm.sha(kd)
    cp = C.CACHE / "agentic_out" / f"{key}.json"
    if cp.exists() and llm.looks_failed(json.loads(cp.read_text()).get("text"), False) and json.loads(cp.read_text()).get("num_turns") == 1:
        cp.unlink()   # a cached usage-limit reply is not an outcome
    if cp.exists():
        out = json.loads(cp.read_text()); hit = True
        llm._ledger({"key": key, "stage": "diag:agentic", "tag": job["job_id"], "cache_hit": True, "cost_usd": 0.0, "t": time.time()})
    else:
        llm.reserve(reserve_usd)
        cmd = ["claude", "-p", "--model", C.MODEL, "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--no-session-persistence",
               "--tools", "Read", "--allowedTools", "Read", "--strict-mcp-config", "--setting-sources", "", "--system-prompt", system]
        if max_turns:
            cmd += ["--max-turns", str(max_turns)]
        env = {k_: v for k_, v in os.environ.items() if not k_.startswith(("CLAUDE", "CURSOR", "MCP_CONNECTION", "VSCODE", "ELECTRON"))}
        env["DISABLE_PROMPT_CACHING"] = "1"; t0 = time.time()
        msg = {"type": "user", "message": {"role": "user", "content": [img, {"type": "text", "text": user}]}}
        try:
            p = subprocess.run(cmd, input=json.dumps(msg).encode(), capture_output=True, env=env, timeout=600, cwd=str(wd))
        finally:
            llm.release(reserve_usd)
        trace, res = _parse(p.stdout.decode(errors="replace"))
        if not res:
            raise RuntimeError(f"agent produced no result: rc={p.returncode} {p.stderr.decode()[:300]}")
        hit_turn_bound = res.get("subtype") == "error_max_turns"
        if not hit_turn_bound and llm.looks_failed(res.get("result", ""), res.get("is_error")):
            llm._ledger({"key": key, "stage": "diag:agentic", "tag": job["job_id"], "cache_hit": False, "failed": True,
                         "cost_usd": float(res.get("total_cost_usd") or 0.0), "text": str(res.get("result", ""))[:200], "t": time.time()})
            raise llm.CallFailed(str(res.get("result", ""))[:200])
        out = {"text": res.get("result", ""), "trace": trace, "cost_usd": float(res.get("total_cost_usd") or 0.0),
               "num_turns": res.get("num_turns"), "usage": res.get("usage"), "model_ids": list((res.get("modelUsage") or {}).keys()),
               "elapsed_s": round(time.time() - t0, 1), "system_prompt": system, "user_message": user, "workdir": str(wd),
               "hit_turn_bound": hit_turn_bound, "max_turns": max_turns}
        cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(json.dumps(out))
        llm._ledger({"key": key, "stage": "diag:agentic", "tag": job["job_id"], "cache_hit": False, "cost_usd": out["cost_usd"],
                     "num_turns": out["num_turns"], "elapsed_s": out["elapsed_s"], "model_ids": out["model_ids"], "t": time.time()})
        hit = False
    reads = [t["input"].get("file_path", "") for t in out["trace"] if t["type"] == "tool_use" and t["tool"] == "Read"]
    cap_reads = [r for r in reads if r.endswith(".txt")]
    pred = prompts.clean_prediction(llm.parse_json(out["text"]), classes)
    return {**{k_: v for k_, v in job.items() if k_ != "memory"}, **pred, "correct": pred["prediction"] == job["label"],
            "k_budget": k, "caption_reads": len(cap_reads), "read_classes": [Path(r).parent.name.replace("_", " ") for r in cap_reads],
            "non_caption_reads": [r for r in reads if not r.endswith(".txt")], "over_budget": len(cap_reads) > k,
            "memory_pool_size": len(job["memory"]), "num_turns": out.get("num_turns"), "cost_usd": 0.0 if hit else out["cost_usd"], "hit_turn_bound": out.get("hit_turn_bound", False),
            "elapsed_s": out.get("elapsed_s"), "usage": out.get("usage"),
            "cache_hit": hit, "model_ids": out.get("model_ids"), "trace_key": key}
