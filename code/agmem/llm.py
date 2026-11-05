"""Claude via the Claude Code CLI in headless mode, with a content-addressed cache, a cost ledger and a hard budget.

Locked down for cost and isolation: no tools, a replaced system prompt, no user/project settings or MCP servers,
prompt caching disabled. Images go inline as base64 blocks. Cost per call is the CLI's `total_cost_usd`, a
list-price estimate. Identical requests are served from the cache at zero cost.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from . import config as C

_lock = threading.Lock()
_budget = {"usd": 0.0}
LEDGER = C.RUNS / "cost_ledger.jsonl"


class BudgetExceeded(RuntimeError):
    pass


class CallFailed(RuntimeError):
    """The CLI answered with an error (for example the account's usage-limit message). Never cached."""


_reserved = {"usd": 0.0}
_spent_cache = {"off": 0, "sum": 0.0}
LIMIT_MARKERS = ("usage limit", "limit reached", "rate limit", "hit your limit", "out of extra usage", "credit balance")


def looks_failed(text: str, is_error) -> bool:
    t = (text or "").lower()
    return bool(is_error) or (len(t) < 400 and any(m in t for m in LIMIT_MARKERS))


def sha(obj) -> str:
    s = obj if isinstance(obj, (bytes, str)) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def set_budget(total_usd: float) -> None:
    """Absolute ceiling on the ledger total (all paid calls ever made from this codebase)."""
    _budget["usd"] = float(total_usd)


def spent() -> float:
    """Ledger total of paid calls, read incrementally (the ledger is append-only)."""
    if not LEDGER.exists():
        return 0.0
    with LEDGER.open() as f:
        f.seek(_spent_cache["off"])
        for line in f:
            if not line.endswith("\n"):
                break
            r = json.loads(line); _spent_cache["off"] += len(line.encode())
            if not r["cache_hit"]:
                _spent_cache["sum"] += r["cost_usd"] or 0.0
    return _spent_cache["sum"]


def reserve(usd: float) -> None:
    """Reserve a conservative maximum charge before dispatch; raise if spent + reserved would pass the budget."""
    with _lock:
        if spent() + _reserved["usd"] + usd > _budget["usd"]:
            raise BudgetExceeded(f"spent {spent():.2f} + reserved {_reserved['usd']:.2f} + {usd:.2f} > budget {_budget['usd']:.2f}")
        _reserved["usd"] += usd


def release(usd: float) -> None:
    with _lock:
        _reserved["usd"] = max(0.0, _reserved["usd"] - usd)


def image_block(path: str | Path, max_side: int = C.IMAGE_MAX_SIDE) -> dict:
    from PIL import Image
    p = C.CACHE / "img" / (sha(f"{Path(path).resolve()}|{max_side}") + ".jpg")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(path).convert("RGB"); im.thumbnail((max_side, max_side))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=85); p.write_bytes(buf.getvalue())
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(p.read_bytes()).decode()}}


def _cli(system: str, content: list[dict]) -> dict:
    cmd = ["claude", "-p", "--model", C.MODEL, "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--system-prompt", system]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "CURSOR", "MCP_CONNECTION", "VSCODE", "ELECTRON"))}
    env["DISABLE_PROMPT_CACHING"] = "1"
    C.CACHE.mkdir(parents=True, exist_ok=True); t0 = time.time()
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    proc = subprocess.run(cmd, input=json.dumps(msg).encode(), capture_output=True, env=env, timeout=300, cwd=str(C.CACHE))
    res = {}
    for line in proc.stdout.decode(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            res = ev
    if not res:
        raise RuntimeError(f"no result event; rc={proc.returncode} {proc.stderr.decode()[:300]}")
    u = res.get("usage", {})
    return {"text": res.get("result", ""), "cost_usd": float(res.get("total_cost_usd") or 0.0),
            "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
            "model_ids": list((res.get("modelUsage") or {}).keys()), "is_error": res.get("is_error"),
            "elapsed_s": round(time.time() - t0, 2)}


RUNNER = _cli   # tests replace this with a fake


def _ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock, LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


def call(system: str, content: list[dict], *, stage: str, tag: str = "", retries: int = 1, reserve_usd: float = 0.05) -> dict:
    """Cached call. Error and usage-limit answers are never cached; at most `retries` retries; each attempt is
    reserved against the budget before dispatch, and failed attempts that were charged are written to the ledger."""
    key = sha({"m": C.MODEL, "s": system, "c": content})
    cp = C.CACHE / "llm" / f"{key}.json"
    if cp.exists():
        out = json.loads(cp.read_text())
        if not looks_failed(out.get("text"), out.get("is_error")):
            out["cache_hit"] = True
            _ledger({"key": key, "stage": stage, "tag": tag, "cache_hit": True, "cost_usd": 0.0, "t": time.time()})
            return out
    err = None
    for attempt in range(retries + 1):
        reserve(reserve_usd)
        try:
            out = RUNNER(system, content)
        except Exception as e:  # transient CLI or network failure
            err = e; release(reserve_usd); time.sleep(2 * (attempt + 1)); continue
        release(reserve_usd)
        if looks_failed(out.get("text"), out.get("is_error")):
            _ledger({"key": key, "stage": stage, "tag": tag, "cache_hit": False, "failed": True, "cost_usd": out.get("cost_usd") or 0.0,
                     "elapsed_s": out.get("elapsed_s"), "text": (out.get("text") or "")[:200], "t": time.time()})
            err = CallFailed(out.get("text")); time.sleep(5 * (attempt + 1)); continue
        break
    else:
        raise err if isinstance(err, CallFailed) else RuntimeError(f"call failed after {retries + 1} attempts: {err}")
    out["parsed"] = parse_json(out["text"])
    cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(json.dumps(out))
    _ledger({"key": key, "stage": stage, "tag": tag, "cache_hit": False,
             **{k: out.get(k) for k in ("cost_usd", "input_tokens", "output_tokens", "elapsed_s", "model_ids")}, "t": time.time()})
    out["cache_hit"] = False
    return out


def parse_json(text: str):
    t = (text or "").strip()
    if "```" in t:
        t = next((s[4:] if s.startswith("json") else s for s in t.split("```") if "{" in s), t)
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None
