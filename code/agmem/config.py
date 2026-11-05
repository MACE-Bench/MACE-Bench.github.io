"""Paths and fixed protocol constants. Change the protocol here only, before a run, never after."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Frozen AgMEM-Bench release (HF tirtho149/agmem-bench, tag HF_REV). Default: the local copy the release was built from.
DATA = Path(os.environ.get("AGMEM_DATA", ROOT.parent / "sage-seqground/memlock/release"))
HF_REPO, HF_REV = "tirtho149/agmem-bench", "v2.1-frozen-2026-11-03"
RUNS = ROOT / "runs"
CACHE = ROOT / ".cache"

MODEL = "sonnet"                 # Claude Code CLI alias; resolved model ids are logged per call
KS = (1, 2, 4, 8, 16)            # memory images per disease (nested sets)
K_RETRIEVE = 4                   # captions retrieved per test image (SAGE agentic view budget)
CAP_TOKENS = 3000                # evidence packet budget (KB claims + captions)
MIN_CLAIMS_PER_CLASS = 1         # every candidate disease keeps at least this many KB claims
IMAGE_MAX_SIDE = 512
PARALLEL = int(os.environ.get("AGMEM_PARALLEL", "8"))
ORGANS = {"foliar": "leaf", "leaf": "leaf", "stem": "stem", "root": "root", "pod": "pod", "seed": "seed",
          "vascular": "stem", "whole plant": "whole_plant", "fruit": "fruit", "crown": "stem", "flower": "flower"}
