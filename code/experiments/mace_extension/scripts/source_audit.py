"""Held-out results by source collection, the 2x2 summary and the fallback record, from saved rows. No model call.
Exploratory: chosen after the results were seen.

  python -m experiments.mace_extension.scripts.source_audit    # writes statistics/source_audit.json, tables/table_E_source.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .analyze import ST, TB, contrast, eq, table


def main():
    H = table(); o = H[H.split == "ood"]; S = {}
    pc = o.groupby(["collection", "crop"]).agg(n=("image_id", "size"), sonnet_mace=("mace", "mean"), sonnet_sage=("sage", "mean"), haiku_mace=("h_mace", "mean"), haiku_sage=("h_sage", "mean"), hit=("hit", "mean"), dino=("dino_s0", "mean")).reset_index()
    for c in ("sonnet_mace", "sonnet_sage", "haiku_mace", "haiku_sage", "hit", "dino"):
        pc[c] = (pc[c] * 100).round(1)
    pc["sonnet_gain"] = (pc.sonnet_mace - pc.sonnet_sage).round(1); pc["haiku_gain"] = (pc.haiku_mace - pc.haiku_sage).round(1)
    pc.to_csv(TB / "table_E_source.csv", index=False); S["per_crop"] = pc.to_dict("records")
    bug = o[o.collection == "Bugwood"]; oth = o[o.collection != "Bugwood"]
    for name, d in (("bugwood", bug), ("other", oth)):
        S[name] = {"crops": sorted(d.crop.unique()), "collections": sorted(d.collection.unique()), "n": int(len(d)), "hit": eq(d, "hit"), "sage": eq(d, "sage"), "mace": eq(d, "mace"), "h_sage": eq(d, "h_sage"), "h_mace": eq(d, "h_mace"), "dino": eq(d, "dino_s0"),
                   "sonnet": contrast(d, ["mace"], ["sage"], "s"), "haiku": contrast(d, ["h_mace"], ["h_sage"], "h"), "sonnet_vs_dino": contrast(d, ["mace"], ["dino_s0"], "sd"), "haiku_vs_dino": contrast(d, ["h_mace"], ["dino_s0"], "hd")}
    for k, (m, s) in {"sonnet": ("mace", "sage"), "haiku": ("h_mace", "h_sage")}.items():
        cg = (o.groupby(["collection", "crop"])[m].mean() - o.groupby(["collection", "crop"])[s].mean()) * 100
        S["by_collection_" + k] = cg.groupby("collection").mean().round(1).to_dict()
    S["hero_iid_same_crops"] = {n: {"sonnet": contrast(H[(H.split == "iid") & H.crop.isin(d.crop.unique())], ["mace"], ["sage"], "s"), "haiku": contrast(H[(H.split == "iid") & H.crop.isin(d.crop.unique())], ["h_mace"], ["h_sage"], "h")} for n, d in (("bugwood", bug), ("other", oth))}
    (ST / "source_audit.json").write_text(json.dumps(S, indent=1, default=float))
    print(pc.to_string())
    for n in ("bugwood", "other"):
        v = S[n]; print(n, v["collections"], {k: round(v[k], 1) for k in ("hit", "sage", "mace", "h_sage", "h_mace", "dino")}, [(k, round(v[k]["est"], 1), [round(x, 1) for x in v[k]["ci"]]) for k in ("sonnet", "haiku", "sonnet_vs_dino", "haiku_vs_dino")])
        print(' iid same crops', [(k, round(x["est"], 1)) for k, x in S["hero_iid_same_crops"][n].items()])
    print(S["by_collection_sonnet"], S["by_collection_haiku"])


if __name__ == "__main__":
    main()
