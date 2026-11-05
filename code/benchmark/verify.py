"""Re-check every AgMEM-Bench guarantee from the released files alone. Run from the dataset root:
    python code/verify.py
Needs numpy, pandas, pyarrow. Exits non-zero on any failure."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent.parent
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))

L = json.loads((R / "LOCK.json").read_text())
M = pd.read_parquet(R / "metadata/images.parquet")
A = pd.read_parquet(R / "splits/assignments.parquet")
bad = [f for f, h in L["files"].items() if f != "images/*" and hashlib.sha256((R / f).read_bytes()).hexdigest() != h]
check("LOCK.json hashes match every split/metadata/kb/audit/code file", not bad, ", ".join(bad[:5]))
img_digest = hashlib.sha256("".join(sorted(M.sha256 + M.image_id)).encode()).hexdigest()
check("image set matches LOCK.json", img_digest == L["files"]["images/*"])
check("every image file exists", all((R / f).exists() for f in M.file), f"{len(M)} files")
check("no duplicate image_id or sha256", M.image_id.is_unique and M.sha256.is_unique)
role = dict(zip(M.image_id, M.role)); crop = dict(zip(M.image_id, M.crop)); cls = dict(zip(M.image_id, M["class"]))
coll = dict(zip(M.image_id, M.collection)); held = dict(zip(M.crop, M.ood_collection_of_crop))
iid = A[A.split.isin(["train", "test_iid"])]
nt = iid[iid.split == "test_iid"].groupby(["seed", "image_id"]).size(); ntr = iid[iid.split == "train"].groupby(["seed", "image_id"]).size()
check("each IID image is test in 3 runs and train in 2 runs of every seed", (nt == 3).all() and (ntr == 2).all() and len(nt) == 3 * (M.role == "iid").sum())
viol_overlap = viol_role = viol_held = viol_crop = 0
for (s, f, c), g in A.groupby(["seed", "run", "crop"]):
    tr = set(g[g.split == "train"].image_id); te = set(g[g.split != "train"].image_id)
    viol_overlap += len(tr & te)
    viol_role += sum(role[i] != "iid" for i in tr)
    viol_held += sum(coll[i] == held[c] for i in tr if held[c])
    viol_crop += sum(crop[i] != c for i in tr | te)
check("train and test never overlap in any (seed, run, crop)", viol_overlap == 0)
check("train holds only IID images", viol_role == 0)
check("held-out OOD collection never in train", viol_held == 0)
check("train and test stay within one crop", viol_crop == 0)
check("every disease has >= 15 IID images", (M[M.role == "iid"].groupby(["crop", "class"]).size() >= 15).all())
z = np.load(R / "audit/clip_vitb32_embeddings.npz"); E = z["E"].astype(np.float32); ix = {k: i for i, k in enumerate(z["ids"].tolist())}
worst = 0.0
for (s, f, c), g in A[A.seed == 0].groupby(["seed", "run", "crop"]):
    tr = g[g.split == "train"].image_id.tolist(); te = g[g.split == "test_iid"].image_id.tolist()
    if not tr or not te: continue
    S = E[[ix[i] for i in te]] @ E[[ix[i] for i in tr]].T
    same = np.array([cls[i] for i in te])[:, None] == np.array([cls[i] for i in tr])[None, :]
    if same.any(): worst = max(worst, float(S[same].max()))
check("same-disease test-train cosine < 0.90 (OpenCLIP ViT-B-32, fp16, seed 0)", worst < 0.9005, f"max {worst:.4f}")
Mm = pd.read_parquet(R / "splits/memory.parquet")
trainset = {(r.seed, r.run, r.crop, r.image_id) for r in A[A.split == "train"].itertuples()}
check("every memory image is a train image of the same (seed, run, crop)", all((r.seed, r.run, r.crop, r.image_id) in trainset for r in Mm.itertuples()))
cnt = Mm.groupby(["seed", "run", "crop", "memory", "test_collection", "K", "disease"], dropna=False).size().reset_index(name="n")
check("at most K memory images per disease", (cnt.n <= cnt.K).all())
nest = Mm.groupby(["seed", "run", "crop", "memory", "test_collection", "disease"], dropna=False)
check("memories are nested (size-K set contains every smaller set)",
      all(all(set(g[g.K == a].image_id) <= set(g[g.K == b].image_id) for a, b in ((1, 2), (2, 4), (4, 8), (8, 16))) for _, g in nest))
ss = Mm[Mm.memory == "same_source"]; os_ = Mm[Mm.memory == "other_source"]
check("same_source memory comes only from the test collection", (ss.collection == ss.test_collection).all())
check("other_source memory never contains the test collection", (os_.collection != os_.test_collection).all())
grp = Mm.groupby(["seed", "run", "crop", "memory", "test_collection", "K"], dropna=False)
check("shuffled labels are a permutation within each memory", all(sorted(g.disease) == sorted(g.shuffled_class) for _, g in grp))
mx = Mm[Mm.memory == "mixed"]; rate = (mx.noise25_class != mx.disease).mean()
check("label noise rate near 25% on the mixed memory", 0.18 < rate < 0.30, f"{rate:.3f}")
check("no near-copy or contamination files shipped", not (R / "contamination").exists() and not (R / "splits/contamination_near_copies.parquet").exists())
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
