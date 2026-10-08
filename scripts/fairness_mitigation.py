import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--alpha", type=float, default=0.5, help="weight = group size ** -alpha")
ap.add_argument("--boot", type=int, default=100)
args = ap.parse_args()
args.cutoff = None
args.pending = None
args.workload = False
cfg = load_config(args)
slug = cfg["slug"]
meta = json.load(open(f"models/{slug}_aft_meta.json"))
cutoff = pd.Timestamp(meta["cutoff"])
CATS = meta["categorical_fields"]
PATH = f"data/processed/{normalized_name(cfg)}"
DIMS = ["court_tier", "district_name", "type_name_normalized"]

df = pd.read_csv(PATH, dtype=str, usecols=CATS + ["date_of_filing", "date_of_decision", "event"])
df["date_of_filing"] = pd.to_datetime(df["date_of_filing"], errors="coerce")
df["date_of_decision"] = pd.to_datetime(df["date_of_decision"], errors="coerce")
df = df[df["date_of_filing"].notna()].copy()
for c in CATS:
    df[c] = df[c].fillna("missing")
decided = (df["event"] == "1") & (df["date_of_decision"] <= cutoff)
end = df["date_of_decision"].where(decided, cutoff)
df["duration"] = (end - df["date_of_filing"]).dt.days.clip(lower=1)
df["obs"] = decided.astype(int)

X = pd.get_dummies(df[CATS], dtype="uint8")
names = list(X.columns)
assert names == json.load(open(f"models/{meta['columns_file']}")), "feature columns differ from the saved model"
M = X.values.astype("float32")
d_all = df["duration"].values.astype("float32")
o_all = df["obs"].values
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=o_all)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=o_all[tr])
d_te, o_te = d_all[te], o_all[te]
G = {dim: df[dim].values[te] for dim in DIMS}

lo, hi = np.percentile(d_te, [100 / 3, 200 / 3])
known = (o_te == 1) | (d_te > hi)
true = np.where(d_te <= lo, 0, np.where(d_te <= hi, 1, 2))
print(f"[{slug}] train {len(tr)} | test {len(te)} | scored cases {known.sum()} | true cutoffs {lo:.0f}/{hi:.0f} days")


def macro_f1(t, p):
    cm = np.bincount(t * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f.mean())


masks_f, masks_c = {}, {}
for dim in DIMS:
    groups = pd.Series(G[dim]).value_counts().index
    masks_f[dim] = [G[dim] == g for g in groups if ((G[dim] == g) & known).sum() >= 1000]
    masks_c[dim] = [G[dim] == g for g in groups if (G[dim] == g).sum() >= 1000 and o_te[G[dim] == g].sum() >= 50]
    print(f"   {dim}: {len(masks_f[dim])} groups for Macro F1, {len(masks_c[dim])} for C-index")

params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
          "aft_loss_distribution_scale": 1.2, "tree_method": "hist", "learning_rate": 0.10,
          "max_depth": 6, "nthread": 4}


def dm(i, w=None):
    m = xgb.DMatrix(M[i], feature_names=names)
    m.set_float_info("label_lower_bound", d_all[i])
    m.set_float_info("label_upper_bound", np.where(o_all[i] == 1, d_all[i], np.inf).astype("float32"))
    if w is not None:
        m.set_weight(w)
    return m


def fit_predict(w=None):
    dtr, dva = dm(tr, w), dm(va)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    return bst[: bst.best_iteration + 1].predict(xgb.DMatrix(M[te], feature_names=names))


def evaluate(pred):
    pt = np.digitize(pred, np.percentile(pred, [100 / 3, 200 / 3]), right=True)
    res = {"c_index": concordance_index(d_te, pred, o_te), "macro_f1": macro_f1(true[known], pt[known])}
    for dim in DIMS:
        f1s = [macro_f1(true[mk & known], pt[mk & known]) for mk in masks_f[dim]]
        cs = [concordance_index(d_te[mk], pred[mk], o_te[mk]) for mk in masks_c[dim]]
        res[f"f1gap_{dim}"] = max(f1s) - min(f1s)
        res[f"cgap_{dim}"] = max(cs) - min(cs)
    return res, pt


def weights(dim):
    g = pd.Series(df[dim].values[tr])
    w = g.map(g.value_counts()).values.astype(float) ** (-args.alpha)
    return (w / w.mean()).astype("float32")


results, t0 = {}, time.time()
base_pred = fit_predict()
results["baseline"], base_pt = evaluate(base_pred)
print(f"baseline done ({time.time() - t0:.0f}s) | C-index {results['baseline']['c_index']:.4f} "
      f"| saved model {meta['test_c_index']:.4f}")
assert abs(results["baseline"]["c_index"] - meta["test_c_index"]) < 0.003, "baseline does not match the saved model - stop"

rng = np.random.default_rng(42)
noise = {}
for dim in DIMS:
    parts = [(true[mk & known], base_pt[mk & known]) for mk in masks_f[dim]]
    gaps = []
    for _ in range(args.boot):
        f = [macro_f1(t[j], p[j]) for t, p in parts for j in [rng.integers(0, len(t), len(t))]]
        gaps.append(max(f) - min(f))
    noise[dim] = float(np.std(gaps))
print("noise (std of the Macro F1 gap from resampling):", {k: round(v, 3) for k, v in noise.items()})

for dim in DIMS:
    t1 = time.time()
    results[f"reweight_{dim}"], _ = evaluate(fit_predict(weights(dim)))
    print(f"reweight_{dim} done ({time.time() - t1:.0f}s)")

tbl = pd.DataFrame(results).T
cols = ["c_index", "macro_f1"] + [f"f1gap_{d}" for d in DIMS] + [f"cgap_{d}" for d in DIMS]
print("\n", tbl[cols].round(4).to_string())

b = results["baseline"]
print("\nVerdicts (Macro F1 gap; IMPROVED/WORSE only if the change is beyond 2 noise units):")
for v, r in results.items():
    if v == "baseline":
        continue
    parts = []
    for dim in DIMS:
        delta = r[f"f1gap_{dim}"] - b[f"f1gap_{dim}"]
        lab = "IMPROVED" if delta < -2 * noise[dim] else ("WORSE" if delta > 2 * noise[dim] else "within noise")
        parts.append(f"{dim} {delta:+.3f} {lab}")
    print(f"  {v}: " + " | ".join(parts))
    print(f"      side effects: C-index {r['c_index'] - b['c_index']:+.4f}, Macro F1 {r['macro_f1'] - b['macro_f1']:+.4f}")
json.dump({"state": slug, "alpha": args.alpha, "noise_std_f1_gap": noise, "results": results},
          open(f"reports/fairness_mitigation_{slug}.json", "w"), indent=2)
print(f"Saved reports/fairness_mitigation_{slug}.json")
