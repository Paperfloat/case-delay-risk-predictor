import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--low", type=float, default=175, help="true Low/Medium cutoff in days")
ap.add_argument("--high", type=float, default=742, help="true Medium/High cutoff in days")
ap.add_argument("--min-n", type=int, default=1000)
ap.add_argument("--auto", action="store_true",
                help="use this state's own tertiles of true days instead of --low/--high")
args = ap.parse_args()
args.cutoff = None
args.pending = None
args.workload = False
cfg = load_config(args)
slug = cfg["slug"]

meta = json.load(open(f"models/{slug}_aft_meta.json"))
assert not meta["workload"], "this report expects a model trained without the workload feature"
cutoff = pd.Timestamp(meta["cutoff"])
PATH = f"data/processed/{normalized_name(cfg)}"
CATS = meta["categorical_fields"]

# ---- rebuild the data exactly like train_survival.py ----
df = pd.read_csv(PATH, dtype=str,
                 usecols=CATS + ["date_of_filing", "date_of_decision", "event"])
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
feature_names = json.load(open(f"models/{meta['columns_file']}"))
assert list(X.columns) == feature_names, "feature columns differ from the saved model"
M = X.values.astype("float32")
y_dur = df["duration"].values.astype("float32")
y_obs = df["obs"].values

# same first split as training (same seed, same stratify)
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_obs)

bst = xgb.Booster()
bst.load_model(f"models/{meta['model_file']}")
pred = bst.predict(xgb.DMatrix(M[te], feature_names=feature_names))
d_te, o_te = y_dur[te], y_obs[te]

ci = concordance_index(d_te, pred, o_te)
print(f"[{slug}] C-index now {ci:.4f} | saved in meta {meta['test_c_index']:.4f}")
assert abs(ci - meta["test_c_index"]) < 1e-3, "test split does not match training - stop"

# ---- predicted tier: same cut points the API uses (tertiles of predicted days) ----
t = meta["tier_thresholds"]
pred_tier = np.digitize(pred, [t["low_max_days"], t["medium_max_days"]], right=True)

# ---- true tier. A pending case is only certain to be High if it has already
# waited longer than the High cutoff. Other pending cases are left out. ----
if args.auto:
    args.low, args.high = (float(v) for v in np.percentile(d_te, [100 / 3, 200 / 3]))
    print(f"Per-state true cutoffs (days): low {args.low:.0f}, high {args.high:.0f}")
known = (o_te == 1) | (d_te > args.high)
true_tier = np.where(d_te <= args.low, 0, np.where(d_te <= args.high, 1, 2))
print(f"Cases with a known true tier: {known.sum()} of {len(known)} "
      f"({known.mean():.1%}); the rest are pending and still unclear")
print("True tier counts     (Low, Medium, High):", np.bincount(true_tier[known], minlength=3))
print("Predicted tier counts (Low, Medium, High):", np.bincount(pred_tier[known], minlength=3))
print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(true_tier[known], pred_tier[known], labels=[0, 1, 2]))


def score(m):
    p, r, f, _ = precision_recall_fscore_support(
        true_tier[m], pred_tier[m], labels=[0, 1, 2], average="macro", zero_division=0)
    return round(float(p), 4), round(float(r), 4), round(float(f), 4)


rows = [("overall", "all", int(known.sum()), *score(known))]
for col in ["court_tier", "district_name", "type_name_normalized"]:
    groups = df[col].iloc[te].values
    sub = []
    for g in pd.Series(groups).value_counts().index:
        m = (groups == g) & known
        if m.sum() >= args.min_n:
            sub.append((col, g, int(m.sum()), *score(m)))
    rows += sub
    if len(sub) > 1:
        f1s = [r[5] for r in sub]
        print(f"\n{col}: {len(sub)} groups | best-worst Macro F1 gap = {max(f1s) - min(f1s):.4f}")

out = pd.DataFrame(rows, columns=["dimension", "group", "n", "macro_precision",
                                  "macro_recall", "macro_f1"])
print("\n", out.to_string(index=False))
os.makedirs("reports", exist_ok=True)
path = f"reports/{slug}_survival_tier_report_{cutoff.date()}" + ("_auto" if args.auto else "") + ".csv"
out.to_csv(path, index=False)
print("Saved", path)
