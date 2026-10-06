import argparse
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
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
HORIZONS = [365, 730, 1095, 1825]

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
feature_names = json.load(open(f"models/{meta['columns_file']}"))
assert list(X.columns) == feature_names, "feature columns differ from the saved model"
M = X.values.astype("float32")
d_all = df["duration"].values.astype("float32")
o_all = df["obs"].values
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=o_all)

bst = xgb.Booster()
bst.load_model(f"models/{meta['model_file']}")
pred = bst.predict(xgb.DMatrix(M[te], feature_names=feature_names))
d, o = d_all[te], o_all[te]
ci = concordance_index(d, pred, o)
print(f"[{slug}] C-index now {ci:.4f} | saved {meta['test_c_index']:.4f}")
assert abs(ci - meta["test_c_index"]) < 1e-3, "test split does not match training - stop"

t = meta["tier_thresholds"]
tier = np.digitize(pred, [t["low_max_days"], t["medium_max_days"]], right=True)
names = ["Low", "Medium", "High"]

rows = []
for k in range(3):
    m = tier == k
    km = KaplanMeierFitter().fit(d[m], o[m])
    row = {"state": slug, "tier": names[k], "n": int(m.sum()), "censored": round(float(1 - o[m].mean()), 3),
           "median_predicted_days": round(float(np.median(pred[m]))),
           "km_median_days": (round(float(km.median_survival_time_))
                              if np.isfinite(km.median_survival_time_) else np.nan)}
    for h in HORIZONS:
        row[f"decided_by_{h}d"] = round(1 - float(km.survival_function_at_times(h).iloc[0]), 3)
    rows.append(row)

out = pd.DataFrame(rows)
print(out.to_string(index=False))
ok = True
for h in HORIZONS:
    v = out[f"decided_by_{h}d"].values
    good = bool(v[0] > v[1] > v[2])
    ok &= good
    print(f"  decided by {h:>4} days: Low {v[0]:.3f} > Medium {v[1]:.3f} > High {v[2]:.3f} -> {'PASS' if good else 'FAIL'}")
print(f">>> {slug}: tiers ordered correctly at every horizon: {'YES' if ok else 'NO'}")
out.to_csv(f"reports/{slug}_survival_calibration_{cutoff.date()}.csv", index=False)
print("Saved", f"reports/{slug}_survival_calibration_{cutoff.date()}.csv")
