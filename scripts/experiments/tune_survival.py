import argparse
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

CONFIGS = {
    "base":     dict(lr=0.05, depth=6, scale=1.2, dist="normal",   rounds=1500),
    "lr10":     dict(lr=0.10, depth=6, scale=1.2, dist="normal",   rounds=3000),
    "d8":       dict(lr=0.10, depth=8, scale=1.2, dist="normal",   rounds=3000),
    "s08":      dict(lr=0.10, depth=6, scale=0.8, dist="normal",   rounds=3000),
    "s20":      dict(lr=0.10, depth=6, scale=2.0, dist="normal",   rounds=3000),
    "logistic": dict(lr=0.10, depth=6, scale=1.0, dist="logistic", rounds=3000),
}

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--configs", default="lr10,d8,s08,s20,logistic")
args = ap.parse_args()
args.pending = None
cfg = load_config(args)
slug = cfg["slug"]
cutoff = pd.Timestamp(cfg["cutoff"])
names = [n.strip() for n in args.configs.split(",")]
assert all(n in CONFIGS for n in names), f"unknown config; choose from {list(CONFIGS)}"

# data prep: identical to scripts/train_survival.py
CATS = ["type_name_normalized", "court_tier", "district_name",
        "female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet"]
df = pd.read_csv(f"data/processed/{normalized_name(cfg)}", dtype=str,
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
print(f"{cfg['state']} | cutoff {cutoff.date()} | rows {len(df)} | censored {1 - df.obs.mean():.1%}", flush=True)

X = pd.get_dummies(df[CATS], dtype="uint8")
y_dur, y_obs = df["duration"].values, df["obs"].values
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_obs)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=y_obs[tr])


def dm(i):
    d = xgb.DMatrix(X.iloc[i].values.astype("float32"), feature_names=list(X.columns))
    lo = y_dur[i].astype("float32")
    hi = np.where(y_obs[i] == 1, lo, np.inf).astype("float32")
    d.set_float_info("label_lower_bound", lo)
    d.set_float_info("label_upper_bound", hi)
    return d


dtr, dva, dte = dm(tr), dm(va), dm(te)
rows = []
for name in names:
    c = CONFIGS[name]
    params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
              "aft_loss_distribution": c["dist"], "aft_loss_distribution_scale": c["scale"],
              "tree_method": "hist", "learning_rate": c["lr"], "max_depth": c["depth"], "nthread": 4}
    t0 = time.time()
    bst = xgb.train(params, dtr, num_boost_round=c["rounds"],
                    evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    rng = (0, bst.best_iteration + 1)
    ci_va = concordance_index(y_dur[va], bst.predict(dva, iteration_range=rng), y_obs[va])
    ci_te = concordance_index(y_dur[te], bst.predict(dte, iteration_range=rng), y_obs[te])
    rows.append(dict(config=name, rounds_used=bst.best_iteration + 1, cap=c["rounds"],
                     hit_cap=bst.best_iteration + 1 >= c["rounds"] - 1,
                     val_c=round(ci_va, 4), test_c=round(ci_te, 4), minutes=round((time.time() - t0) / 60, 1)))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"experiments/{slug}_tuning.csv", index=False)

res = pd.DataFrame(rows).sort_values("val_c", ascending=False)
print("\nRanked by VALIDATION C-index (test shown for comparison only):")
print(res.to_string(index=False))
