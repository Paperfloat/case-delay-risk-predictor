import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

import argparse
sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--cutoff", default=None, help="override the cutoff in config/states.yaml")
args = ap.parse_args()
args.pending = None
cfg = load_config(args)
assert cfg["keep_pending"], "train_survival.py needs a state with keep_pending: true"
slug = cfg["slug"]
cutoff = pd.Timestamp(args.cutoff or cfg["cutoff"])
PATH = f"data/processed/{normalized_name(cfg)}"

# filing-time features only
CATS = ["type_name_normalized", "court_tier", "district_name",
        "female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet"]
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
print(f"Cutoff {cutoff.date()} | rows {len(df)} | decided {df.obs.mean():.1%} | "
      f"censored {1 - df.obs.mean():.1%}")

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
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
          "tree_method": "hist", "learning_rate": 0.05, "max_depth": 6, "nthread": 4}
bst = xgb.train(params, dtr, num_boost_round=1500,
                evals=[(dtr, "train"), (dva, "val")],
                early_stopping_rounds=50, verbose_eval=100)
print("Best iteration:", bst.best_iteration)

pred = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
d_te, o_te = y_dur[te], y_obs[te]
ci = concordance_index(d_te, pred, o_te)
print(f"\nTest C-index: {ci:.4f}  (0.5 = random; previous 500-round run: 0.7307)")

def audit(col):
    groups = df[col].iloc[te].values
    rows = []
    for g in pd.Series(groups).value_counts().index:
        m = groups == g
        if m.sum() >= 1000 and o_te[m].sum() >= 50:
            rows.append((col, g, int(m.sum()), round(1 - float(o_te[m].mean()), 3),
                         round(concordance_index(d_te[m], pred[m], o_te[m]), 4)))
    t = pd.DataFrame(rows, columns=["dimension", "group", "n", "censored", "c_index"])
    t = t.sort_values("c_index")
    gap = float(t.c_index.max() - t.c_index.min()) if len(t) > 1 else float("nan")
    print(f"\n=== {col}: {len(t)} groups (n>=1000), best-worst C-index gap = {gap:.4f}")
    print(t[["group", "n", "censored", "c_index"]].to_string(index=False))
    return t, gap

tables, gaps = [], {}
for col in ["court_tier", "district_name", "type_name_normalized"]:
    t, g = audit(col)
    tables.append(t)
    gaps[col] = g

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
pd.concat(tables).to_csv(f"reports/{slug}_survival_fairness_{cutoff.date()}.csv", index=False)
out = f"models/{slug}_aft_{cutoff.date()}.json"
bst.save_model(out)
json.dump(list(X.columns), open(f"models/{slug}_aft_feature_columns.json", "w"))
print("\nSaved", out)

try:
    import mlflow
    mlflow.set_experiment("case-delay-survival")
    with mlflow.start_run(run_name=f"{slug}_aft_{cutoff.date()}_v2"):
        mlflow.set_tags({"state": slug, "cutoff": str(cutoff.date())})
        mlflow.log_params(params)
        mlflow.log_metric("test_c_index", ci)
        mlflow.log_metric("best_iteration", bst.best_iteration)
        for k, v in gaps.items():
            mlflow.log_metric(f"c_index_gap_{k}", v)
except Exception as e:
    print("MLflow logging skipped:", e)
