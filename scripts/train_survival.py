import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--cutoff", default=None, help="override the cutoff in config/states.yaml")
ap.add_argument("--workload", action="store_true",
                help="add court_filings_90d: filings in the same court in the 90 days before filing")
args = ap.parse_args()
args.pending = None
cfg = load_config(args)
assert cfg["keep_pending"], "train_survival.py needs a state with keep_pending: true"
slug = cfg["slug"]
cutoff = pd.Timestamp(args.cutoff or cfg["cutoff"])
PATH = f"data/processed/{normalized_name(cfg)}"
WORKLOAD = "court_filings_90d"
WINDOW = 90
BOOT = 100

# filing-time features only (no purpose, disposition, listing dates, multiple_hearings)
CATS = ["type_name_normalized", "court_tier", "district_name",
        "female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet"]
extra = ["dist_code", "court_no"] if args.workload else []
df = pd.read_csv(PATH, dtype=str,
                 usecols=CATS + ["date_of_filing", "date_of_decision", "event"] + extra)
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
      f"censored {1 - df.obs.mean():.1%} | workload feature: {args.workload}")


def workload(days, keys):
    out = np.full(len(days), np.nan, dtype="float32")
    frame = pd.DataFrame({"k": keys, "d": days, "i": np.arange(len(days))})
    for _, g in frame.groupby("k"):
        d = np.sort(g["d"].values)
        hi = np.searchsorted(d, g["d"].values, side="left")
        lo = np.searchsorted(d, g["d"].values - WINDOW, side="left")
        out[g["i"].values] = hi - lo
    out[days < days.min() + WINDOW] = np.nan
    return out


X = pd.get_dummies(df[CATS], dtype="uint8")
feature_names = list(X.columns)
M = X.values.astype("float32")
if args.workload:
    days = df["date_of_filing"].values.astype("datetime64[D]").astype("int64")
    key = (df["dist_code"].fillna("?") + "|" + df["court_no"].fillna("?")).values
    M = np.column_stack([M, workload(days, key)])
    feature_names.append(WORKLOAD)
y_dur = df["duration"].values.astype("float32")
y_obs = df["obs"].values

idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_obs)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=y_obs[tr])


def dm(i):
    d = xgb.DMatrix(M[i], feature_names=feature_names)
    d.set_float_info("label_lower_bound", y_dur[i])
    d.set_float_info("label_upper_bound", np.where(y_obs[i] == 1, y_dur[i], np.inf).astype("float32"))
    return d


dtr, dva, dte = dm(tr), dm(va), dm(te)
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
          "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}
bst = xgb.train(params, dtr, num_boost_round=3000,
                evals=[(dtr, "train"), (dva, "val")],
                early_stopping_rounds=50, verbose_eval=200)
print("Best iteration:", bst.best_iteration)

final = bst[: bst.best_iteration + 1]  # keep only the trees up to the best iteration
pred = final.predict(dte)              # predicted time to disposition, in days
d_te, o_te = y_dur[te], y_obs[te]
ci = concordance_index(d_te, pred, o_te)
print(f"Test C-index: {ci:.4f}  (0.5 = random)")

rng = np.random.default_rng(42)


def boot_ci(d, p, o):
    vals = []
    for _ in range(BOOT):
        i = rng.integers(0, len(d), len(d))
        if o[i].sum() >= 10:
            vals.append(concordance_index(d[i], p[i], o[i]))
    return np.percentile(vals, [2.5, 97.5])


def audit(col):
    groups = df[col].iloc[te].values
    rows = []
    for g in pd.Series(groups).value_counts().index:
        m = groups == g
        if m.sum() >= 1000 and o_te[m].sum() >= 50:
            d, p, o = d_te[m], pred[m], o_te[m]
            lo, hi = boot_ci(d, p, o)
            rows.append((col, g, int(m.sum()), round(1 - float(o.mean()), 3),
                         round(concordance_index(d, p, o), 4), round(lo, 4), round(hi, 4)))
    t = pd.DataFrame(rows, columns=["dimension", "group", "n", "censored", "c_index", "ci_low", "ci_high"])
    t = t.sort_values("c_index")
    gap = float(t.c_index.max() - t.c_index.min()) if len(t) > 1 else float("nan")
    print(f"\n=== {col}: {len(t)} groups (n>=1000), best-worst C-index gap = {gap:.4f}")
    print(t[["group", "n", "censored", "c_index", "ci_low", "ci_high"]].to_string(index=False))
    return t, gap


tables, gaps = [], {}
for col in ["court_tier", "district_name", "type_name_normalized"]:
    t, g = audit(col)
    tables.append(t)
    gaps[col] = g

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
pd.concat(tables).to_csv(f"reports/{slug}_survival_fairness_{cutoff.date()}.csv", index=False)

model_file = f"{slug}_aft_{cutoff.date()}.json"
columns_file = f"{slug}_aft_feature_columns.json"
final.save_model(f"models/{model_file}")
json.dump(feature_names, open(f"models/{columns_file}", "w"))

# the saved file must reproduce the in-memory predictions exactly (this is what the API loads)
chk = xgb.Booster()
chk.load_model(f"models/{model_file}")
assert np.allclose(chk.predict(dte), pred, rtol=1e-4), "reloaded model differs from trained model"

q1, q2 = np.percentile(pred, [100 / 3, 200 / 3])
meta = {
    "state": cfg["state"], "slug": slug, "cutoff": str(cutoff.date()),
    "model_version": f"{slug}_aft_{cutoff.date()}" + ("_wl" if args.workload else ""),
    "model_file": model_file, "columns_file": columns_file,
    "categorical_fields": CATS, "workload": bool(args.workload), "workload_window_days": WINDOW,
    "tier_thresholds": {"low_max_days": round(float(q1), 1), "medium_max_days": round(float(q2), 1),
                        "basis": "tertiles of predicted days on the held-out test set"},
    "test_c_index": round(float(ci), 4), "best_iteration": int(bst.best_iteration),
    "n_rows": int(len(df)), "censored_rate": round(float(1 - df.obs.mean()), 4),
    "trained_at": datetime.now(timezone.utc).isoformat(),
}
json.dump(meta, open(f"models/{slug}_aft_meta.json", "w"), indent=2)
print("Saved", f"models/{model_file}", "| tier thresholds (days):", meta["tier_thresholds"])

try:
    import mlflow
    mlflow.set_experiment("case-delay-survival")
    with mlflow.start_run(run_name=f"{meta['model_version']}_v3"):
        mlflow.set_tags({"state": slug, "cutoff": str(cutoff.date()), "workload": str(args.workload)})
        mlflow.log_params(params)
        mlflow.log_metric("test_c_index", ci)
        mlflow.log_metric("best_iteration", bst.best_iteration)
        for k, v in gaps.items():
            mlflow.log_metric(f"c_index_gap_{k}", v)
except Exception as e:
    print("MLflow logging skipped:", e)
