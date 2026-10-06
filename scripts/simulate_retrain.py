import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--ref-max-year", type=int, default=2011, help="cases filed up to this year = the old data")
ap.add_argument("--test-frac", type=float, default=0.3, help="share of newer cases held out for the comparison")
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
feature_names = list(X.columns)
assert feature_names == json.load(open(f"models/{meta['columns_file']}")), "feature columns differ from the saved model"
M = X.values.astype("float32")
d = df["duration"].values.astype("float32")
o = df["obs"].values
year = df["date_of_filing"].dt.year.values

early = np.where(year <= args.ref_max_year)[0]
late = np.where(year > args.ref_max_year)[0]
late_tr, late_te = train_test_split(late, test_size=args.test_frac, random_state=42, stratify=o[late])
print(f"[{slug}] old cases (filed <= {args.ref_max_year}): {len(early)} | newer cases: {len(late)} "
      f"(train {len(late_tr)}, held-out test {len(late_te)})")


def psi(ref, cur):
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    return float(((c - r) * np.log(c / r)).sum())


psi_case = psi(df["type_name_normalized"].iloc[early], df["type_name_normalized"].iloc[late])
print(f"Drift at this time slice: case-type PSI = {psi_case:.4f} (threshold 0.10) -> "
      f"{'RETRAIN TRIGGERED' if psi_case > 0.10 else 'no retrain needed'}")

params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
          "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}


def dm(i):
    m = xgb.DMatrix(M[i], feature_names=feature_names)
    m.set_float_info("label_lower_bound", d[i])
    m.set_float_info("label_upper_bound", np.where(o[i] == 1, d[i], np.inf).astype("float32"))
    return m


def fit(idx):
    t0 = time.time()
    tr, va = train_test_split(idx, test_size=0.15, random_state=42, stratify=o[idx])
    dtr, dva = dm(tr), dm(va)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    return bst[: bst.best_iteration + 1], time.time() - t0


def evaluate(bst, te):
    pred = bst.predict(xgb.DMatrix(M[te], feature_names=feature_names))
    dt, ot = d[te], o[te]
    ci = concordance_index(dt, pred, ot)
    lo, hi = np.percentile(dt, [100 / 3, 200 / 3])
    known = (ot == 1) | (dt > hi)
    true = np.where(dt <= lo, 0, np.where(dt <= hi, 1, 2))
    pt = np.digitize(pred, np.percentile(pred, [100 / 3, 200 / 3]), right=True)
    p, r, f, _ = precision_recall_fscore_support(true[known], pt[known], labels=[0, 1, 2],
                                                 average="macro", zero_division=0)
    return float(ci), float(p), float(r), float(f)


print("\nTraining the stale model (old cases only) ...")
stale, t_stale = fit(early)
print("Training the retrained model (old cases + most newer cases) ...")
fresh, t_fresh = fit(np.concatenate([early, late_tr]))

res = {}
for name, bst, secs in [("stale", stale, t_stale), ("retrained", fresh, t_fresh)]:
    ci, p, r, f = evaluate(bst, late_te)
    res[name] = {"c_index": round(ci, 4), "macro_precision": round(p, 4), "macro_recall": round(r, 4),
                 "macro_f1": round(f, 4), "train_seconds": round(secs)}
out = {"state": slug, "ref_max_year": args.ref_max_year, "case_type_psi": round(psi_case, 4),
       "old_rows": int(len(early)), "newer_train_rows": int(len(late_tr)), "newer_test_rows": int(len(late_te)),
       "results_on_held_out_newer_cases": res}
json.dump(out, open(f"reports/simulated_retrain_{slug}.json", "w"), indent=2)
print("\nResults on held-out NEWER cases (neither model saw them):")
print(pd.DataFrame(res).T.to_string())
print(f"\nChange from retraining: C-index {res['retrained']['c_index'] - res['stale']['c_index']:+.4f}, "
      f"Macro F1 {res['retrained']['macro_f1'] - res['stale']['macro_f1']:+.4f}")
print(f"Saved reports/simulated_retrain_{slug}.json")
