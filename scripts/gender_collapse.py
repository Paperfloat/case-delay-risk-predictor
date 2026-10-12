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
args = ap.parse_args()
args.cutoff = None
args.pending = None
args.workload = False
cfg = load_config(args)
slug = cfg["slug"]
meta = json.load(open(f"models/{slug}_aft_meta.json"))
cutoff = pd.Timestamp(meta["cutoff"])
CATS = meta["categorical_fields"]
GENDER = [c for c in CATS if c.startswith("female_")]
NOG = [c for c in CATS if c not in GENDER]
prev = json.load(open(f"reports/gender_ablation_{slug}.json"))
PATH = f"data/processed/{normalized_name(cfg)}"

df = pd.read_csv(PATH, dtype=str, usecols=CATS + ["date_of_filing", "date_of_decision", "event"])
df["date_of_filing"] = pd.to_datetime(df["date_of_filing"], errors="coerce")
df["date_of_decision"] = pd.to_datetime(df["date_of_decision"], errors="coerce")
df = df[df["date_of_filing"].notna()].copy()
for c in CATS:
    df[c] = df[c].fillna("missing")
decided = (df["event"] == "1") & (df["date_of_decision"] <= cutoff)
end = df["date_of_decision"].where(decided, cutoff)
d_all = (end - df["date_of_filing"]).dt.days.clip(lower=1).values.astype("float32")
o_all = decided.astype(int).values
year = df["date_of_filing"].dt.year.values

X2 = df[NOG].copy()
for g in GENDER:
    v = df[g].astype(str)
    X2[g] = np.where(v.str.startswith("-") | (v == "missing"), "unclear", "known")
print(f"[{slug}] collapsed values:", {g: X2[g].value_counts(normalize=True).round(3).to_dict() for g in GENDER})
X = pd.get_dummies(X2, dtype="uint8")
M, names = X.values.astype("float32"), list(X.columns)
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
          "aft_loss_distribution_scale": 1.2, "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}


def c_index(tr, va, te):
    def dm(i):
        m = xgb.DMatrix(M[i], feature_names=names)
        m.set_float_info("label_lower_bound", d_all[i])
        m.set_float_info("label_upper_bound", np.where(o_all[i] == 1, d_all[i], np.inf).astype("float32"))
        return m
    dtr, dva = dm(tr), dm(va)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    p = bst[: bst.best_iteration + 1].predict(xgb.DMatrix(M[te], feature_names=names))
    return float(concordance_index(d_all[te], p, o_all[te]))


idx = np.arange(len(df))
tr_all = np.where(year <= 2012)[0]
te_t = np.where(year == 2013)[0]
tr_t, va_t = train_test_split(tr_all, test_size=0.15, random_state=42, stratify=o_all[tr_all])
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=o_all)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=o_all[tr])
out = {"state": slug}
for label, key, split in [("time split", "time_split", (tr_t, va_t, te_t)), ("random split", "random_split", (tr, va, te))]:
    t0 = time.time()
    coll = c_index(*split)
    full, none = prev[key]["with"], prev[key]["without"]
    kept = (coll - none) / (full - none) if full > none else float("nan")
    out[key] = {"with_gender": full, "collapsed": round(coll, 4), "without": none, "share_of_gain_kept": round(kept, 2)}
    print(f"[{slug}] {label}: full {full:.4f} | collapsed (known/unclear) {coll:.4f} | none {none:.4f} "
          f"| share of the gain kept {kept:.0%} | {time.time() - t0:.0f}s", flush=True)
json.dump(out, open(f"reports/gender_collapse_{slug}.json", "w"), indent=2)
