import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

args = argparse.Namespace(state="delhi_survival", cutoff=None, pending=None, workload=False)
cfg = load_config(args)
meta0 = json.load(open("models/delhi_aft_meta.json"))
cutoff = pd.Timestamp(meta0["cutoff"])
CATS = meta0["categorical_fields"]
ACT_CATS = ["primary_act", "primary_section"]
TOP, BOOT = 30, 100
PATH = f"data/processed/{normalized_name(cfg)}"

df = pd.read_csv(PATH, dtype=str, usecols=CATS + ["ddl_case_id", "date_of_filing", "date_of_decision", "event"])
df["date_of_filing"] = pd.to_datetime(df["date_of_filing"], errors="coerce")
df["date_of_decision"] = pd.to_datetime(df["date_of_decision"], errors="coerce")
df = df[df["date_of_filing"].notna()].copy()
for c in CATS:
    df[c] = df[c].fillna("missing")
decided = (df["event"] == "1") & (df["date_of_decision"] <= cutoff)
end = df["date_of_decision"].where(decided, cutoff)
df["duration"] = (end - df["date_of_filing"]).dt.days.clip(lower=1)
df["obs"] = decided.astype(int)

acts = pd.read_csv("data/processed/acts_case_level.csv.gz", dtype=str,
                   usecols=["ddl_case_id", "state", "primary_act", "primary_section"])
acts = acts[acts["state"] == "delhi"].drop_duplicates("ddl_case_id")
n0 = len(df)
df = df.merge(acts[["ddl_case_id", "primary_act", "primary_section"]], on="ddl_case_id", how="inner")
for c in ACT_CATS:
    df[c] = df[c].astype("string").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
bad = ["", "missing", "nan"]
ok = df[ACT_CATS].notna().all(axis=1) & ~df["primary_act"].isin(bad) & ~df["primary_section"].isin(bad)
df = df[ok].reset_index(drop=True)
print(f"Delhi cases {n0} | with a known act and section: {len(df)} ({len(df) / n0:.1%})")
for c in ACT_CATS:
    keep = df[c].value_counts().head(TOP).index
    df[c] = df[c].where(df[c].isin(keep), "other").astype(str)

Xb = pd.get_dummies(df[CATS], dtype="uint8")
Xa = pd.get_dummies(df[CATS + ACT_CATS], dtype="uint8")
nb, na = list(Xb.columns), list(Xa.columns)
Mb, Ma = Xb.values.astype("float32"), Xa.values.astype("float32")
d_all = df["duration"].values.astype("float32")
o_all = df["obs"].values
year = df["date_of_filing"].dt.year.values
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
          "aft_loss_distribution_scale": 1.2, "tree_method": "hist", "learning_rate": 0.10,
          "max_depth": 6, "nthread": 4}


def fit(M, names, tr, va):
    def dm(i):
        m = xgb.DMatrix(M[i], feature_names=names)
        m.set_float_info("label_lower_bound", d_all[i])
        m.set_float_info("label_upper_bound", np.where(o_all[i] == 1, d_all[i], np.inf).astype("float32"))
        return m
    dtr, dva = dm(tr), dm(va)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    return bst[: bst.best_iteration + 1], int(bst.best_iteration)


def compare(label, tr, va, te):
    t0 = time.time()
    bb, _ = fit(Mb, nb, tr, va)
    ba, it = fit(Ma, na, tr, va)
    pb = bb.predict(xgb.DMatrix(Mb[te], feature_names=nb))
    pa = ba.predict(xgb.DMatrix(Ma[te], feature_names=na))
    d, o = d_all[te], o_all[te]
    cb, ca = concordance_index(d, pb, o), concordance_index(d, pa, o)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(te), len(te))
        if o[i].sum() >= 10:
            diffs.append(concordance_index(d[i], pa[i], o[i]) - concordance_index(d[i], pb[i], o[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "no clear difference")
    print(f"\n{label}: test rows {len(te)} | without acts {cb:.4f} -> with acts {ca:.4f} | "
          f"difference {ca - cb:+.4f} (95% {lo:+.4f} to {hi:+.4f}) -> {verdict} | {time.time() - t0:.0f}s")
    return {"without_acts": round(float(cb), 4), "with_acts": round(float(ca), 4),
            "diff": round(float(ca - cb), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "verdict": verdict, "test_rows": int(len(te))}, ba, pa, it


idx = np.arange(len(df))
tr_all = np.where(year <= 2012)[0]
te_t = np.where(year == 2013)[0]
tr_t, va_t = train_test_split(tr_all, test_size=0.15, random_state=42, stratify=o_all[tr_all])
time_res, _, _, _ = compare("TIME SPLIT (train 2010-12 filings, test 2013 filings)", tr_t, va_t, te_t)

tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=o_all)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=o_all[tr])
rand_res, final, pa, best_it = compare("RANDOM SPLIT", tr, va, te)

q1, q2 = np.percentile(pa, [100 / 3, 200 / 3])
model_file, cols_file = f"delhi_acts_aft_{cutoff.date()}.json", "delhi_acts_aft_feature_columns.json"
final.save_model(f"models/{model_file}")
json.dump(na, open(f"models/{cols_file}", "w"))
chk = xgb.Booster()
chk.load_model(f"models/{model_file}")
assert np.allclose(chk.predict(xgb.DMatrix(Ma[te], feature_names=na)), pa, rtol=1e-4), "reloaded model differs"
meta = {"state": "Delhi", "slug": "delhi_acts", "cutoff": str(cutoff.date()),
        "model_version": f"delhi_acts_aft_{cutoff.date()}", "model_file": model_file, "columns_file": cols_file,
        "categorical_fields": CATS + ACT_CATS, "act_top_n": TOP,
        "scope": "Delhi cases with a known act and section (optional model)",
        "tier_thresholds": {"low_max_days": round(float(q1), 1), "medium_max_days": round(float(q2), 1),
                            "basis": "tertiles of predicted days on this model's held-out test set"},
        "test_c_index": round(float(rand_res["with_acts"]), 4), "test_c_index_without_acts": rand_res["without_acts"],
        "time_split": time_res, "n_rows": int(len(df)), "best_iteration": best_it,
        "trained_at": datetime.now(timezone.utc).isoformat()}
json.dump(meta, open("models/delhi_acts_aft_meta.json", "w"), indent=2)
print("\nSaved models/delhi_acts_aft_meta.json | tier thresholds (days):", meta["tier_thresholds"])
print(">>> WIRE IN" if time_res["verdict"] == "HELPS" and rand_res["verdict"] == "HELPS"
      else ">>> DO NOT WIRE IN: the gain is not confirmed with act name and section name only")
