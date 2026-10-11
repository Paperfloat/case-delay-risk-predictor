import argparse
import glob
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

HORIZONS = {"1": 365, "2": 730, "3": 1095, "5": 1825}
TIERS = ["Low", "Medium", "High"]
out = {}

for key, slug in [("delhi", "delhi"), ("odisha", "orissa"), ("bihar", "bihar")]:
    files = glob.glob(f"reports/{slug}_survival_calibration_*.csv")
    assert len(files) == 1, f"expected one calibration file for {slug}, found {files}"
    cal = pd.read_csv(files[0]).set_index("tier")
    out[key] = {}
    for t in TIERS:
        med = cal.loc[t, "km_median_days"]
        out[key][t] = {"decided_within_years": {y: round(float(cal.loc[t, f"decided_by_{d}d"]), 3) for y, d in HORIZONS.items()},
                       "median_years": None if pd.isna(med) else round(float(med) / 365, 1)}

# act-aware Delhi model: rebuild its test split exactly as train_delhi_acts.py did
cfg = load_config(argparse.Namespace(state="delhi_survival", cutoff=None, pending=None, workload=False))
meta0 = json.load(open("models/delhi_aft_meta.json"))
am = json.load(open("models/delhi_acts_aft_meta.json"))
cutoff = pd.Timestamp(meta0["cutoff"])
CATS = meta0["categorical_fields"]
ACT_CATS = ["primary_act", "primary_section"]
df = pd.read_csv(f"data/processed/{normalized_name(cfg)}", dtype=str,
                 usecols=CATS + ["ddl_case_id", "date_of_filing", "date_of_decision", "event"])
df["date_of_filing"] = pd.to_datetime(df["date_of_filing"], errors="coerce")
df["date_of_decision"] = pd.to_datetime(df["date_of_decision"], errors="coerce")
df = df[df["date_of_filing"].notna()].copy()
for c in CATS:
    df[c] = df[c].fillna("missing")
decided = (df["event"] == "1") & (df["date_of_decision"] <= cutoff)
end = df["date_of_decision"].where(decided, cutoff)
df["duration"] = (end - df["date_of_filing"]).dt.days.clip(lower=1)
df["obs"] = decided.astype(int)
acts = pd.read_csv("data/processed/acts_case_level.csv.gz", dtype=str, usecols=["ddl_case_id", "state", "primary_act", "primary_section"])
acts = acts[acts["state"] == "delhi"].drop_duplicates("ddl_case_id")
df = df.merge(acts[["ddl_case_id", "primary_act", "primary_section"]], on="ddl_case_id", how="inner")
for c in ACT_CATS:
    df[c] = df[c].astype("string").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
bad = ["", "missing", "nan"]
ok = df[ACT_CATS].notna().all(axis=1) & ~df["primary_act"].isin(bad) & ~df["primary_section"].isin(bad)
df = df[ok].reset_index(drop=True)
for c in ACT_CATS:
    keep = df[c].value_counts().head(am["act_top_n"]).index
    df[c] = df[c].where(df[c].isin(keep), "other").astype(str)
X = pd.get_dummies(df[CATS + ACT_CATS], dtype="uint8")
names = json.load(open(f"models/{am['columns_file']}"))
assert list(X.columns) == names, "act-aware feature columns differ from the saved model - stop"
M = X.values.astype("float32")
d_all = df["duration"].values.astype("float32")
o_all = df["obs"].values
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=o_all)
bst = xgb.Booster()
bst.load_model(f"models/{am['model_file']}")
pred = bst.predict(xgb.DMatrix(M[te], feature_names=names))
d, o = d_all[te], o_all[te]
ci = concordance_index(d, pred, o)
print(f"act-aware Delhi: C-index now {ci:.4f} | saved {am['test_c_index']}")
assert abs(ci - am["test_c_index"]) < 1.5e-3, "test split does not match training - stop"
th = am["tier_thresholds"]
tier = np.digitize(pred, [th["low_max_days"], th["medium_max_days"]], right=True)
out["delhi_acts"] = {}
for k, t in enumerate(TIERS):
    m = tier == k
    km = KaplanMeierFitter().fit(d[m], o[m])
    med = km.median_survival_time_
    out["delhi_acts"][t] = {
        "decided_within_years": {y: round(1 - float(km.survival_function_at_times(h).iloc[0]), 3) for y, h in HORIZONS.items()},
        "median_years": round(float(med) / 365, 1) if np.isfinite(med) else None}
json.dump(out, open("api/tier_outcomes.json", "w"), indent=1)
print("saved api/tier_outcomes.json")
for key in out:
    print(key, {t: (v["decided_within_years"]["1"], v["decided_within_years"]["3"], v["decided_within_years"]["5"], v["median_years"])
                for t, v in out[key].items()})
