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
ap.add_argument("--top", type=int, default=30)
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

acts = pd.read_csv("data/processed/acts_case_level.csv.gz", dtype=str)
acts = acts[acts["state"] == slug].drop_duplicates("ddl_case_id")
n0 = len(df)
df = df.merge(acts[["ddl_case_id", "criminal_act", "bailable_ipc", "ipc_sections", "primary_act", "primary_section"]],
              on="ddl_case_id", how="left", indicator=True)
assert len(df) == n0, "join changed the row count"
has_act = (df["_merge"] == "both").values
print(f"[{slug}] rows {n0} | with act info {has_act.mean():.1%}")


def top_other(s, n, none_label):
    s = s.astype("string").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.mask(s.isin(["", "missing", "nan"])).fillna(none_label)
    keep = s[s != none_label].value_counts().head(n).index
    return s.where(s.isin(keep) | (s == none_label), "other").astype(str)


act_cat = pd.DataFrame({"primary_act": top_other(df["primary_act"], args.top, "no_act"),
                        "primary_section": top_other(df["primary_section"], args.top, "no_section")})
A = pd.get_dummies(act_cat, dtype="uint8")
num = df[["criminal_act", "bailable_ipc", "ipc_sections"]].apply(pd.to_numeric, errors="coerce").astype("float32")
num["has_act_info"] = has_act.astype("float32")

X = pd.get_dummies(df[CATS], dtype="uint8")
names_base = list(X.columns)
assert names_base == json.load(open(f"models/{meta['columns_file']}")), "feature columns differ from the saved model"
Mb = X.values.astype("float32")
Ma = np.column_stack([Mb, A.values.astype("float32"), num.values])
names_acts = names_base + list(A.columns) + list(num.columns)
print(f"features: baseline {Mb.shape[1]} | with acts {Ma.shape[1]}")

y_dur = df["duration"].values.astype("float32")
y_obs = df["obs"].values
year = df["date_of_filing"].dt.year.values
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
          "aft_loss_distribution_scale": 1.2, "tree_method": "hist", "learning_rate": 0.10,
          "max_depth": 6, "nthread": 4}


def fit_predict(M, names, tr, va, te):
    def dm(i):
        m = xgb.DMatrix(M[i], feature_names=names)
        m.set_float_info("label_lower_bound", y_dur[i])
        m.set_float_info("label_upper_bound", np.where(y_obs[i] == 1, y_dur[i], np.inf).astype("float32"))
        return m
    dtr, dva = dm(tr), dm(va)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    return bst[: bst.best_iteration + 1].predict(xgb.DMatrix(M[te], feature_names=names))


def compare(label, tr, va, te):
    t0 = time.time()
    pb = fit_predict(Mb, names_base, tr, va, te)
    pa = fit_predict(Ma, names_acts, tr, va, te)
    d, o = y_dur[te], y_obs[te]
    cb, ca = concordance_index(d, pb, o), concordance_index(d, pa, o)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(args.boot):
        i = rng.integers(0, len(te), len(te))
        if o[i].sum() >= 10:
            diffs.append(concordance_index(d[i], pa[i], o[i]) - concordance_index(d[i], pb[i], o[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    h = has_act[te]
    sub = {}
    for nm, m in [("with_act_info", h), ("without_act_info", ~h)]:
        if m.sum() > 500 and o[m].sum() > 50:
            sub[nm] = [round(concordance_index(d[m], pb[m], o[m]), 4), round(concordance_index(d[m], pa[m], o[m]), 4)]
    verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "no clear difference")
    print(f"\n[{slug}] {label}: test rows {len(te)} | baseline {cb:.4f} -> with acts {ca:.4f} "
          f"| difference {ca - cb:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}) -> {verdict}")
    print(f"   by subset (baseline, with acts): {sub} | {time.time() - t0:.0f}s")
    return {"baseline": round(cb, 4), "with_acts": round(ca, 4), "diff": round(ca - cb, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4), "verdict": verdict, "by_subset": sub,
            "test_rows": int(len(te))}


idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_obs)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=y_obs[tr])
res = {"state": slug, "with_act_info_share": round(float(has_act.mean()), 3)}
res["random_split"] = compare("random split", tr, va, te)
print(f"   (saved model's random-split C-index for reference: {meta['test_c_index']})")

tr_all = np.where(year <= 2012)[0]
te_t = np.where(year == 2013)[0]
tr_t, va_t = train_test_split(tr_all, test_size=0.15, random_state=42, stratify=y_obs[tr_all])
res["time_split"] = compare("time split (train 2010-12 filings, test 2013 filings)", tr_t, va_t, te_t)
json.dump(res, open(f"reports/acts_experiment_{slug}.json", "w"), indent=2)
print(f"Saved reports/acts_experiment_{slug}.json")
