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
GENDER = [c for c in CATS if c.startswith("female_")]
NOG = [c for c in CATS if c not in GENDER]
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

Xf, Xn = pd.get_dummies(df[CATS], dtype="uint8"), pd.get_dummies(df[NOG], dtype="uint8")
assert list(Xf.columns) == json.load(open(f"models/{meta['columns_file']}")), "columns differ from the saved model"
Mf, Mn = Xf.values.astype("float32"), Xn.values.astype("float32")
nf, nn = list(Xf.columns), list(Xn.columns)
params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
          "aft_loss_distribution_scale": 1.2, "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}


def fit_predict(M, names, tr, va, te):
    def dm(i):
        m = xgb.DMatrix(M[i], feature_names=names)
        m.set_float_info("label_lower_bound", d_all[i])
        m.set_float_info("label_upper_bound", np.where(o_all[i] == 1, d_all[i], np.inf).astype("float32"))
        return m
    dtr, dva = dm(tr), dm(va)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    return bst[: bst.best_iteration + 1].predict(xgb.DMatrix(M[te], feature_names=names))


def compare(label, tr, va, te):
    t0 = time.time()
    pf, pn = fit_predict(Mf, nf, tr, va, te), fit_predict(Mn, nn, tr, va, te)
    d, o = d_all[te], o_all[te]
    cf, cn = concordance_index(d, pf, o), concordance_index(d, pn, o)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(args.boot):
        i = rng.integers(0, len(te), len(te))
        if o[i].sum() >= 10:
            diffs.append(concordance_index(d[i], pf[i], o[i]) - concordance_index(d[i], pn[i], o[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "gender fields HELP" if lo > 0 else ("gender fields HURT" if hi < 0 else "no clear difference")
    print(f"\n[{slug}] {label}: with gender {cf:.4f} | without {cn:.4f} | difference {cf - cn:+.4f} "
          f"(95% {lo:+.4f} to {hi:+.4f}) -> {verdict} | {time.time() - t0:.0f}s", flush=True)
    return pf, pn, {"with": round(float(cf), 4), "without": round(float(cn), 4), "diff": round(float(cf - cn), 4),
                    "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "verdict": verdict}


idx = np.arange(len(df))
tr_all = np.where(year <= 2012)[0]
te_t = np.where(year == 2013)[0]
tr_t, va_t = train_test_split(tr_all, test_size=0.15, random_state=42, stratify=o_all[tr_all])
_, _, res_time = compare("time split", tr_t, va_t, te_t)
tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=o_all)
tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=o_all[tr])
pf, pn, res_rand = compare("random split", tr, va, te)

print(f"\n[{slug}] by defendant / petitioner gender (random split; groups with at least 1,000 test cases)")
d, o = d_all[te], o_all[te]
rows = []
for name, p in [("with gender", pf), ("without gender", pn)]:
    tier = np.digitize(p, np.percentile(p, [100 / 3, 200 / 3]), right=True)
    for f in ["female_defendant", "female_petitioner"]:
        g = df[f].values[te]
        for val in pd.Series(g).value_counts().index:
            m = g == val
            if m.sum() >= 1000 and o[m].sum() >= 50:
                rows.append({"model": name, "field": f, "value": val, "n": int(m.sum()),
                             "share_High": round(float((tier[m] == 2).mean()), 3),
                             "c_index": round(float(concordance_index(d[m], p[m], o[m])), 4)})
print(pd.DataFrame(rows).to_string(index=False))
json.dump({"state": slug, "time_split": res_time, "random_split": res_rand, "by_gender": rows},
          open(f"reports/gender_ablation_{slug}.json", "w"), indent=2)
