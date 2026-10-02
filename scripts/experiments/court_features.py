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

STATES = ["delhi_survival", "odisha", "bihar"]
WINDOW = 90
CATS = ["type_name_normalized", "court_tier", "district_name", "female_defendant",
        "female_petitioner", "female_adv_def", "female_adv_pet"]
PARAMS = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
          "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}
rows = []


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


def dmat(M, i, dur, obs):
    d = xgb.DMatrix(M[i])
    d.set_float_info("label_lower_bound", dur[i])
    d.set_float_info("label_upper_bound", np.where(obs[i] == 1, dur[i], np.inf).astype("float32"))
    return d


def run_state(state):
    cfg = load_config(argparse.Namespace(state=state, pending=None))
    cutoff = pd.Timestamp(cfg["cutoff"])
    cols = CATS + ["date_of_filing", "date_of_decision", "event", "dist_code", "court_no", "court_name"]
    df = pd.read_csv(f"data/processed/{normalized_name(cfg)}", dtype=str, usecols=cols)
    filed = pd.to_datetime(df["date_of_filing"], errors="coerce")
    decided_on = pd.to_datetime(df["date_of_decision"], errors="coerce")
    keep = filed.notna().values
    df = df[keep].reset_index(drop=True)
    filed = filed[keep].reset_index(drop=True)
    decided_on = decided_on[keep].reset_index(drop=True)
    obs = ((df["event"] == "1") & (decided_on <= cutoff)).values.astype(int)
    end = decided_on.where(obs == 1, cutoff)
    dur = (end - filed).dt.days.clip(lower=1).values.astype("float32")

    key = (df["dist_code"].fillna("?") + "|" + df["court_no"].fillna("?")).values
    names = pd.DataFrame({"k": key, "n": df["court_name"].fillna("?")}).groupby("k")["n"].nunique()
    print(f"{state}: rows {len(df)}, courts {len(names)}, courts with one name across years: {(names == 1).mean():.0%}", flush=True)
    days = filed.values.astype("datetime64[D]").astype("int64")
    wl = workload(days, key)
    print(f"  workload: missing {np.isnan(wl).mean():.1%}, median {np.nanmedian(wl):.0f}, 90th pct {np.nanpercentile(wl, 90):.0f}", flush=True)

    base = pd.get_dummies(df[CATS].fillna("missing"), dtype="uint8").values.astype("float32")
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=obs)
    tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=obs[tr])
    for name, M in {"baseline": base, "+workload": np.column_stack([base, wl])}.items():
        t0 = time.time()
        dtr, dva, dte = dmat(M, tr, dur, obs), dmat(M, va, dur, obs), dmat(M, te, dur, obs)
        bst = xgb.train(PARAMS, dtr, num_boost_round=3000, evals=[(dva, "val")],
                        early_stopping_rounds=50, verbose_eval=False)
        rng = (0, bst.best_iteration + 1)
        c_va = concordance_index(dur[va], bst.predict(dva, iteration_range=rng), obs[va])
        c_te = concordance_index(dur[te], bst.predict(dte, iteration_range=rng), obs[te])
        share = float("nan")
        if name == "+workload":
            g = bst.get_score(importance_type="total_gain")
            share = g.get(f"f{M.shape[1] - 1}", 0.0) / sum(g.values())
        rows.append(dict(state=state, variant=name, rounds=bst.best_iteration + 1,
                         val_c=round(float(c_va), 4), test_c=round(float(c_te), 4),
                         workload_gain_share=round(share, 3), minutes=round((time.time() - t0) / 60, 1)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("experiments/court_features.csv", index=False)


for s in STATES:
    run_state(s)

res = pd.DataFrame(rows)
print("\nChange from adding court workload (same split, same settings):")
for s in STATES:
    a = res[(res.state == s) & (res.variant == "baseline")].iloc[0]
    b = res[(res.state == s) & (res.variant == "+workload")].iloc[0]
    print(f"{s}: baseline val {a.val_c} test {a.test_c} | +workload val {b.val_c - a.val_c:+.4f} test {b.test_c - a.test_c:+.4f} | workload share of gain {b.workload_gain_share:.1%}")
