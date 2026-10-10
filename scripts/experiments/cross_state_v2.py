import argparse
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

STATES = ["delhi_survival", "odisha", "bihar"]
LABEL = {"delhi_survival": "delhi", "odisha": "odisha", "bihar": "bihar"}
CATS = ["type_name_normalized", "court_tier", "female_defendant",
        "female_petitioner", "female_adv_def", "female_adv_pet"]
NO_TYPE = [c for c in CATS if c != "type_name_normalized"]
PARAMS = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
          "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}
BOOT = 60
rows = []
rng_boot = np.random.default_rng(42)


def load(state):
    cfg = load_config(argparse.Namespace(state=state, pending=None, cutoff=None, workload=False))
    cutoff = pd.Timestamp(cfg["cutoff"])
    df = pd.read_csv(f"data/processed/{normalized_name(cfg)}", dtype=str,
                     usecols=CATS + ["date_of_filing", "date_of_decision", "event"])
    filed = pd.to_datetime(df["date_of_filing"], errors="coerce")
    decided_on = pd.to_datetime(df["date_of_decision"], errors="coerce")
    keep = filed.notna().values
    df, filed, decided_on = df[keep], filed[keep], decided_on[keep]
    obs = ((df["event"] == "1") & (decided_on <= cutoff)).values.astype(int)
    end = decided_on.where(obs == 1, cutoff)
    dur = (end - filed).dt.days.clip(lower=1).values.astype("float32")
    cats = df[CATS].fillna("missing").reset_index(drop=True)
    cats["state"] = LABEL[state]
    idx = np.arange(len(cats))
    tr, te = train_test_split(idx, test_size=0.15, random_state=42, stratify=obs)
    tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=obs[tr])
    print(f"{state}: rows {len(cats)}, censored {1 - obs.mean():.1%}, cutoff {cutoff.date()}", flush=True)
    return dict(cats=cats, dur=dur, obs=obs, tr=tr, va=va, te=te)


def xy(D, states, part, cols):
    X = pd.concat([D[s]["cats"].iloc[D[s][part]] for s in states], ignore_index=True)
    dur = np.concatenate([D[s]["dur"][D[s][part]] for s in states])
    obs = np.concatenate([D[s]["obs"][D[s][part]] for s in states])
    return X[cols], dur, obs


def dmat(A, dur, obs):
    d = xgb.DMatrix(A)
    d.set_float_info("label_lower_bound", dur)
    d.set_float_info("label_upper_bound", np.where(obs == 1, dur, np.inf).astype("float32"))
    return d


def run(D, train_states, eval_states, cols, mode):
    t0 = time.time()
    Xtr, dtr_, otr = xy(D, train_states, "tr", cols)
    Xva, dva_, ova = xy(D, train_states, "va", cols)
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    dtr = dmat(enc.fit_transform(Xtr), dtr_, otr)
    dva = dmat(enc.transform(Xva), dva_, ova)
    bst = xgb.train(PARAMS, dtr, num_boost_round=3000, evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    rng = (0, bst.best_iteration + 1)
    for s in eval_states:
        Xte, dte_, ote = xy(D, [s], "te", cols)
        pred = bst.predict(xgb.DMatrix(enc.transform(Xte)), iteration_range=rng)
        c = concordance_index(dte_, pred, ote)
        vals = []
        for _ in range(BOOT):
            i = rng_boot.integers(0, len(dte_), len(dte_))
            if ote[i].sum() >= 10:
                vals.append(concordance_index(dte_[i], pred[i], ote[i]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append(dict(mode=mode, train_on="+".join(LABEL[x] for x in train_states),
                         eval_state=LABEL[s], rounds=bst.best_iteration + 1, test_c=round(float(c), 4),
                         ci_low=round(float(lo), 4), ci_high=round(float(hi), 4),
                         minutes=round((time.time() - t0) / 60, 1)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("experiments/cross_state_v2.csv", index=False)


D = {s: load(s) for s in STATES}
for s in STATES:
    run(D, [s], [s], CATS, "in-state")
run(D, STATES, STATES, CATS + ["state"], "pooled + state input")
run(D, STATES, STATES, CATS, "pooled shared (no state input)")
for s in STATES:
    run(D, [o for o in STATES if o != s], [s], CATS, "hold-out")
for s in STATES:
    run(D, [s], [s], NO_TYPE, "in-state, no case type")
for s in STATES:
    run(D, [o for o in STATES if o != s], [s], NO_TYPE, "hold-out, no case type")

res = pd.DataFrame(rows)
order = ["in-state", "pooled + state input", "pooled shared (no state input)", "hold-out",
         "in-state, no case type", "hold-out, no case type"]
piv = res.pivot_table(index="eval_state", columns="mode", values="test_c")[order]
print("\nTest C-index by evaluated state:")
print(piv.round(4).to_string())
print("\nPooled shared minus in-state:", (piv["pooled shared (no state input)"] - piv["in-state"]).round(4).to_dict())
print("Pooled + state input minus in-state:", (piv["pooled + state input"] - piv["in-state"]).round(4).to_dict())
print("Hold-out minus in-state, no case type:", (piv["hold-out, no case type"] - piv["in-state, no case type"]).round(4).to_dict())
