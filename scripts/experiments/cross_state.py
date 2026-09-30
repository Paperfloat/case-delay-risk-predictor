import argparse
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lifelines.utils import concordance_index

sys.path.insert(0, "scripts/pipeline")
from common import load_config, normalized_name

STATES = ["delhi_survival", "odisha", "bihar"]
LABEL = {"delhi_survival": "delhi", "odisha": "odisha", "bihar": "bihar"}
CATS = ["type_name_normalized", "court_tier", "female_defendant",
        "female_petitioner", "female_adv_def", "female_adv_pet"]
PARAMS = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
          "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
          "tree_method": "hist", "learning_rate": 0.10, "max_depth": 6, "nthread": 4}
rows = []


def load(state):
    cfg = load_config(argparse.Namespace(state=state, pending=None))
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


def xy(D, states, part, use_state):
    X = pd.concat([D[s]["cats"].iloc[D[s][part]] for s in states], ignore_index=True)
    dur = np.concatenate([D[s]["dur"][D[s][part]] for s in states])
    obs = np.concatenate([D[s]["obs"][D[s][part]] for s in states])
    return X[CATS + (["state"] if use_state else [])], dur, obs


def dmat(A, dur, obs):
    d = xgb.DMatrix(A)
    d.set_float_info("label_lower_bound", dur)
    d.set_float_info("label_upper_bound", np.where(obs == 1, dur, np.inf).astype("float32"))
    return d


def run(D, train_states, eval_states, use_state, mode):
    t0 = time.time()
    Xtr, dtr_, otr = xy(D, train_states, "tr", use_state)
    Xva, dva_, ova = xy(D, train_states, "va", use_state)
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    dtr = dmat(enc.fit_transform(Xtr), dtr_, otr)
    dva = dmat(enc.transform(Xva), dva_, ova)
    bst = xgb.train(PARAMS, dtr, num_boost_round=3000,
                    evals=[(dtr, "train"), (dva, "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    rng = (0, bst.best_iteration + 1)
    for s in eval_states:
        Xte, dte_, ote = xy(D, [s], "te", use_state)
        pred = bst.predict(xgb.DMatrix(enc.transform(Xte)), iteration_range=rng)
        c = concordance_index(dte_, pred, ote)
        rows.append(dict(mode=mode, train_on="+".join(LABEL[x] for x in train_states),
                         eval_state=LABEL[s], rounds=bst.best_iteration + 1,
                         test_c=round(float(c), 4), minutes=round((time.time() - t0) / 60, 1)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("experiments/cross_state.csv", index=False)


D = {s: load(s) for s in STATES}
for s in STATES:
    run(D, [s], [s], False, "in-state")
run(D, STATES, STATES, True, "pooled")
for s in STATES:
    run(D, [o for o in STATES if o != s], [s], False, "hold-out")

res = pd.DataFrame(rows)
print("\nTest C-index by evaluated state:")
print(res.pivot_table(index="eval_state", columns="mode", values="test_c")[["in-state", "pooled", "hold-out"]].to_string())
