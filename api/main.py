import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Case Delay Risk Predictor", version="2.0")

SLUGS = {"delhi": "delhi", "odisha": "orissa", "bihar": "bihar"}  # API state name -> file slug
CATS = ["type_name_normalized", "court_tier", "district_name", "female_defendant",
        "female_petitioner", "female_adv_def", "female_adv_pet"]
WORKLOAD = "court_filings_90d"
DB_PATH = "audit_log.db"
NOTE = ("expected_days is the survival model's predicted time to disposition. The model was validated "
        "for ranking cases (C-index); the day counts have not been checked for calibration, so use the "
        "tier for planning. Tiers are tertiles of predicted days among the state's held-out test cases.")


def options_from(columns):
    out = {c: [] for c in CATS}
    for col in columns:
        for field in CATS:
            if col.startswith(field + "_"):
                out[field].append(col[len(field) + 1:])
                break
    return out


MODELS = {}
for state, slug in SLUGS.items():
    meta_path = f"models/{slug}_aft_meta.json"
    if not os.path.exists(meta_path):
        print(f"WARNING: no model metadata for {state} ({meta_path}); state not served")
        continue
    meta = json.load(open(meta_path))
    booster = xgb.Booster()
    booster.load_model(f"models/{meta['model_file']}")
    columns = json.load(open(f"models/{meta['columns_file']}"))
    MODELS[state] = {"meta": meta, "booster": booster, "columns": columns,
                     "options": options_from(columns), "workload": WORKLOAD in columns}
if not MODELS:
    raise RuntimeError("no survival models found in models/")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS survival_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            state TEXT NOT NULL,
            input_json TEXT NOT NULL,
            expected_days REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            top_factors_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


class CaseInput(BaseModel):
    state: str
    type_name_normalized: str
    court_tier: str
    district_name: str
    female_defendant: str
    female_petitioner: str
    female_adv_def: str
    female_adv_pet: str
    court_filings_90d: Optional[int] = None  # only used by models trained with the workload feature


@app.post("/predict")
def predict(case: CaseInput):
    state = case.state.lower()
    if state not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown state '{case.state}'. Available: {sorted(MODELS)}")
    m = MODELS[state]
    row = {c: getattr(case, c) for c in CATS}
    enc = pd.get_dummies(pd.DataFrame([row]), dtype="uint8").reindex(columns=m["columns"], fill_value=0)
    enc = enc.astype("float32")
    warnings = [f"'{v}' is not a known {f} value for {state}; it was treated as unseen"
                for f, v in row.items() if v not in m["options"][f]]
    if m["workload"]:
        if case.court_filings_90d is None:
            warnings.append("court_filings_90d not supplied; the model treats it as missing")
        enc[WORKLOAD] = np.nan if case.court_filings_90d is None else float(case.court_filings_90d)
    elif case.court_filings_90d is not None:
        warnings.append("court_filings_90d ignored: this model does not use it")

    dm = xgb.DMatrix(enc.values, feature_names=m["columns"])
    days = float(m["booster"].predict(dm)[0])
    contribs = m["booster"].predict(dm, pred_contribs=True)[0][:-1]  # last value is the bias term
    top = np.argsort(-np.abs(contribs))[:5]
    factors = [{"feature": m["columns"][i], "shap_value": round(float(contribs[i]), 4),
                "effect": "longer" if contribs[i] > 0 else "shorter"} for i in top]

    t = m["meta"]["tier_thresholds"]
    tier = "Low" if days <= t["low_max_days"] else "Medium" if days <= t["medium_max_days"] else "High"
    result = {
        "state": state,
        "expected_days": round(days, 1),
        "risk_tier": tier,
        "tier_thresholds_days": {"low_max": t["low_max_days"], "medium_max": t["medium_max_days"]},
        "top_contributing_factors": factors,
        "explanation_scale": "log of days; positive values push the expected duration longer",
        "model_version": m["meta"]["model_version"],
        "warnings": warnings,
        "note": NOTE,
    }

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO survival_predictions (timestamp, model_version, state, input_json, expected_days, risk_tier, top_factors_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), result["model_version"], state,
         json.dumps(case.model_dump()), result["expected_days"], tier, json.dumps(factors)),
    )
    conn.commit()
    conn.close()
    return result


@app.get("/health")
def health():
    return {"status": "ok", "states": sorted(MODELS)}


@app.get("/states")
def states():
    return {s: {"model_version": m["meta"]["model_version"], "cutoff": m["meta"]["cutoff"],
                "tier_thresholds_days": m["meta"]["tier_thresholds"],
                "uses_court_filings_90d": m["workload"], "test_c_index": m["meta"]["test_c_index"]}
            for s, m in MODELS.items()}


@app.get("/options/{state}")
def options(state: str):
    state = state.lower()
    if state not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown state '{state}'. Available: {sorted(MODELS)}")
    return {"state": state, "fields": MODELS[state]["options"],
            "uses_court_filings_90d": MODELS[state]["workload"]}
