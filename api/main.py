import json
import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Case Delay Risk Predictor")

model = joblib.load("models/risk_tier_model.joblib")
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)
with open("models/thresholds.json") as f:
    THRESHOLDS = json.load(f)

explainer = shap.TreeExplainer(model)

TIER_NAMES = {0: "Low", 1: "Medium", 2: "High"}

class CaseInput(BaseModel):
    type_name_normalized: str
    purpose_name_s: str
    court_tier: str
    district_name: str
    female_defendant: str
    female_petitioner: str

@app.post("/predict")
def predict(case: CaseInput):
    raw = pd.DataFrame([case.model_dump()])
    encoded = pd.get_dummies(raw)
    # Align to the exact training columns — critical: any column the model
    # was trained on but missing here must be filled with 0, and any column
    # here not seen in training must be dropped
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    pred = model.predict(encoded)[0]
    proba = model.predict_proba(encoded)[0]

    shap_values = explainer.shap_values(encoded)
    # shap_values shape depends on model type — for multiclass XGBoost this
    # is typically a list of arrays (one per class) or a 3D array; handle both
    if isinstance(shap_values, list):
        class_shap = shap_values[int(pred)][0]
    else:
        class_shap = shap_values[0, :, int(pred)]

    contributions = pd.Series(class_shap, index=FEATURE_COLUMNS)
    top5 = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)

    return {
        "risk_tier": TIER_NAMES[int(pred)],
        "confidence": float(proba[int(pred)]),
        "thresholds_days": THRESHOLDS,
        "top_contributing_factors": [
            {"feature": k, "shap_value": float(v)} for k, v in top5.items()
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
