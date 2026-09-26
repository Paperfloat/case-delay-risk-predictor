import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import ClassificationPreset

df = pd.read_csv('data/processed/cases_2010_2013_delhi_features.csv')
df = df[df['days_to_disposition'].notna()].copy()

with open('models/thresholds.json') as f:
    THRESHOLDS = json.load(f)

def assign_risk_tier(days):
    if days <= THRESHOLDS['low_max']:
        return 'Low'
    elif days <= THRESHOLDS['medium_max']:
        return 'Medium'
    else:
        return 'High'

df['risk_tier'] = df['days_to_disposition'].apply(assign_risk_tier)

feature_cols = ['type_name_normalized', 'purpose_name_s', 'court_tier', 'district_name', 'female_defendant', 'female_petitioner']
X_raw = df[feature_cols].copy()
y = df['risk_tier'].copy()

with open('models/feature_columns.json') as f:
    FEATURE_COLUMNS = json.load(f)

X = pd.get_dummies(X_raw, columns=feature_cols).reindex(columns=FEATURE_COLUMNS, fill_value=0)

X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X, y, X_raw, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load('models/risk_tier_model.joblib')
TIER_NAMES = {0: 'Low', 1: 'Medium', 2: 'High'}
y_pred_raw = model.predict(X_test)
y_pred = [TIER_NAMES[p] for p in y_pred_raw]

from evidently import Dataset, DataDefinition, MulticlassClassification

eval_df = raw_test[['type_name_normalized', 'court_tier', 'district_name']].copy()
eval_df['target'] = y_test.values
eval_df['prediction'] = y_pred

data_definition = DataDefinition(
    classification=[MulticlassClassification(
        target="target",
        prediction_labels="prediction"
    )],
    categorical_columns=["type_name_normalized", "court_tier", "district_name", "target", "prediction"]
)

eval_dataset = Dataset.from_pandas(eval_df, data_definition=data_definition)

report = Report([ClassificationPreset()])
result = report.run(eval_dataset, None)
result.save_html('monitoring/fairness_report.html')
print("Saved HTML report to monitoring/fairness_report.html")