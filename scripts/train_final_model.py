import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv('data/processed/cases_2010_2013_delhi_features.csv')
df = df[df['days_to_disposition'].notna()].copy()
print("Rows with valid target:", df.shape[0])

q33, q66 = df['days_to_disposition'].quantile([0.33, 0.66])
print(f"Thresholds — Low <= {q33:.0f}, Medium <= {q66:.0f}, High > {q66:.0f}")

def assign_risk_tier(days):
    if days <= q33:
        return 0
    elif days <= q66:
        return 1
    else:
        return 2

df['risk_tier'] = df['days_to_disposition'].apply(assign_risk_tier)

feature_cols = ['type_name_normalized', 'purpose_name_s', 'court_tier', 'district_name', 'female_defendant', 'female_petitioner']
X_raw = df[feature_cols].copy()
y = df['risk_tier'].copy()
X = pd.get_dummies(X_raw, columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'eval_metric': 'mlogloss', 'random_state': 42}
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

from sklearn.metrics import f1_score
print("Macro F1 on holdout:", f1_score(y_test, model.predict(X_test), average='macro'))

# Save everything needed for serving, not just MLflow tracking
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/risk_tier_model.joblib')

with open('models/feature_columns.json', 'w') as f:
    json.dump(list(X.columns), f)

with open('models/thresholds.json', 'w') as f:
    json.dump({'low_max': float(q33), 'medium_max': float(q66)}, f)

print("Saved model, feature_columns.json, thresholds.json to models/")
