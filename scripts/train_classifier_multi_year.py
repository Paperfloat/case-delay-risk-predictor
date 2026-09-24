import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv('data/processed/cases_2010_2013_delhi_features.csv')
df = df[df['days_to_disposition'].notna()].copy()
df['date_of_filing'] = pd.to_datetime(df['date_of_filing'], errors='coerce')
df['filing_month'] = df['date_of_filing'].dt.month.astype('Int64').astype(str)
print("Rows with valid target:", df.shape[0])

# Recompute tertile thresholds on the EXPANDED dataset, don't reuse 2012-only values
q33, q66 = df['days_to_disposition'].quantile([0.33, 0.66])
print(f"New thresholds — Low <= {q33:.0f}, Medium <= {q66:.0f}, High > {q66:.0f}")

def assign_risk_tier(days):
    if days <= q33:
        return 0
    elif days <= q66:
        return 1
    else:
        return 2

df['risk_tier'] = df['days_to_disposition'].apply(assign_risk_tier)

feature_cols = ['type_name_normalized', 'purpose_name_s', 'court_tier', 'district_name', 'female_defendant', 'female_petitioner', 'filing_month']
X_raw = df[feature_cols].copy()
y = df['risk_tier'].copy()
X = pd.get_dummies(X_raw, columns=feature_cols)
print("Feature matrix shape:", X.shape)

X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X, y, X_raw, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("case-delay-risk-predictor")

with mlflow.start_run(run_name="xgboost_risk_tier_classifier_2010_2013"):
    params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'eval_metric': 'mlogloss', 'random_state': 42}
    mlflow.log_params(params)
    mlflow.log_param("years", "2010-2013")
    mlflow.log_param("train_rows", X_train.shape[0])

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    mlflow.log_metric("macro_f1", macro_f1)
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    test_eval = raw_test.copy()
    test_eval['y_true'] = y_test
    test_eval['y_pred'] = y_pred

    for col in ['type_name_normalized', 'court_tier', 'district_name']:
        print(f"\n=== Subgroup Fairness Report: {col} ===")
        f1_scores = []
        for val, group in test_eval.groupby(col):
            if len(group) < 50:
                continue
            sub_f1 = f1_score(group['y_true'], group['y_pred'], average='macro')
            f1_scores.append(sub_f1)
            print(f"[{val}] (n={len(group)}): Macro F1 = {sub_f1:.4f}")
        if f1_scores:
            gap = max(f1_scores) - min(f1_scores)
            print(f"Max F1 Disparity Gap for {col}: {gap*100:.2f}%")
            mlflow.log_metric(f"fairness_f1_gap_{col}", gap)

    mlflow.xgboost.log_model(model, "model")
