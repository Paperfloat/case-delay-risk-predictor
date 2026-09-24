import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1. Load data
df = pd.read_csv("data/processed/cases_2012_delhi_features.csv")
df = df[df["days_to_disposition"].notna()].copy()


# 2. Define target classes using quantile thresholds
def assign_risk_tier(days):
    if days <= 175:
        return 0  # Low Risk
    elif days <= 742:
        return 1  # Medium Risk
    else:
        return 2  # High Risk


df["risk_tier"] = df["days_to_disposition"].apply(assign_risk_tier)

# 3. Features and Target
feature_cols = [
    "type_name_normalized",
    "purpose_name_s",
    "court_tier",
    "district_name",
    "female_defendant",
    "female_petitioner",
]
X_raw = df[feature_cols].copy()
y = df["risk_tier"].copy()

# One-hot encode categoricals
X = pd.get_dummies(X_raw, columns=feature_cols)

# Train/Test Split (stratified on risk_tier)
X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X, y, X_raw, test_size=0.2, random_state=42, stratify=y
)

# 4. Train XGBClassifier
mlflow.set_experiment("case-delay-risk-predictor")

with mlflow.start_run(run_name="xgboost_risk_tier_classifier"):
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "eval_metric": "mlogloss",
        "random_state": 42,
    }
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Core Classification Metrics
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n--- Overall Metrics ---")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    mlflow.log_metric("macro_f1", macro_f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Low", "Medium", "High"]
        )
    )

    # 5. PRD 3D Fairness Audit (Case Type, Court Tier, Region/District)
    test_eval = raw_test.copy()
    test_eval["y_true"] = y_test
    test_eval["y_pred"] = y_pred

    def eval_subgroup_fairness(df_sub, group_col):
        print(f"\n=== Subgroup Fairness Report: {group_col} ===")
        f1_scores = []
        for val, group in df_sub.groupby(group_col):
            if len(group) < 50:  # Skip tiny subgroups
                continue
            sub_f1 = f1_score(
                group["y_true"], group["y_pred"], average="macro"
            )
            f1_scores.append(sub_f1)
            print(f"Subgroup [{val}] (n={len(group)}): Macro F1 = {sub_f1:.4f}")

        if f1_scores:
            max_gap = max(f1_scores) - min(f1_scores)
            print(f"Max F1 Disparity Gap for {group_col}: {max_gap * 100:.2f}%")
            mlflow.log_metric(f"fairness_f1_gap_{group_col}", max_gap)

    # Run checks across all 3 PRD dimensions
    for col in ["type_name_normalized", "court_tier", "district_name"]:
        eval_subgroup_fairness(test_eval, col)

    mlflow.xgboost.log_model(model, "model")
    print("\nClassifier run successfully logged to MLflow!")

    subgroup_data = []
for val, group in test_eval.groupby('type_name_normalized'):
    if len(group) < 50:
        continue
    sub_f1 = f1_score(group['y_true'], group['y_pred'], average='macro')
    subgroup_data.append({'type': val, 'n': len(group), 'f1': sub_f1})

subgroup_df = pd.DataFrame(subgroup_data)
print("\nCorrelation between subgroup size and F1:")
print(subgroup_df[['n', 'f1']].corr())
print("\nSmallest subgroups (most likely noisy):")
print(subgroup_df.sort_values('n').head(10))
print(df[df['type_name_normalized'] == 'lac']['days_to_disposition'].describe())
print(df[df['type_name_normalized'] == 'lac']['risk_tier'].value_counts() if 'risk_tier' in df.columns else 'risk_tier not in df')