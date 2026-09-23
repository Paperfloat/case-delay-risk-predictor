import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/cases_2012_delhi_features.csv')

# Drop rows with no valid target (corrupted dates, negative durations already NA)
df = df[df['days_to_disposition'].notna()].copy()
print("Rows with valid target:", df.shape[0])

# Filing-time-only feature set — no leakage
feature_cols = ['type_name_normalized', 'purpose_name_s', 'court_tier', 'district_name', 'female_defendant', 'female_petitioner']
target_col = 'days_to_disposition'

X = df[feature_cols].copy()
y = df[target_col].copy()

# One-hot encode all categorical features
X = pd.get_dummies(X, columns=feature_cols)
print("Feature matrix shape after encoding:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, r2_score

mlflow.set_experiment("case-delay-risk-predictor")

with mlflow.start_run(run_name="baseline_filing_time_only"):
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    }
    mlflow.log_params(params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.xgboost.log_model(model, "model")

    print(f"MAE: {mae:.2f} days")
    print(f"R2: {r2:.4f}")

    # Naive baseline for comparison: always predict the median
    naive_pred = [y_train.median()] * len(y_test)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    mlflow.log_metric("naive_baseline_mae", naive_mae)
    print(f"Naive baseline MAE (always predict median): {naive_mae:.2f} days")

# Feature importance — which categories matter most
importances = pd.Series(model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(20)
print("\nTop 20 most important features:")
print(top_features)

# --- Subgroup fairness check: MAE by district, court tier, case type ---
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['predicted'] = y_pred
test_df['abs_error'] = abs(test_df['actual'] - test_df['predicted'])

# Reconstruct original categorical columns for grouping (one-hot columns → back to category)
orig_test = df.loc[X_test.index, ['district_name', 'court_tier', 'female_defendant']]
test_df = test_df.join(orig_test)

print("\n--- MAE by district ---")
print(test_df.groupby('district_name')['abs_error'].agg(['mean', 'count']))

print("\n--- MAE by court tier ---")
print(test_df.groupby('court_tier')['abs_error'].agg(['mean', 'count']))

print("\n--- MAE by female_defendant ---")
print(test_df.groupby('female_defendant')['abs_error'].agg(['mean', 'count']))

# Correlation between subgroup size and error
district_stats = test_df.groupby('district_name')['abs_error'].agg(['mean', 'count'])
print("\n--- Correlation (District Count vs Mean Error) ---")
print(district_stats.corr())

