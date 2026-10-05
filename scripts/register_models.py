import json
import sys

import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("case-delay-survival-registry")
client = MlflowClient()

for slug in ["delhi", "orissa", "bihar"]:
    meta = json.load(open(f"models/{slug}_aft_meta.json"))
    name = f"case-delay-aft-{slug}"
    bst = xgb.Booster()
    bst.load_model(f"models/{meta['model_file']}")

    with mlflow.start_run(run_name=meta["model_version"]) as run:
        mlflow.set_tags({"state": slug, "cutoff": meta["cutoff"], "workload": str(meta["workload"])})
        mlflow.log_metric("test_c_index", meta["test_c_index"])
        mlflow.log_metric("censored_rate", meta["censored_rate"])
        mlflow.log_param("n_rows", meta["n_rows"])
        mlflow.log_param("best_iteration", meta["best_iteration"])
        mlflow.log_artifact(f"models/{slug}_aft_meta.json")
        mlflow.log_artifact(f"models/{meta['columns_file']}")
        try:
            mlflow.xgboost.log_model(bst, name="model")           # MLflow 3.x
        except TypeError:
            mlflow.xgboost.log_model(bst, artifact_path="model")  # MLflow 2.x
        run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, "champion", mv.version)
    print(f"{name}: version {mv.version} registered, alias 'champion' set "
          f"(C-index {meta['test_c_index']})")

print("\nRegistered models:")
for rm in client.search_registered_models():
    print(" -", rm.name)
