# mlflow_utils.py
# MLflow experiment tracking with optional DagsHub remote backend.
#
# By default this logs to a local mlruns/ folder which works perfectly
# for local development. When deployed to Streamlit Cloud, you can point it
# at DagsHub (free) by adding credentials to .streamlit/secrets.toml —
# that gives you a persistent tracking UI that survives app restarts.
#
# DagsHub setup: https://dagshub.com — create a free account, create a repo
# that matches your GitHub repo name, then grab the MLflow tracking URI
# from the Remote tab. Add those details to secrets.toml and it just works.

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import os

EXPERIMENT_NAME = "gold_price_prediction"


def setup_mlflow() -> str:
    """
    Initialises MLflow. Tries DagsHub first (if secrets are configured),
    falls back to local mlruns/ folder otherwise.
    Returns a string describing which backend is active.
    """
    try:
        import streamlit as st
        username = st.secrets.get("mlflow", {}).get("dagshub_username", "")
        repo     = st.secrets.get("mlflow", {}).get("dagshub_repo",     "")
        token    = st.secrets.get("mlflow", {}).get("dagshub_token",    "")

        if username and repo and token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(f"https://dagshub.com/{username}/{repo}.mlflow")
            mlflow.set_experiment(EXPERIMENT_NAME)
            return "dagshub"
    except Exception:
        pass

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    return "local"


def log_training_run(
    model,
    model_type: str,
    params: dict,
    metrics: dict,
    feature_importance_df: pd.DataFrame,
    include_macro: bool = True,
    period: str = "max",
) -> str:
    """
    Logs a complete training run to MLflow. Every time the model is retrained
    a new run is created so I can track how metrics change over time and
    compare Random Forest vs XGBoost side by side.

    Saves: hyperparameters, RMSE/MAE/R²/MAPE, the trained model artifact,
    feature importance CSV, and some metadata about the training config.
    """
    setup_mlflow()
    run_name = f"{model_type.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Config metadata
            mlflow.log_param("model_type",     model_type)
            mlflow.log_param("include_macro",  include_macro)
            mlflow.log_param("data_period",    period)

            # Hyperparameters
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Evaluation metrics
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae",  metrics["mae"])
            mlflow.log_metric("r2",   metrics["r2"])
            mlflow.log_metric("mape", metrics["mape"])

            # Model artifact
            if model_type == "Random Forest":
                mlflow.sklearn.log_model(model, "model",
                                         registered_model_name="GoldPricePredictor")
            else:
                mlflow.xgboost.log_model(model, "model",
                                         registered_model_name="GoldPricePredictor")

            # Feature importance CSV
            tmp = "_fi_tmp.csv"
            feature_importance_df.to_csv(tmp, index=False)
            mlflow.log_artifact(tmp, artifact_path="analysis")
            os.remove(tmp)

            return run.info.run_id

    except Exception as e:
        print(f"MLflow logging error: {e}")
        return "logging-failed"


def get_run_history() -> pd.DataFrame:
    """Retrieves all past runs from the active MLflow backend, newest first."""
    setup_mlflow()
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            order_by=["start_time DESC"],
        )
        if runs.empty:
            return pd.DataFrame()

        col_map = {
            "run_id":             "Run ID",
            "start_time":         "Timestamp",
            "params.model_type":  "Model",
            "params.data_period": "Period",
            "metrics.rmse":       "RMSE ($)",
            "metrics.mae":        "MAE ($)",
            "metrics.r2":         "R²",
            "metrics.mape":       "MAPE (%)",
        }
        available = {k: v for k, v in col_map.items() if k in runs.columns}
        df = runs[list(available.keys())].rename(columns=available)

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        if "Run ID" in df.columns:
            df["Run ID"] = df["Run ID"].str[:8] + "..."
        for col in ["RMSE ($)", "MAE ($)", "R²", "MAPE (%)"]:
            if col in df.columns:
                df[col] = df[col].round(4)

        return df.reset_index(drop=True)
    except Exception as e:
        print(f"MLflow history error: {e}")
        return pd.DataFrame()


def get_best_run() -> dict:
    """Returns metrics from the best run (lowest RMSE) for the sidebar badge."""
    setup_mlflow()
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            order_by=["metrics.rmse ASC"],
        )
        if runs.empty:
            return {}
        b = runs.iloc[0]
        return {
            "model_type": b.get("params.model_type", "N/A"),
            "rmse":       round(float(b.get("metrics.rmse", 0)), 4),
            "mae":        round(float(b.get("metrics.mae",  0)), 4),
            "r2":         round(float(b.get("metrics.r2",   0)), 4),
            "run_id":     str(b.get("run_id", ""))[:8],
        }
    except:
        return {}
