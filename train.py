# train.py
# Standalone training script designed to run as a scheduled GitHub Actions job.
# It can also be run manually from the terminal whenever I want to retrain
# outside of Streamlit — useful for testing or forcing a fresh model.
#
# What it does:
#   1. Fetches the latest gold + macro data from Yahoo Finance
#   2. Engineers all features
#   3. Optionally runs Optuna to find the best hyperparameters
#   4. Trains both Random Forest and XGBoost
#   5. Evaluates both models on the test set
#   6. Saves both models + metadata to the models/ folder
#   7. Logs both runs to MLflow (DagsHub if credentials are set)
#
# To run manually:
#   python train.py                  # default settings, no tuning
#   python train.py --tune           # run Optuna tuning first
#   python train.py --tune --trials 50
#   python train.py --period 5y      # use 5 years of data instead of max
#
# GitHub Actions runs this with default settings every weekday at 6am UTC.
# The saved model files (models/*.pkl + models/*.json) get committed back
# to the repo automatically by the workflow, which triggers a Streamlit
# Cloud redeploy so the live app always has yesterday's freshest model.

import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from data        import fetch_gold_data, fetch_macro_data, merge_with_macro, \
                        engineer_features, prepare_data
from model       import train_model, evaluate_model, get_feature_importance, \
                        save_model, get_model_params
from tuning      import run_tuning
from mlflow_utils import setup_mlflow, log_training_run


def parse_args():
    p = argparse.ArgumentParser(description="Gold Price Predictor — Training Script")
    p.add_argument("--period",  default="max",  help="yfinance data period (1y, 2y, 5y, max)")
    p.add_argument("--test",    default=0.20,   type=float, help="Test set fraction (default 0.20)")
    p.add_argument("--tune",    action="store_true",        help="Run Optuna tuning before training")
    p.add_argument("--trials",  default=30,     type=int,   help="Number of Optuna trials (default 30)")
    p.add_argument("--no-macro",action="store_true",        help="Skip macro features")
    return p.parse_args()


def train_and_save(model_type: str, X_train, X_test, y_train, y_test,
                   feature_names: list, args, df_features) -> dict:
    """
    Trains one model (RF or XGBoost), evaluates it, logs to MLflow, and saves to disk.
    Returns the metrics dict.
    """
    include_macro = not args.no_macro

    # ── Optuna tuning (optional) ───────────────────────────────────────────────
    if args.tune:
        print(f"  Running Optuna tuning ({args.trials} trials)…")
        best_params, best_val_rmse = run_tuning(model_type, X_train, y_train, args.trials)
        print(f"  Best val RMSE: ${best_val_rmse:.2f}")
        print(f"  Best params:   {best_params}")

        # Add fixed params that Optuna doesn't search
        best_params["random_state"] = 42
        if model_type == "Random Forest":
            best_params["n_jobs"] = -1
        else:
            best_params["verbosity"] = 0
        params = best_params
    else:
        params = get_model_params(model_type)

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"  Training {model_type}…")
    model, _ = train_model(X_train, y_train, model_type)

    # If we tuned, retrain with best params on full training set
    if args.tune:
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        if model_type == "Random Forest":
            model = RandomForestRegressor(**params)
        else:
            model = XGBRegressor(**params)
        model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  RMSE: ${metrics['rmse']:.2f}  MAE: ${metrics['mae']:.2f}"
          f"  R²: {metrics['r2']:.4f}  MAPE: {metrics['mape']:.2f}%")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    fi_df = get_feature_importance(model, feature_names)
    log_training_run(model, model_type, params, metrics, fi_df,
                     include_macro, args.period)

    # ── Save to disk ───────────────────────────────────────────────────────────
    save_model(model, model_type, metrics, feature_names, params)
    print(f"  Saved ✅")

    return metrics


def main():
    args = parse_args()
    include_macro = not args.no_macro

    print("=" * 60)
    print("  Gold Price Predictor — Training Pipeline")
    print(f"  Period: {args.period}  |  Tuning: {args.tune}"
          f"  |  Macro: {include_macro}")
    print("=" * 60)

    # ── Data ───────────────────────────────────────────────────────────────────
    print("\n[1/4] Fetching data…")
    df_raw = fetch_gold_data(args.period)

    if include_macro:
        macro_df  = fetch_macro_data(args.period)
        df_merged = merge_with_macro(df_raw, macro_df)
    else:
        df_merged = df_raw.copy()

    # ── Features ───────────────────────────────────────────────────────────────
    print("[2/4] Engineering features…")
    df_features = engineer_features(df_merged, include_macro=include_macro)
    X_train, X_test, y_train, y_test, _, feature_names = \
        prepare_data(df_features, args.test, include_macro)

    print(f"  Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"  Features:   {len(feature_names)}")

    # ── MLflow setup ───────────────────────────────────────────────────────────
    print("[3/4] Setting up MLflow…")
    backend = setup_mlflow()
    print(f"  Backend: {backend}")

    # ── Train both models ──────────────────────────────────────────────────────
    print("[4/4] Training models…")
    results = {}

    for model_type in ["Random Forest", "XGBoost"]:
        print(f"\n── {model_type} ──")
        results[model_type] = train_and_save(
            model_type, X_train, X_test, y_train, y_test,
            feature_names, args, df_features,
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete — Results Summary")
    print("=" * 60)
    for mt, m in results.items():
        print(f"  {mt:15s}  RMSE ${m['rmse']:.2f}  R² {m['r2']:.4f}")

    winner = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n  Best model: {winner} (lowest RMSE)")
    print("=" * 60)


if __name__ == "__main__":
    main()
