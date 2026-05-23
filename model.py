# model.py
# ML model training, evaluation, prediction, and persistence.
# I kept both Random Forest and XGBoost because they genuinely complement each other —
# RF is more stable and easier to trust, XGBoost usually squeezes out a bit more accuracy.
# Letting the user pick (and compare) them is intentional — it's a good learning exercise.
#
# In v3 I added save_model() and load_model() to support the GitHub Actions retraining
# pipeline — models get trained overnight by a scheduled script and saved to the models/
# folder, then the app loads them on startup instead of training in the browser.

import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

MODELS_DIR = "models"


def get_model_params(model_type: str) -> dict:
    """
    Default hyperparameters used when Optuna tuning is skipped.
    When tuning is enabled these get replaced by Optuna's best params.
    """
    if model_type == "Random Forest":
        return {
            "n_estimators":      200,
            "max_depth":          10,
            "min_samples_split":   5,
            "min_samples_leaf":    2,
            "random_state":       42,
            "n_jobs":             -1,   # use all CPU cores
        }
    else:  # XGBoost
        return {
            "n_estimators":    200,
            "max_depth":         6,
            "learning_rate":  0.05,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "random_state":     42,
            "verbosity":         0,
        }


def train_model(X_train, y_train, model_type: str = "Random Forest"):
    """Trains the selected model and returns (model, params)."""
    params = get_model_params(model_type)
    model  = RandomForestRegressor(**params) if model_type == "Random Forest" else XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model, params


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Runs the model on the test set and computes 4 standard regression metrics.

    Quick plain-English guide:
      RMSE  — average error in dollars (e.g. $18 means we're typically off by $18)
      MAE   — similar to RMSE but outlier errors don't dominate the average
      R²    — how much of the price variation the model explains (0 to 1, higher is better)
      MAPE  — error as a percentage of the actual price (below 1% is very good for gold)
    """
    preds = model.predict(X_test)
    return {
        "predictions": preds,
        "rmse":  round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae":   round(float(mean_absolute_error(y_test, preds)),          4),
        "r2":    round(float(r2_score(y_test, preds)),                     4),
        "mape":  round(float(np.mean(np.abs((np.array(y_test) - preds) / np.array(y_test))) * 100), 4),
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extracts feature importances from the trained model.
    Both RF and XGBoost expose .feature_importances_ so this works for both.
    I added readable labels so the chart in the UI doesn't show raw column names.
    """
    label_map = {
        "lag_1": "Yesterday's Price",    "lag_2": "2 Days Ago",
        "lag_3": "3 Days Ago",            "lag_5": "5 Days Ago",
        "lag_10": "10 Days Ago",
        "rolling_mean_7": "7-Day MA",     "rolling_mean_20": "20-Day MA",
        "rolling_mean_50": "50-Day MA",   "rolling_std_7": "7-Day Volatility",
        "rolling_std_20": "20-Day Volatility", "rolling_std_50": "50-Day Volatility",
        "momentum_5": "5-Day Momentum",   "pct_change_1": "1-Day Return",
        "pct_change_5": "5-Day Return",   "high_low_spread": "High-Low Range",
        "open_close_diff": "Open-Close Gap", "day_of_week": "Day of Week",
        "month": "Month",                 "year": "Year",
        "dxy_close": "USD Index (DXY)",   "dxy_change_1": "DXY Daily Change",
        "tnx_close": "10Y Treasury Yield","oil_close": "Oil Price",
        "oil_change_1": "Oil Daily Change","sp500_change_1": "S&P 500 Return",
        "vix_close": "VIX (Fear Index)",
    }

    # Categorise each feature for color-coding in the chart
    category_map = {
        "lag_": "Price History", "rolling_": "Trend & Volatility",
        "momentum_": "Momentum", "pct_change_": "Momentum",
        "high_low": "Intraday",  "open_close": "Intraday",
        "day_of_week": "Calendar", "month": "Calendar", "year": "Calendar",
        "dxy": "Macro", "tnx": "Macro", "oil": "Macro",
        "sp500": "Macro", "vix": "Macro",
    }

    def get_category(feat):
        for prefix, cat in category_map.items():
            if feat.startswith(prefix) or feat == prefix:
                return cat
        return "Other"

    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
        "Label":      [label_map.get(f, f) for f in feature_names],
        "Category":   [get_category(f) for f in feature_names],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df


def predict_next_day(model, df_features: pd.DataFrame, feature_names: list,
                     model_type: str, eval_rmse: float = 0.0):
    """
    Predicts the next trading day's gold price and computes a confidence interval.

    For Random Forest I use the variance across individual trees — each tree makes
    its own independent prediction and the spread tells us how uncertain the model is.
    For XGBoost I use the RMSE from evaluation as a proxy for uncertainty since
    XGBoost doesn't expose individual estimator predictions the same way.
    """
    last_row   = df_features[feature_names].iloc[-1:]
    prediction = float(model.predict(last_row)[0])

    if model_type == "Random Forest":
        tree_preds = np.array([t.predict(last_row)[0] for t in model.estimators_])
        std        = float(tree_preds.std())
    else:
        std = eval_rmse if eval_rmse > 0 else prediction * 0.005

    return prediction, prediction - 1.96 * std, prediction + 1.96 * std, std


def predict_multi_step(model, df_features: pd.DataFrame, feature_names: list,
                       model_type: str, eval_rmse: float = 0.0,
                       horizon: int = 7) -> pd.DataFrame:
    """
    Recursive multi-step forecasting — predicts the next `horizon` trading days.

    The approach: use the model to predict Day 1, then feed that prediction back
    as if it were real data to predict Day 2, and so on up to `horizon` days.

    The honest trade-off with recursive forecasting is that errors compound —
    each predicted price feeds into the next step as though it were real.
    I account for this by widening the confidence interval with each step,
    which is exactly how uncertainty should grow in a sequential forecast.

    The widening factor I use is sqrt(step) — a standard statistical approach
    for capturing how uncertainty grows over a forecast horizon. It's conservative
    enough to be honest without making the outer days look absurdly wide.

    Returns a DataFrame with columns:
        day         — step number (1 to horizon)
        date        — calendar date (skips weekends)
        predicted   — point estimate in USD
        lower       — lower bound of 95% confidence interval
        upper       — upper bound of 95% confidence interval
        change_usd  — dollar change from today's actual price
        change_pct  — percentage change from today's actual price
        std         — standard deviation used for this step's interval
    """
    # Start from the most recent real feature row
    last_features = df_features[feature_names].iloc[-1:].copy()
    current_close = float(df_features["Close"].iloc[-1])

    # Base uncertainty — step 1 uses this directly
    if model_type == "Random Forest":
        tree_preds = np.array([t.predict(last_features)[0] for t in model.estimators_])
        base_std   = float(tree_preds.std())
    else:
        base_std = eval_rmse if eval_rmse > 0 else current_close * 0.005

    results    = []
    prev_pred  = current_close
    feat_row   = last_features.copy()
    today      = datetime.now()

    for step in range(1, horizon + 1):
        # Skip weekends for the date label
        forecast_date = today + timedelta(days=step)
        while forecast_date.weekday() >= 5:   # 5=Sat, 6=Sun
            forecast_date += timedelta(days=1)

        # Make prediction from current feature row
        pred = float(model.predict(feat_row)[0])

        # Confidence interval widens with sqrt(step) — standard forecasting practice
        step_std = base_std * np.sqrt(step)
        lower    = pred - 1.96 * step_std
        upper    = pred + 1.96 * step_std

        change_usd = pred - current_close
        change_pct = change_usd / current_close * 100

        results.append({
            "day":        step,
            "date":       forecast_date.strftime("%a %b %d"),
            "predicted":  round(pred,        2),
            "lower":      round(lower,        2),
            "upper":      round(upper,        2),
            "change_usd": round(change_usd,   2),
            "change_pct": round(change_pct,   3),
            "std":        round(step_std,     2),
        })

        # Feed prediction back as next step's lag_1 and roll forward lag features
        # I update only the lag and momentum features since macro data stays the same
        if "lag_1" in feature_names:
            feat_row = feat_row.copy()

            # Shift lags forward: lag_2 ← lag_1, lag_3 ← lag_2, etc.
            for lag in [10, 5, 3, 2]:
                prev_col = f"lag_{lag - 1}" if lag > 2 else "lag_1"
                curr_col = f"lag_{lag}"
                if curr_col in feat_row.columns and prev_col in feat_row.columns:
                    feat_row[curr_col] = feat_row[prev_col].values

            # lag_1 becomes the prediction we just made
            feat_row["lag_1"] = pred

            # Update momentum and pct_change features
            if "momentum_5" in feat_row.columns and "lag_5" in feat_row.columns:
                feat_row["momentum_5"] = pred - float(feat_row["lag_5"].values[0])
            if "pct_change_1" in feat_row.columns:
                feat_row["pct_change_1"] = (pred - prev_pred) / prev_pred if prev_pred != 0 else 0
            if "pct_change_5" in feat_row.columns and "lag_5" in feat_row.columns:
                lag5_val = float(feat_row["lag_5"].values[0])
                feat_row["pct_change_5"] = (pred - lag5_val) / lag5_val if lag5_val != 0 else 0

            # Update calendar features for the forecasted date
            if "day_of_week" in feat_row.columns:
                feat_row["day_of_week"] = forecast_date.weekday()
            if "month" in feat_row.columns:
                feat_row["month"] = forecast_date.month
            if "year" in feat_row.columns:
                feat_row["year"] = forecast_date.year

        prev_pred = pred

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE — save and load pre-trained models
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, model_type: str, metrics: dict,
               feature_names: list, params: dict) -> str:
    """
    Saves a trained model and its metadata to the models/ folder.

    Files created:
      models/model_rf.pkl  or  models/model_xgb.pkl  — the trained model
      models/meta_rf.json  or  models/meta_xgb.json  — metrics, params, timestamp

    Returns the path to the saved model file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    suffix     = "rf" if model_type == "Random Forest" else "xgb"
    model_path = os.path.join(MODELS_DIR, f"model_{suffix}.pkl")
    meta_path  = os.path.join(MODELS_DIR, f"meta_{suffix}.json")

    # Save the model as a pickle file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata as JSON — strip the predictions array since it can't serialise
    clean_metrics = {k: v for k, v in metrics.items() if k != "predictions"}
    meta = {
        "model_type":    model_type,
        "metrics":       clean_metrics,
        "feature_names": list(feature_names),
        "params":        {k: str(v) for k, v in params.items()},
        "trained_at":    datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved → {model_path}")
    return model_path


def load_model(model_type: str) -> tuple:
    """
    Loads a pre-trained model and its metadata from the models/ folder.

    Returns:
        (model, metrics, feature_names, meta) if found
        (None, None, None, None) if no saved model exists
    """
    suffix     = "rf" if model_type == "Random Forest" else "xgb"
    model_path = os.path.join(MODELS_DIR, f"model_{suffix}.pkl")
    meta_path  = os.path.join(MODELS_DIR, f"meta_{suffix}.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta["metrics"], meta["feature_names"], meta


def list_saved_models() -> dict:
    """
    Checks which pre-trained model files exist in the models/ folder.
    Returns a dict with keys 'Random Forest' and 'XGBoost', values True/False.
    """
    return {
        "Random Forest": os.path.exists(os.path.join(MODELS_DIR, "model_rf.pkl")),
        "XGBoost":       os.path.exists(os.path.join(MODELS_DIR, "model_xgb.pkl")),
    }
