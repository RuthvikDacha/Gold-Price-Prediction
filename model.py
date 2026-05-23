# model.py
# ML model training, evaluation, and prediction.
# I kept both Random Forest and XGBoost because they genuinely complement each other —
# RF is more stable and easier to trust, XGBoost usually squeezes out a bit more accuracy.
# Letting the user pick (and compare) them is intentional — it's a good learning exercise.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def get_model_params(model_type: str) -> dict:
    """
    Hyperparameters I tuned manually for gold price prediction.
    These aren't perfectly optimal but they're solid starting values.
    A proper grid search would improve them further — that's a v3 idea.
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
