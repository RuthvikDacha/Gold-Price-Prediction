# shap_utils.py
# SHAP (SHapley Additive exPlanations) for per-prediction explainability.
#
# Feature importance tells you what matters globally across all predictions.
# SHAP tells you why the model made one specific prediction — which features
# pushed the price up, which pushed it down, and by exactly how much.
#
# I use TreeExplainer here because both Random Forest and XGBoost are
# tree-based models. It's the fastest and most accurate SHAP explainer
# for this model type — no approximations needed.

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")


def get_explainer(model, X_train: pd.DataFrame, model_type: str):
    """
    Builds a SHAP TreeExplainer from the trained model.
    I pass X_train as background data so SHAP has a reference
    distribution to compute expected values against.

    Returns (explainer, expected_value) or (None, None) if SHAP isn't available.
    """
    if not SHAP_AVAILABLE:
        return None, None

    try:
        explainer      = shap.TreeExplainer(model, X_train)
        expected_value = float(explainer.expected_value)
        return explainer, expected_value
    except Exception as e:
        print(f"SHAP explainer error: {e}")
        return None, None


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Computes SHAP values for a given set of rows.
    Each value represents how much that feature contributed to pushing
    the prediction above or below the model's baseline (expected value).

    Positive SHAP value = feature pushed prediction UP
    Negative SHAP value = feature pushed prediction DOWN
    """
    if explainer is None:
        return None
    try:
        values = explainer.shap_values(X)
        # Random Forest returns a list for regression — take the first element
        if isinstance(values, list):
            values = values[0]
        return np.array(values)
    except Exception as e:
        print(f"SHAP values error: {e}")
        return None


def get_waterfall_data(
    shap_values: np.ndarray,
    feature_names: list,
    feature_values: pd.Series,
    expected_value: float,
    label_map: dict,
    top_n: int = 12,
) -> pd.DataFrame:
    """
    Builds a DataFrame for the waterfall chart showing how each feature
    contributed to today's specific prediction.

    A waterfall chart starts at the baseline (expected value = average prediction
    across all training data) and shows each feature adding or subtracting from
    that baseline until we reach the final prediction.

    Only shows the top_n features by absolute SHAP value to keep the chart readable.
    """
    if shap_values is None or len(shap_values) == 0:
        return pd.DataFrame()

    # Take the last row (most recent prediction)
    row_shap = shap_values[-1] if shap_values.ndim > 1 else shap_values

    df = pd.DataFrame({
        "feature":       feature_names,
        "shap_value":    row_shap,
        "feature_value": feature_values.values,
        "label":         [label_map.get(f, f) for f in feature_names],
    })

    # Sort by absolute contribution — biggest movers first
    df["abs_shap"] = df["shap_value"].abs()
    df = df.nlargest(top_n, "abs_shap").reset_index(drop=True)
    df = df.sort_values("shap_value", ascending=True).reset_index(drop=True)

    return df


def get_summary_data(
    shap_values: np.ndarray,
    feature_names: list,
    label_map: dict,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Builds a summary DataFrame showing average SHAP impact across the
    entire test set — how much each feature typically matters and in
    which direction it tends to push predictions.

    This is different from feature importance:
      Feature importance = how often a feature is used in splits
      SHAP summary       = how much each feature actually moves the output
    """
    if shap_values is None:
        return pd.DataFrame()

    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)

    df = pd.DataFrame({
        "feature":    feature_names,
        "mean_abs":   mean_abs,
        "mean_shap":  mean_shap,
        "label":      [label_map.get(f, f) for f in feature_names],
    }).nlargest(top_n, "mean_abs").reset_index(drop=True)

    df = df.sort_values("mean_abs", ascending=True).reset_index(drop=True)
    return df


# Readable labels and category colours reused from model.py
LABEL_MAP = {
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
