# monitoring.py
# Data drift and model monitoring using scipy and PSI.
# I replaced Evidently AI with this custom implementation because:
#   1. No external dependency — scipy is already installed with scikit-learn
#   2. PSI (Population Stability Index) was developed specifically for financial
#      model monitoring so it's a better fit than a generic ML observability tool
#   3. KS test gives a statistically rigorous drift signal with a proper p-value
#
# PSI thresholds (standard in finance / credit risk):
#   PSI < 0.10  → stable, no action needed
#   PSI < 0.25  → minor shift, worth monitoring
#   PSI >= 0.25 → significant shift, model should be retrained

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# PSI threshold labels I use throughout the UI
PSI_THRESHOLDS = {
    "stable":  0.10,
    "warning": 0.25,
}


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Population Stability Index between a reference distribution
    (training data) and a current distribution (recent/test data).

    The formula is: PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    I create bins based on the expected distribution's percentiles so that
    each bin contains roughly equal amounts of reference data. Then I check
    how much of the actual data falls into each bin.
    """
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual,   dtype=float)

    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Build bins from the expected (reference) distribution
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Convert to proportions and avoid division by zero
    e_pct = expected_counts / len(expected)
    a_pct = actual_counts   / len(actual)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return round(psi, 5)


def psi_status(psi: float) -> tuple:
    """Returns (status_label, color_class, emoji) for a given PSI value."""
    if psi < PSI_THRESHOLDS["stable"]:
        return "Stable",    "green",  "🟢"
    elif psi < PSI_THRESHOLDS["warning"]:
        return "Monitor",   "yellow", "🟡"
    else:
        return "Retrain",   "red",    "🔴"


def run_ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov two-sample test. Checks if two distributions are
    significantly different at the 5% significance level.

    Returns:
        statistic — KS test statistic (0 to 1, higher = more different)
        p_value   — probability the two distributions are the same
        drifted   — True if p-value < 0.05 (statistically significant drift)
    """
    reference = np.array(reference, dtype=float)
    current   = np.array(current,   dtype=float)
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]

    if len(reference) < 5 or len(current) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "drifted": False}

    stat, p_val = ks_2samp(reference, current)
    return {
        "statistic": round(float(stat),   4),
        "p_value":   round(float(p_val),  4),
        "drifted":   bool(p_val < 0.05),
    }


def run_full_monitoring(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        train_preds: np.ndarray, test_preds: np.ndarray) -> dict:
    """
    Runs the complete monitoring suite across all features and returns
    a structured results dict ready for the UI to consume.

    Returns:
        feature_results — per-feature PSI and KS test results
        prediction_psi  — PSI on the model's prediction distribution
        error_psi       — PSI on the residuals (are errors changing shape?)
        overall_status  — highest severity status across all features
        n_drifted       — number of features with PSI >= 0.10
        summary_df      — DataFrame suitable for displaying in st.dataframe
    """
    feature_results = {}

    for col in X_train.columns:
        psi = calculate_psi(X_train[col].values, X_test[col].values)
        ks  = run_ks_test(X_train[col].values, X_test[col].values)
        status, color, emoji = psi_status(psi)
        feature_results[col] = {
            "psi":       psi,
            "status":    status,
            "color":     color,
            "emoji":     emoji,
            "ks_stat":   ks["statistic"],
            "p_value":   ks["p_value"],
            "drifted":   ks["drifted"],
        }

    # Check the prediction and error distributions too
    train_errors = np.array(y_train) - train_preds
    test_errors  = np.array(y_test)  - test_preds

    prediction_psi = calculate_psi(train_preds, test_preds)
    error_psi      = calculate_psi(train_errors, test_errors)

    # Overall status — worst case across all features
    all_psi    = [v["psi"] for v in feature_results.values()]
    max_psi    = max(all_psi) if all_psi else 0.0
    n_drifted  = sum(1 for p in all_psi if p >= PSI_THRESHOLDS["stable"])
    overall_status, overall_color, overall_emoji = psi_status(max_psi)

    # Build a clean summary DataFrame for the UI
    from data import BASE_FEATURES, MACRO_FEATURES
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
        "dxy_close": "USD Index (DXY)",   "dxy_change_1": "DXY Change",
        "tnx_close": "10Y Yield",         "oil_close": "Oil Price",
        "oil_change_1": "Oil Change",     "sp500_change_1": "S&P 500 Return",
        "vix_close": "VIX",
    }
    rows = []
    for feat, res in feature_results.items():
        rows.append({
            "Feature":    label_map.get(feat, feat),
            "PSI":        res["psi"],
            "Status":     f"{res['emoji']} {res['status']}",
            "KS Stat":    res["ks_stat"],
            "p-value":    res["p_value"],
            "KS Drifted": "Yes" if res["drifted"] else "No",
        })
    summary_df = pd.DataFrame(rows)

    return {
        "feature_results":  feature_results,
        "prediction_psi":   prediction_psi,
        "error_psi":        error_psi,
        "overall_status":   overall_status,
        "overall_color":    overall_color,
        "overall_emoji":    overall_emoji,
        "max_psi":          max_psi,
        "n_drifted":        n_drifted,
        "n_total":          len(feature_results),
        "summary_df":       summary_df,
    }
