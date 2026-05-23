# data.py
# Handles all data fetching and feature engineering.
# I use GC=F (gold futures) instead of GLD (the ETF) because futures prices
# better represent actual physical gold market conditions.
# Macro features were added in v2 — they genuinely improve prediction quality
# because gold doesn't move in isolation, it reacts to USD, rates, and risk sentiment.

import yfinance as yf
import pandas as pd
import numpy as np

# Base price features I settled on after a lot of experimentation.
# Lags, rolling stats, and momentum together capture most of the
# information the model needs from raw price history.
BASE_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "rolling_mean_7", "rolling_mean_20", "rolling_mean_50",
    "rolling_std_7",  "rolling_std_20",  "rolling_std_50",
    "momentum_5",     "pct_change_1",    "pct_change_5",
    "high_low_spread", "open_close_diff",
    "day_of_week",    "month",           "year",
]

# Macro features added in v2.
# Gold has well-known relationships with each of these —
# inverse to USD, sensitive to rate expectations, correlated with oil/inflation,
# and a safe haven when the VIX spikes.
MACRO_FEATURES = [
    "dxy_close",     "dxy_change_1",    # US Dollar Index
    "tnx_close",                         # 10-Year Treasury Yield
    "oil_close",     "oil_change_1",    # Crude Oil Futures
    "sp500_change_1",                    # S&P 500 daily return (risk-on/off signal)
    "vix_close",                         # CBOE VIX — fear index
]

# Yahoo Finance tickers for each macro indicator
MACRO_TICKERS = {
    "dxy":   "DX-Y.NYB",
    "tnx":   "^TNX",
    "oil":   "CL=F",
    "sp500": "^GSPC",
    "vix":   "^VIX",
}


def fetch_gold_data(period: str = "max") -> pd.DataFrame:
    """
    Fetches historical gold futures data from Yahoo Finance.
    Using 'max' goes back to ~1999, giving roughly 25 years of data —
    enough to cover multiple economic cycles, crises, and bull markets.
    """
    print(f"Fetching gold data (period={period})...")
    df = yf.Ticker("GC=F").history(period=period)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    print(f"Got {len(df)} trading days.")
    return df


def fetch_macro_data(period: str = "max") -> pd.DataFrame:
    """
    Fetches all macro indicators and combines them into a single DataFrame.

    Different markets trade on slightly different calendars (e.g. bond markets
    close on some holidays that equities don't), so I forward-fill up to 5 days
    to handle those gaps without introducing look-ahead bias.
    """
    print("Fetching macro indicators...")
    frames = {}
    for name, sym in MACRO_TICKERS.items():
        try:
            raw = yf.Ticker(sym).history(period=period)[["Close"]].copy()
            raw.index = raw.index.tz_localize(None)
            raw.rename(columns={"Close": f"{name}_close"}, inplace=True)
            frames[name] = raw
            print(f"  {name} ({sym}) — OK")
        except Exception as e:
            print(f"  {name} ({sym}) — skipped: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=1)
    combined.ffill(limit=5, inplace=True)
    return combined


def merge_with_macro(gold_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins macro data onto the gold DataFrame so I only keep
    dates where gold data exists. Any leftover NaNs get forward-filled.
    """
    if macro_df.empty:
        return gold_df
    merged = gold_df.join(macro_df, how="left")
    merged.ffill(limit=5, inplace=True)
    merged.dropna(subset=["Close"], inplace=True)
    return merged


def engineer_features(df: pd.DataFrame, include_macro: bool = True) -> pd.DataFrame:
    """
    Builds every ML feature from the raw OHLCV + macro data.

    The target is always the NEXT day's closing price — shifted back by 1 row
    so that each training example contains features from day T and a label
    from day T+1.
    """
    df = df.copy()

    # Price lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Rolling trend and volatility
    for w in [7, 20, 50]:
        df[f"rolling_mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["Close"].rolling(w).std()

    # Momentum and return features
    df["momentum_5"]   = df["Close"] - df["Close"].shift(5)
    df["pct_change_1"] = df["Close"].pct_change(1)
    df["pct_change_5"] = df["Close"].pct_change(5)

    # Intraday range
    df["high_low_spread"] = df["High"]  - df["Low"]
    df["open_close_diff"] = df["Close"] - df["Open"]

    # Calendar effects (some months are historically stronger for gold)
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["year"]        = df.index.year

    # Macro-derived change features (only when macro data is present)
    if include_macro and "dxy_close" in df.columns:
        df["dxy_change_1"]   = df["dxy_close"].pct_change(1)
    if include_macro and "oil_close" in df.columns:
        df["oil_change_1"]   = df["oil_close"].pct_change(1)
    if include_macro and "sp500_close" in df.columns:
        df["sp500_change_1"] = df["sp500_close"].pct_change(1)

    # Target: next trading day's closing price
    df["target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df


def get_feature_columns(include_macro: bool = True, available_cols: list = None) -> list:
    """
    Returns the feature list for training. When macro is enabled I filter
    to only features that actually exist in the DataFrame — handles cases
    where one macro ticker might have failed to download.
    """
    cols = BASE_FEATURES + MACRO_FEATURES if include_macro else BASE_FEATURES
    if available_cols:
        cols = [c for c in cols if c in available_cols]
    return cols


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, include_macro: bool = True):
    """
    Chronological train/test split — never random for time series.

    Random splitting would let future data bleed into training,
    making metrics look much better than they actually are in production.
    """
    features  = get_feature_columns(include_macro, available_cols=list(df.columns))
    X         = df[features]
    y         = df["target"]
    split_idx = int(len(df) * (1 - test_size))

    return (
        X.iloc[:split_idx],   # X_train
        X.iloc[split_idx:],   # X_test
        y.iloc[:split_idx],   # y_train
        y.iloc[split_idx:],   # y_test
        df.iloc[split_idx:],  # test_df — kept for monitoring and charts
        features,             # exact feature list used (changes with macro toggle)
    )
