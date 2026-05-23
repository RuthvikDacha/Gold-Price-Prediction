# 🥇 Gold Price Predictor

A machine learning web app that predicts gold prices using **Random Forest** and **XGBoost**, built with macroeconomic features, multi-step forecasting, SHAP explainability, PSI drift monitoring, and MLflow experiment tracking via DagsHub.

Live on **Streamlit Community Cloud** — free, public, always on.

---

## What It Does

I built this to cover the full ML lifecycle — not just training a model but monitoring it, tracking experiments, and explaining predictions. Gold price prediction made sense as a use case because it's driven by measurable macro factors I could pull for free.

### 📊 Market Dashboard
Live gold price chart (line or candlestick) with selectable moving averages and lookback periods up to 5 years. Five real-time macro indicator cards (USD Index, 10Y Treasury Yield, Oil, S&P 500, VIX) with daily change. News sentiment panel powered by VADER scoring recent gold headlines from Yahoo Finance — displayed as supplementary context only, not a model input.

### 🤖 Price Prediction
Predicts tomorrow's gold closing price with a 95% confidence interval and a gauge chart. Below that, a multi-step recursive forecast (3, 5, or 7 trading days ahead) with a widening confidence band chart and a full forecast table showing predicted price, confidence range, and percentage change per day. Two SHAP explainability charts show exactly why the model made each prediction — which features pushed the price up and which pushed it down, in dollar terms.

### 📈 Model Performance
RMSE, MAE, R², and MAPE metric cards. Actual vs predicted price chart with error panel below. Error distribution histogram and actual vs predicted scatter plot with a perfect-fit reference line.

### 🔍 Monitoring
PSI (Population Stability Index) and KS test drift detection across all features, a prediction distribution PSI check, a residual PSI check, and a colour-coded per-feature PSI bar chart. MLflow run history table with RMSE trend chart showing all past training runs.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Gold price data | `yfinance` — GC=F (gold futures) |
| Macro indicators | `yfinance` — DXY, TNX, CL=F, GSPC, VIX |
| ML models | `scikit-learn` (Random Forest) + `XGBoost` |
| Hyperparameter tuning | `optuna` — TPE search, 10–50 trials |
| Explainability | `shap` — waterfall + summary charts |
| Drift monitoring | `scipy.stats` — custom PSI + KS implementation |
| Experiment tracking | `MLflow` — local or DagsHub remote |
| News sentiment | `vaderSentiment` + `yfinance` headlines |
| Visualisation | `Plotly` |
| Frontend | `Streamlit` |
| Hosting | Streamlit Community Cloud (free) |

I replaced Evidently AI with a custom PSI implementation using scipy — PSI was originally developed for financial model monitoring so it's a better fit than a generic observability tool, and it removes a dependency that caused consistent installation issues.

---

## Project Structure

```
gold-price-predictor/
│
├── app.py              ← Main Streamlit app — all 4 tabs and UI logic
├── data.py             ← Gold + macro data fetching and feature engineering
├── model.py            ← Model training, evaluation, multi-step forecast, prediction
├── monitoring.py       ← Custom PSI + KS drift monitoring
├── sentiment.py        ← VADER news sentiment via yfinance headlines
├── shap_utils.py       ← SHAP waterfall and summary chart data
├── mlflow_utils.py     ← MLflow experiment logging (local or DagsHub)
├── tuning.py           ← Optuna hyperparameter search for RF and XGBoost
├── train.py            ← Standalone training script (optional CLI use)
│
├── .github/
│   └── workflows/
│       └── retrain.yml ← GitHub Actions workflow (manual trigger only)
│
├── .streamlit/
│   └── secrets.toml    ← Credentials template (NOT committed to GitHub)
│
├── runtime.txt         ← Pins Python 3.11 for Streamlit Cloud
├── requirements.txt    ← All Python dependencies
├── .gitignore          ← Excludes mlruns/, secrets.toml, __pycache__, etc.
└── README.md           ← This file
```

---

## Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/gold-price-predictor.git
cd gold-price-predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 5. View MLflow runs locally

In a second terminal with the venv active:

```bash
mlflow ui
```

Open `http://localhost:5000` to browse all training runs.

---

## Deploying to Streamlit Community Cloud

1. Push code to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **"Create app"** and fill in:
   - Repository: `your-username/gold-price-predictor`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **"Deploy"** — Streamlit Cloud installs everything from `requirements.txt` automatically

---

## Optional: DagsHub Remote MLflow

By default MLflow logs to a local `mlruns/` folder which resets when Streamlit Cloud restarts. DagsHub gives you persistent tracking for free.

1. Create a free account at [dagshub.com](https://dagshub.com)
2. Connect your GitHub repo
3. Go to your DagsHub repo → **Remote → Experiments** and copy the tracking URI
4. Get an access token from `dagshub.com/user/settings/tokens`
5. Go to Streamlit Cloud → **App Settings → Secrets** and add:

```toml
[mlflow]
dagshub_username = "your-dagshub-username"
dagshub_repo     = "gold-price-predictor"
dagshub_token    = "your-token-here"
```

---

## How to Identify Your MLflow Run

The sidebar has a **MLflow Run Label** field before training. Whatever you type becomes part of the run name:

```
TestRun-1__XGBoost__20260523_143022
```

That exact name appears in DagsHub so you can find your run instantly even if multiple people are using the app at the same time. If you leave the field blank it defaults to `TestRun__ModelType__Timestamp`.

---

## How the ML Model Works

```
Raw OHLCV data (GC=F gold futures, up to 5 years)
    + Macro data (DXY, TNX, CL=F, GSPC, VIX)
            │
            ▼
    Feature Engineering (19 base + 7 macro features)
    ├── Lag prices:      lag_1, lag_2, lag_3, lag_5, lag_10
    ├── Rolling stats:   7/20/50-day MA + std dev
    ├── Momentum:        5-day change, 1-day and 5-day % returns
    ├── Intraday:        High-Low range, Open-Close gap
    ├── Calendar:        Day of week, month, year
    └── Macro:           DXY, 10Y yield, oil, S&P 500 return, VIX
            │
            ▼
    Chronological Train/Test Split (never random for time series)
            │
            ▼
    Optional Optuna Tuning (10–50 trials, TPE sampler)
            │
            ▼
    Model Training — Random Forest or XGBoost
            │
            ▼
    Evaluation → RMSE, MAE, R², MAPE
            │
            ├── MLflow logs params, metrics, model artifact
            ├── SHAP explains each prediction
            └── PSI + KS monitors for data drift
```

---

## Understanding the Metrics

| Metric | What it means | Good value for gold |
|---|---|---|
| **RMSE** | Average error in USD per oz | Lower is better |
| **MAE** | Similar to RMSE, less skewed by outliers | Lower is better |
| **R²** | How much price variance the model explains | Close to 1.0 |
| **MAPE** | Error as a percentage of actual price | Below 1% is excellent |

---

## Understanding SHAP

Feature importance shows which inputs matter globally across all predictions. SHAP goes further — for each specific prediction it shows exactly how much every feature pushed the price up or down in dollar terms.

```
Today's prediction: $2,847
  Yesterday's price   → +$38  (pushed UP)
  VIX fear index      → +$22  (pushed UP)
  USD index           → -$15  (pushed DOWN)
  10Y yield           → -$9   (pushed DOWN)
  Baseline (avg pred) →  $2,811
                     ─────────
  Total predicted     →  $2,847
```

---

## Understanding the Monitoring

### PSI — Population Stability Index

Compares the distribution of features the model was trained on against what it's seeing in the test period. Originally developed for financial model monitoring in banking.

| PSI | Status | Action |
|---|---|---|
| < 0.10 | 🟢 Stable | No action needed |
| 0.10 – 0.25 | 🟡 Monitor | Keep watching |
| ≥ 0.25 | 🔴 Retrain | Retrain the model |

### KS Test

Statistical test checking whether two distributions are significantly different. A p-value below 0.05 means the difference is unlikely to be random noise.

---

## Multi-Step Forecasting

The app uses recursive forecasting — the model predicts Day 1, feeds that prediction back as input, predicts Day 2, and so on up to 7 trading days. The confidence band widens each day using a √step multiplier, which is the standard statistical approach for expressing growing uncertainty over a forecast horizon. By Day 7 the band is intentionally wider than Day 1 — that's correct behaviour, not a bug.

---

## A Note on News Sentiment

The sentiment panel on the Dashboard tab uses VADER to score recent gold headlines from Yahoo Finance. This is a display-only feature — it does not feed into the model. Yahoo Finance only provides the last few days of headlines, which isn't enough historical data to train on. Think of it as background context when interpreting a prediction.

---

## Ideas for Future Versions

- Hyperparameter comparison dashboard — visualise Optuna trial history
- Multi-output forecasting — train separate models per horizon instead of recursive
- Additional macro features — CPI release dates, Fed meeting dates as binary flags
- Scheduled retraining — re-enable the GitHub Actions cron trigger

---

## License

MIT — free to use, modify, and build on.
