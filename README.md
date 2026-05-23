# 🥇 Gold Price Predictor

![Gold Price Predictor](image.png)

**[Live Demo](https://gold-price-prediction-rd.streamlit.app/)**


A machine learning web app that predicts the next day's gold price using
**Random Forest** and **XGBoost**, enriched with macroeconomic features,
news sentiment analysis, PSI drift monitoring, and MLflow experiment tracking.

Built as a portfolio project and deployed for free on **Streamlit Community Cloud**.

---

## What It Does

I built this because I wanted a project that covers the full ML lifecycle —
not just training a model, but also monitoring it in production, tracking experiments
over time, and getting notified when something goes wrong. Gold price prediction
made sense as a use case because it's influenced by a lot of measurable macro
factors that I could pull for free.

The app has four tabs:

**📊 Market Dashboard** — Live gold price chart (line or candlestick) with selectable
moving averages and lookback periods up to the full 25-year history. Also shows
five real-time macro indicators and a news sentiment panel powered by VADER.

**🤖 Price Prediction** — Predicts tomorrow's gold closing price with a 95% confidence
interval. The user picks Random Forest or XGBoost in the sidebar. A gauge chart
shows where the prediction sits relative to the current price and the confidence bounds.
Below the prediction, two SHAP charts explain exactly why the model made that call —
which features pushed the price up and which pushed it down, in dollar terms.

**📈 Model Performance** — RMSE, MAE, R², and MAPE metrics. Actual vs predicted price
chart, error over time, error distribution histogram, and an actual vs predicted scatter
plot with a perfect-fit reference line.

**🔍 Monitoring** — PSI (Population Stability Index) and KS test results across every
feature, a prediction distribution PSI check, a residual PSI check, and an MLflow run
history table with an RMSE trend chart to track model quality over time.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Gold price data | `yfinance` — GC=F (gold futures) |
| Macro indicators | `yfinance` — DXY, TNX, CL=F, GSPC, VIX |
| ML models | `scikit-learn` (Random Forest) + `XGBoost` |
| Drift monitoring | `scipy.stats` — PSI + KS test (custom implementation) |
| Explainability | `shap` — SHAP waterfall + summary charts |
| Experiment tracking | `MLflow` — local or DagsHub remote |
| News sentiment | `vaderSentiment` + `yfinance` headlines |
| Visualisation | `Plotly` |
| Frontend | `Streamlit` |
| Hosting | Streamlit Community Cloud (free) |

I deliberately avoided Evidently AI for monitoring because it caused dependency
issues. The custom PSI implementation I wrote is actually a better fit for financial
data since PSI was originally developed for credit risk model monitoring.

---

## Project Structure

```
gold-price-predictor/
│
├── app.py              ← Main Streamlit app (all 4 tabs and UI logic)
├── data.py             ← Gold + macro data fetching and feature engineering
├── model.py            ← Model training, evaluation, and prediction
├── monitoring.py       ← Custom PSI + KS drift monitoring (no Evidently)
├── sentiment.py        ← VADER news sentiment analysis via yfinance headlines
├── mlflow_utils.py     ← MLflow experiment logging (local or DagsHub)
│
├── .streamlit/
│   └── secrets.toml    ← Credentials template (NOT committed to GitHub)
│
├── requirements.txt    ← Python dependencies (read by Streamlit Cloud)
├── .gitignore          ← Excludes mlruns/, secrets.toml, __pycache__, etc.
└── README.md           ← This file
```

---

## Running Locally

### 1. Clone the repository

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

The app opens automatically at `http://localhost:8501`.

### 5. (Optional) Open the MLflow UI

In a separate terminal with the virtual environment active:

```bash
mlflow ui
```

Open `http://localhost:5000` to browse all training runs, compare metrics,
and download model artifacts.

---

## Deploying to Streamlit Community Cloud

1. Push your code to a **public GitHub repository**

   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

   > Make sure `.streamlit/secrets.toml` is in your `.gitignore` and is NOT pushed.

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **"Create app"** and fill in:
   - Repository: `your-username/gold-price-predictor`
   - Branch: `main`
   - Main file path: `app.py`

4. Click **"Deploy"** — Streamlit Cloud installs everything from `requirements.txt`
   automatically and gives you a live public URL.

5. Add your secrets on Streamlit Cloud:
   - Go to your app → **⋮ menu → Settings → Secrets**
   - Paste the contents of your `.streamlit/secrets.toml` (with real values filled in)

---

## Optional Setup: DagsHub Remote MLflow

By default MLflow logs to a local `mlruns/` folder. This works great locally but
resets every time Streamlit Cloud restarts the app. To get persistent tracking:

1. Create a free account at [dagshub.com](https://dagshub.com)
2. Connect your GitHub repository
3. Go to your DagsHub repo → **Remote → Experiments** and copy your tracking URI
4. Get an access token from `dagshub.com/user/settings/tokens`
5. Fill in the `[mlflow]` section of `.streamlit/secrets.toml`
6. Add those same secrets to Streamlit Cloud under App Settings → Secrets

Once connected, every training run in the deployed app gets logged permanently
to your DagsHub MLflow dashboard — you can browse all runs from any browser.

---

---

## How the ML Model Works

```
Raw OHLCV data (GC=F gold futures)
    + Macro data (DXY, TNX, CL=F, GSPC, VIX)
            │
            ▼
    Feature Engineering
    ├── Lag prices:         lag_1, lag_2, lag_3, lag_5, lag_10
    ├── Rolling stats:      7/20/50-day moving average + std dev
    ├── Momentum:           5-day change, 1-day and 5-day % returns
    ├── Intraday:           High-Low range, Open-Close gap
    ├── Calendar:           Day of week, month, year
    └── Macro:              DXY, 10Y yield, oil, S&P 500 return, VIX
            │
            ▼
    Chronological 80/20 Train/Test Split
    (random splits would leak future data into training)
            │
            ▼
    Model Training
    ├── Random Forest:  200 trees, max depth 10
    └── XGBoost:        200 rounds, learning rate 0.05
            │
            ▼
    Evaluation → RMSE, MAE, R², MAPE
            │
            ▼
    MLflow logs everything (params, metrics, model artifact)
            │
            ▼
    PSI + KS monitoring (train vs test distribution comparison)
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

## Understanding the Monitoring

### PSI — Population Stability Index

I chose PSI over generic drift libraries because it was specifically designed
for financial model monitoring (originally used in credit risk / banking).

| PSI Value | Status | Action |
|---|---|---|
| < 0.10 | 🟢 Stable | No action needed |
| 0.10 – 0.25 | 🟡 Monitor | Keep an eye on it |
| ≥ 0.25 | 🔴 Retrain | Retrain the model |

### KS Test — Kolmogorov-Smirnov

A statistical test that checks whether two distributions are significantly
different. A p-value below 0.05 means the difference is unlikely to be random.
I run this alongside PSI because they catch different types of drift.

---

## A Note on News Sentiment

The sentiment panel on the Dashboard tab uses VADER to score recent gold headlines
fetched via `yfinance`. This is a supplementary display feature, not a model input.

The reason it doesn't feed into the model is that `yfinance` only provides the last
few days of headlines — not enough historical data to build a training set from.
It's useful as background context when interpreting a prediction but it doesn't
affect what the model outputs.

---

## Ideas for v3

- GitHub Actions scheduled workflow for automatic daily retraining
- Pre-trained model artifacts committed to the repo so the app loads instantly
- Hyperparameter tuning with Optuna or GridSearchCV
- Add more macro features (CPI releases, Fed meeting dates as binary flags)
- Multi-step forecasting (predict 5 or 7 days ahead instead of just 1)

---

## License

MIT — free to use, modify, and build on.
