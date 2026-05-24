# app.py
# Main Streamlit application for the Gold Price Predictor v2.
# Everything the user interacts with lives here — 4 tabs covering
# the market dashboard, predictions, model performance, and monitoring.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

from data        import fetch_gold_data, fetch_macro_data, merge_with_macro, \
                        engineer_features, prepare_data, get_feature_columns
from model       import train_model, evaluate_model, get_feature_importance, \
                        predict_next_day, predict_multi_step
from mlflow_utils import log_training_run, get_run_history, get_best_run, setup_mlflow
from tuning      import run_tuning, get_trial_history
from monitoring  import run_full_monitoring, psi_status
from sentiment   import get_gold_news_sentiment
from shap_utils  import (get_explainer, compute_shap_values, get_waterfall_data,
                          get_summary_data, LABEL_MAP, SHAP_AVAILABLE)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Gold Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — gold / dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
.g-card {
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(255,215,0,0.35);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    transition: border-color .2s, transform .15s;
    height:100%;
}
.g-card:hover { border-color:#FFD700; transform:translateY(-2px); }
.g-card .gc-label {
    color:#9ca3af; font-size:10px; text-transform:uppercase;
    letter-spacing:1.8px; margin-bottom:6px;
}
.g-card .gc-value { color:#FFD700; font-size:24px; font-weight:800; line-height:1.1; }
.g-card .gc-sub   { color:#64748b; font-size:12px; margin-top:5px; }

/* ── Macro mini-cards ── */
.macro-card {
    background:#111827;
    border:1px solid #1e293b;
    border-radius:10px;
    padding:12px 14px;
    text-align:center;
}
.macro-card .mc-name  { color:#6b7280; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
.macro-card .mc-val   { color:#e2e8f0; font-size:18px; font-weight:700; margin:4px 0 2px; }
.macro-card .mc-chg   { font-size:12px; font-weight:600; }
.up   { color:#4ade80; }
.down { color:#f87171; }
.neu  { color:#94a3b8; }

/* ── Prediction box ── */
.pred-box {
    background: linear-gradient(145deg,#1a1a2e,#0d2137);
    border: 2px solid #FFD700;
    border-radius: 18px;
    padding: 36px 24px 28px;
    text-align: center;
}
.pred-box .pb-eyebrow { color:#9ca3af; font-size:11px; letter-spacing:2px; text-transform:uppercase; }
.pred-box .pb-price   { color:#FFD700; font-size:56px; font-weight:800; line-height:1; margin:10px 0 6px; }
.pred-box .pb-change  { font-size:20px; font-weight:700; margin-bottom:10px; }
.pred-box .pb-range   { color:#94a3b8; font-size:13px; }
.pred-box .pb-meta    { color:#475569; font-size:11px; margin-top:10px; }

/* ── News cards ── */
.news-card {
    background:#111827;
    border:1px solid #1e293b;
    border-left: 3px solid var(--nc-accent,#FFD700);
    border-radius:0 10px 10px 0;
    padding:12px 14px;
    margin-bottom:8px;
}
.news-card .nc-title  { color:#e2e8f0; font-size:13px; font-weight:500; line-height:1.4; }
.news-card .nc-meta   { color:#64748b; font-size:11px; margin-top:5px; }
.news-card .nc-badge  {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:10px; font-weight:700; margin-right:6px;
}
.badge-pos { background:#052e16; color:#4ade80; border:1px solid #16a34a; }
.badge-neg { background:#200505; color:#f87171; border:1px solid #dc2626; }
.badge-neu { background:#1e293b; color:#94a3b8; border:1px solid #374151; }

/* ── Section header ── */
.s-hdr {
    border-left:4px solid #FFD700;
    padding-left:12px;
    margin:22px 0 12px;
    font-weight:700;
    font-size:15px;
    color:#e2e8f0;
}

/* ── Info callout ── */
.info-pill {
    background:#0f172a;
    border:1px solid #1e3a5f;
    border-radius:8px;
    padding:10px 14px;
    font-size:12px;
    color:#94a3b8;
    line-height:1.6;
    margin:8px 0 14px;
}
.info-pill strong { color:#60a5fa; }

/* ── Drift badge ── */
.drift-banner {
    border-radius:12px;
    padding:16px 20px;
    text-align:center;
    margin-bottom:16px;
}
.drift-green  { background:#052e16; border:1px solid #16a34a; }
.drift-yellow { background:#1c1100; border:1px solid #ca8a04; }
.drift-red    { background:#200505; border:1px solid #dc2626; }
.db-label     { color:#9ca3af; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; }
.db-status    { font-size:20px; font-weight:800; margin:4px 0 2px; }
.db-sub       { font-size:12px; color:#94a3b8; }
.txt-green    { color:#4ade80; }
.txt-yellow   { color:#fbbf24; }
.txt-red      { color:#f87171; }

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg,#92650a,#FFD700) !important;
    color:#000 !important; font-weight:700 !important;
    border:none !important; border-radius:10px !important;
    padding:10px 0 !important;
    transition: opacity .15s !important;
}
.stButton > button:hover { opacity:.88 !important; }

/* ── Tab active underline ── */
button[data-baseweb="tab"][aria-selected="true"] {
    color:#FFD700 !important;
    border-bottom-color:#FFD700 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
_defaults = dict(
    trained=False, model=None, model_type=None, params=None,
    metrics=None, df_raw=None, df_features=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    test_df=None, predictions=None, train_preds=None,
    run_id=None, feature_names=None,
    monitoring_results=None, last_trained=None,
    mlflow_backend="local",
    shap_explainer=None, shap_expected=None, shap_values=None,
    forecast_df=None,
    run_name=None, run_label="",
    optuna_study=None, optuna_model_type=None,
)
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def card(label, value, sub=""):
    s = f'<div class="gc-sub">{sub}</div>' if sub else ""
    return f"""<div class="g-card">
        <div class="gc-label">{label}</div>
        <div class="gc-value">{value}</div>{s}
    </div>"""

def macro_card(name, value, change=None, suffix=""):
    if change is not None:
        sign  = "+" if change >= 0 else ""
        cls   = "up" if change >= 0 else "down"
        chg_h = f'<div class="mc-chg {cls}">{sign}{change:.2f}{suffix}</div>'
    else:
        chg_h = ""
    return f"""<div class="macro-card">
        <div class="mc-name">{name}</div>
        <div class="mc-val">{value}</div>{chg_h}
    </div>"""

def gl(title="", height=420):
    """Reusable gold Plotly layout."""
    return dict(
        title=dict(text=title, font=dict(color="#FFD700", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,22,0.8)",
        font=dict(color="#94a3b8", size=12),
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        margin=dict(l=8, r=8, t=44, b=8),
    )

def _read_secret(section, key, default=""):
    try:
        return st.secrets[section][key]
    except:
        return default

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 6px;'>
      <span style='font-size:38px;'>🥇</span><br>
      <span style='color:#FFD700;font-size:17px;font-weight:800;letter-spacing:.5px;'>
        Gold Predictor
      </span><br>
      <span style='color:#475569;font-size:11px;'>v3.0 — ML + Monitoring</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### ⚙️ Model Settings")

    model_choice = st.selectbox(
        "ML Algorithm",
        ["Random Forest", "XGBoost"],
        help="Random Forest: stable and interpretable.\n"
             "XGBoost: usually more accurate.",
    )

    period_choice = st.selectbox(
        "Historical Data Range",
        ["1y", "2y", "3y", "4y", "5y"],
        index=1,
        help="How much historical gold price data to train on.",
    )

    test_size_choice = st.slider(
        "Test Set Size", 0.10, 0.30, 0.10, 0.05,
        help="Fraction of data reserved for evaluation only.",
    )

    st.markdown("#### 🔬 Feature & Tuning Options")

    include_macro = st.toggle(
        "Include Macro Features", value=True,
        help="Adds USD index, Treasury yields, oil, S&P 500, and VIX.",
    )

    use_tuning = st.toggle(
        "Optuna Hyperparameter Tuning", value=False,
        help="Searches for the best model parameters automatically. "
             "Adds 2–5 minutes but usually improves RMSE.",
    )

    if use_tuning:
        n_trials = st.slider("Optuna Trials", 10, 50, 10, 10,
                             help="More trials = better params but longer wait.")
    else:
        n_trials = 20

    st.markdown("#### 🏷️ MLflow Run Label")
    run_label = st.text_input(
        "MLflow Run Label  *(required)*",
        value="",
        placeholder="e.g. TestRun-1 or XGB-Macro-Test",
        help="Give this training run a label so you can find it in the MLflow / DagsHub UI. "
             "Your run will be saved as: YourLabel__ModelType__Timestamp",
    )
    st.caption("💡 This exact label will appear as the run name in DagsHub — "
               "useful when multiple people use the app.")

    if run_label.strip() == "":
        st.warning("Please enter a run label before training.", icon="⚠️")

    st.markdown("---")
    train_btn = st.button(
        "🚀  Train Model",
        use_container_width=True,
        disabled=(run_label.strip() == ""),
    )
    load_btn = False

    if st.session_state.trained:
        st.markdown("---")
        st.success("✅  Model ready")
        st.caption(f"Algorithm: **{st.session_state.model_type}**")
        if st.session_state.metrics:
            m = st.session_state.metrics
            st.caption(f"RMSE **${m['rmse']:.2f}** · R² **{m['r2']:.4f}**")
        if st.session_state.last_trained:
            st.caption(f"Trained: {st.session_state.last_trained}")
        if st.session_state.run_name:
            st.info(f"MLflow run name:\n`{st.session_state.run_name}`", icon="🏷️")
        if st.session_state.mlflow_backend == "dagshub":
            st.info("📡 Logging to DagsHub", icon="📡")
        else:
            st.caption("MLflow: local (mlruns/)")

    st.markdown("---")
    st.caption("Data: Yahoo Finance · GC=F")
    st.caption("Stack: scikit-learn · XGBoost\nMLflow · Optuna · scipy · VADER")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if train_btn:
    bar = st.progress(0, "Fetching gold price data…")
    df_raw = fetch_gold_data(period_choice)
    st.session_state.df_raw = df_raw
    bar.progress(15, "Fetching macro indicators…" if include_macro else "Skipping macro…")

    if include_macro:
        macro_df  = fetch_macro_data(period_choice)
        df_merged = merge_with_macro(df_raw, macro_df)
    else:
        df_merged = df_raw.copy()

    bar.progress(35, "Engineering features…")
    df_features = engineer_features(df_merged, include_macro=include_macro)
    st.session_state.df_features = df_features

    bar.progress(50, f"Running Optuna ({n_trials} trials)…" if use_tuning
                 else f"Training {model_choice}…")
    X_train, X_test, y_train, y_test, test_df, feature_names = \
        prepare_data(df_features, test_size_choice, include_macro)

    if use_tuning:
        tuned_params, best_val_rmse, optuna_study = run_tuning(
            model_choice, X_train, y_train, n_trials
        )
        tuned_params["random_state"] = 42
        if model_choice == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor as RFR
            tuned_params["n_jobs"] = -1
            model = RFR(**tuned_params)
        else:
            from xgboost import XGBRegressor as XGBR
            tuned_params["verbosity"] = 0
            model = XGBR(**tuned_params)
        model.fit(X_train, y_train)
        params = tuned_params
        st.session_state.optuna_study      = optuna_study
        st.session_state.optuna_model_type = model_choice
        st.toast(f"✅ Optuna best val RMSE: ${best_val_rmse:.2f}", icon="🎯")
    else:
        model, params = train_model(X_train, y_train, model_choice)
        st.session_state.optuna_study      = None
        st.session_state.optuna_model_type = None

    st.session_state.update(dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        test_df=test_df, model=model,
        params=params,   model_type=model_choice,
        feature_names=feature_names,
    ))

    bar.progress(72, "Evaluating model…")
    metrics      = evaluate_model(model, X_test, y_test)
    train_preds  = model.predict(X_train)
    st.session_state.metrics      = metrics
    st.session_state.predictions  = metrics["predictions"]
    st.session_state.train_preds  = train_preds

    bar.progress(88, "Logging to MLflow…")
    fi_df   = get_feature_importance(model, feature_names)
    backend = setup_mlflow()

    # Build a unique run name using the user's label so they can
    # find their run in DagsHub / MLflow UI immediately
    label      = run_label.strip() if run_label.strip() else "TestRun"
    named_run  = f"{label}__{model_choice.replace(' ', '_')}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_id = log_training_run(
        model, model_choice, params, metrics, fi_df,
        include_macro, period_choice,
        run_name=named_run,
    )
    st.session_state.run_id         = run_id
    st.session_state.run_name       = named_run
    st.session_state.mlflow_backend = backend

    bar.progress(94, "Computing SHAP values…")
    if SHAP_AVAILABLE:
        explainer, expected_val = get_explainer(model, X_train, model_choice)
        shap_vals = compute_shap_values(explainer, X_test)
        st.session_state.shap_explainer = explainer
        st.session_state.shap_expected  = expected_val
        st.session_state.shap_values    = shap_vals
    else:
        st.session_state.shap_explainer = None
        st.session_state.shap_expected  = None
        st.session_state.shap_values    = None

    st.session_state.monitoring_results = None
    st.session_state.forecast_df        = None
    st.session_state.last_trained       = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.trained            = True

    bar.progress(100, "Done!")
    st.toast(f"✅ Model trained! Your MLflow run name: {named_run}", icon="🥇")
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:10px 0 4px;'>
  <h1 style='color:#FFD700;font-size:2.4rem;font-weight:800;
             letter-spacing:-.5px;margin:0;line-height:1.1;'>
    🥇 Gold Price Predictor
  </h1>
  <p style='color:#64748b;font-size:13px;margin:6px 0 0;'>
    Machine Learning &nbsp;·&nbsp; Macro Features &nbsp;·&nbsp;
    MLflow Tracking &nbsp;·&nbsp; PSI Drift Monitoring
  </p>
</div>
<hr style='border-color:#1e293b;margin:14px 0 20px;'>
""", unsafe_allow_html=True)

# ── Prompt if untrained ────────────────────────────────────────────────────────
if not st.session_state.trained:
    st.info(
        "👈 **Choose your settings in the sidebar and click 'Train Model' to get started.**\n\n"
        "The model will fetch live gold price data, train on it, and be ready to predict.",
        icon="ℹ️",
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, t, d in [
        (c1, "📊", "Market Dashboard",  "Gold price chart, macro indicators, and news sentiment."),
        (c2, "🤖", "Price Prediction",  "Tomorrow's predicted price with a confidence interval."),
        (c3, "📈", "Model Performance", "RMSE, R², actual vs predicted, and error breakdown."),
        (c4, "🔍", "Monitoring",        "PSI drift detection and MLflow experiment history."),
    ]:
        with col:
            st.markdown(f"**{icon} {t}**\n\n{d}")
    st.stop()

# ── Convenience aliases ────────────────────────────────────────────────────────
df_raw       = st.session_state.df_raw
df_features  = st.session_state.df_features
model        = st.session_state.model
model_type   = st.session_state.model_type
metrics      = st.session_state.metrics
predictions  = st.session_state.predictions
train_preds  = st.session_state.train_preds
y_test       = st.session_state.y_test
y_train      = st.session_state.y_train
test_df      = st.session_state.test_df
X_train      = st.session_state.X_train
X_test       = st.session_state.X_test
feature_names = st.session_state.feature_names
shap_explainer = st.session_state.shap_explainer
shap_expected  = st.session_state.shap_expected
shap_values    = st.session_state.shap_values
forecast_df    = st.session_state.forecast_df
optuna_study   = st.session_state.optuna_study
optuna_mtype   = st.session_state.optuna_model_type

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Dashboard",
    "🤖  Prediction",
    "📈  Performance",
    "🔍  Monitoring",
    "🎯  Optuna",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MARKET DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # Gold stats
    cur  = float(df_raw["Close"].iloc[-1])
    prev = float(df_raw["Close"].iloc[-2])
    chg  = cur - prev
    pct  = chg / prev * 100
    h52  = float(df_raw["Close"].tail(252).max())
    l52  = float(df_raw["Close"].tail(252).min())
    sign = "+" if chg >= 0 else ""
    cls  = "up" if chg >= 0 else "down"
    arr  = "▲" if chg >= 0 else "▼"

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card("Gold Price (USD/oz)", f"${cur:,.2f}",
                              f'<span class="{cls}">{arr} ${abs(chg):.2f} today</span>'),
                         unsafe_allow_html=True)
    with c2: st.markdown(card("Daily Change",
                              f'<span class="{cls}">{sign}{pct:.2f}%</span>',
                              f"Previous close ${prev:,.2f}"),
                         unsafe_allow_html=True)
    with c3: st.markdown(card("52-Week High", f"${h52:,.2f}"), unsafe_allow_html=True)
    with c4: st.markdown(card("52-Week Low",  f"${l52:,.2f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Macro indicators row ──────────────────────────────────────────────────
    st.markdown('<div class="s-hdr">Macro Indicators</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
      <strong>Why macro?</strong> Gold doesn't move in isolation. It has well-known
      inverse relationships with the USD and interest rates, and tends to rise during
      periods of market fear (high VIX). These indicators give the model — and you —
      valuable context around each price move.
    </div>""", unsafe_allow_html=True)

    macro_tickers = {
        "USD Index":  ("DX-Y.NYB", "",     "dxy"),
        "10Y Yield":  ("^TNX",     "%",    "tnx"),
        "Oil (WTI)":  ("CL=F",     "",     "oil"),
        "S&P 500":    ("^GSPC",    "",     "sp500"),
        "VIX":        ("^VIX",     "",     "vix"),
    }
    macro_cols = st.columns(5)
    for i, (display_name, (sym, sfx, key)) in enumerate(macro_tickers.items()):
        with macro_cols[i]:
            try:
                col_name = f"{key}_close"
                if col_name in df_features.columns:
                    val = float(df_features[col_name].iloc[-1])
                    prev_val = float(df_features[col_name].iloc[-2])
                    delta = val - prev_val
                    if sfx == "%":
                        st.markdown(macro_card(display_name, f"{val:.2f}%", delta, "%"),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(macro_card(display_name, f"{val:,.2f}", delta, ""),
                                    unsafe_allow_html=True)
                else:
                    st.markdown(macro_card(display_name, "—", None), unsafe_allow_html=True)
            except:
                st.markdown(macro_card(display_name, "—", None), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="s-hdr">Gold Price Chart</div>', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([2, 3, 3])
    with ctrl1:
        chart_type = st.radio("Style", ["Line", "Candlestick"], horizontal=True)
    with ctrl2:
        mas = st.multiselect("Moving Averages",
                             ["MA 20", "MA 50", "MA 200"], default=["MA 20", "MA 50"])
    with ctrl3:
        lb = st.select_slider("Period", ["3M", "6M", "1Y", "2Y", "5Y", "All"], value="2Y")

    lb_map = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "All": len(df_raw)}
    dp     = df_raw.tail(lb_map[lb])

    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=dp.index, open=dp["Open"], high=dp["High"],
            low=dp["Low"], close=dp["Close"], name="OHLC",
            increasing_line_color="#4ade80", decreasing_line_color="#f87171",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=dp.index, y=dp["Close"], name="Gold",
            line=dict(color="#FFD700", width=2),
            fill="tozeroy", fillcolor="rgba(255,215,0,0.04)",
        ))

    MA_C = {"MA 20": "#60a5fa", "MA 50": "#f472b6", "MA 200": "#a78bfa"}
    MA_D = {"MA 20": 20,        "MA 50": 50,        "MA 200": 200}
    for ma in mas:
        vals = df_raw["Close"].rolling(MA_D[ma]).mean().reindex(dp.index)
        fig.add_trace(go.Scatter(x=dp.index, y=vals, name=ma,
                                 line=dict(color=MA_C[ma], width=1.5, dash="dot")))
    fig.update_layout(**gl("Gold Futures Price  (USD / troy oz)", 430))
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    vol_c = ["#4ade80" if dp["Close"].iloc[i] >= dp["Open"].iloc[i] else "#f87171"
             for i in range(len(dp))]
    fig_v = go.Figure(go.Bar(x=dp.index, y=dp["Volume"], marker_color=vol_c, opacity=.7))
    fig_v.update_layout(**gl("Volume", 160))
    st.plotly_chart(fig_v, use_container_width=True)

    # ── News sentiment ────────────────────────────────────────────────────────
    st.markdown('<div class="s-hdr">News Sentiment</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
      <strong>📌 What this is:</strong> Recent gold-related headlines scored using
      VADER sentiment analysis. This is a <strong>supplementary signal only</strong>
      — it is not part of the core model training because Yahoo Finance only provides
      the last few days of headlines, which isn't enough historical data to train on.
      Think of it as background context for today's prediction, not a model input.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Fetching latest gold news…"):
        sentiment_data = get_gold_news_sentiment(max_articles=8)

    if not sentiment_data["available"]:
        st.warning("Install `vaderSentiment` to enable news sentiment: `pip install vaderSentiment`")
    elif not sentiment_data["articles"]:
        st.info("No recent headlines found.")
    else:
        avg  = sentiment_data["avg_compound"]
        ovrl = sentiment_data["overall"]
        n    = sentiment_data["article_count"]

        score_col = "up" if avg > 0.05 else ("down" if avg < -0.05 else "neu")
        sign2     = "+" if avg >= 0 else ""

        sc1, sc2, sc3 = st.columns(3)
        with sc1: st.markdown(card("Overall Sentiment", ovrl,
                                   f"Across {n} recent articles"),
                               unsafe_allow_html=True)
        with sc2: st.markdown(card("Avg Compound Score",
                                   f'<span class="{score_col}">{sign2}{avg:.3f}</span>',
                                   "Range: −1.0 (most negative) to +1.0 (most positive)"),
                               unsafe_allow_html=True)
        with sc3:
            pos_c = sum(1 for a in sentiment_data["articles"] if a["sentiment"] == "Positive")
            neg_c = sum(1 for a in sentiment_data["articles"] if a["sentiment"] == "Negative")
            st.markdown(card("Headline Split",
                             f"🟢 {pos_c}  🔴 {neg_c}  ➡️ {n - pos_c - neg_c}",
                             "Positive · Negative · Neutral"),
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        badge_cls = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}
        accent    = {"Positive": "#16a34a",  "Negative": "#dc2626",   "Neutral": "#374151"}

        for a in sentiment_data["articles"]:
            bc  = badge_cls.get(a["sentiment"], "badge-neu")
            acc = accent.get(a["sentiment"], "#374151")
            link_html = (f'<a href="{a["link"]}" target="_blank" '
                         f'style="color:#60a5fa;font-size:11px;">Read more →</a>'
                         if a.get("link") else "")
            st.markdown(f"""
            <div class="news-card" style="--nc-accent:{acc};">
              <div class="nc-title">{a["title"]}</div>
              <div class="nc-meta">
                <span class="nc-badge {bc}">{a["emoji"]} {a["sentiment"]}</span>
                <span style="color:#64748b;">Score: {a["compound"]:+.3f}</span>
                &nbsp;·&nbsp;
                <span style="color:#64748b;">{a["source"]}</span>
                &nbsp;·&nbsp; {link_html}
              </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="s-hdr">Tomorrow\'s Price Prediction</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-pill">
          The model uses the most recent gold price data and macro conditions
          as inputs and estimates the next trading day's closing price.
          The confidence interval is derived from the variance across individual
          decision trees (Random Forest) or from the evaluation RMSE (XGBoost).
        </div>""", unsafe_allow_html=True)

        if st.button("🔮  Generate Prediction", use_container_width=True):
            with st.spinner("Running prediction…"):
                pred, lo, hi, std = predict_next_day(
                    model, df_features, feature_names, model_type, metrics["rmse"]
                )
                delta   = pred - cur
                delta_p = delta / cur * 100
                d_sign  = "+" if delta >= 0 else ""
                d_cls   = "up" if delta >= 0 else "down"
                d_icon  = "📈" if delta >= 0 else "📉"
                next_dt = (datetime.now() + timedelta(days=1)).strftime("%A, %B %d %Y")

                st.markdown(f"""
                <div class="pred-box">
                  <div class="pb-eyebrow">Predicted Closing Price</div>
                  <div class="pb-price">${pred:,.2f}</div>
                  <div class="pb-change {d_cls}">
                    {d_icon} {d_sign}${delta:.2f} &nbsp;({d_sign}{delta_p:.2f}%)
                  </div>
                  <div class="pb-range">
                    95% Confidence Interval: ${lo:,.2f} – ${hi:,.2f}
                  </div>
                  <div class="pb-meta">
                    {next_dt} &nbsp;·&nbsp; {model_type} &nbsp;·&nbsp; σ = ${std:.2f}
                  </div>
                </div>""", unsafe_allow_html=True)

                # Gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    delta={"reference": cur, "valueformat": ".2f", "prefix": "$"},
                    number={"prefix": "$", "valueformat": ",.2f",
                            "font": {"color": "#FFD700", "size": 28}},
                    gauge={
                        "axis": {"range": [lo * 0.999, hi * 1.001], "tickformat": ",.0f"},
                        "bar":  {"color": "#FFD700"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [lo * 0.999, lo], "color": "rgba(248,113,113,0.15)"},
                            {"range": [hi, hi * 1.001], "color": "rgba(74,222,128,0.15)"},
                        ],
                        "threshold": {
                            "line":      {"color": "#60a5fa", "width": 3},
                            "thickness": 0.85,
                            "value":     cur,
                        },
                    },
                ))
                fig_g.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    height=240,
                    margin=dict(l=20, r=20, t=24, b=8),
                )
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption(f"🔵 Blue marker = current price (${cur:,.2f})")
        else:
            st.info("Click 'Generate Prediction' to get tomorrow's forecast.", icon="💡")

    with right:
        st.markdown('<div class="s-hdr">Feature Importance</div>', unsafe_allow_html=True)
        st.caption("Which inputs have the most influence on the model's predictions?")

        fi_df = get_feature_importance(model, feature_names)
        top   = fi_df.head(15)

        CAT_COLORS = {
            "Price History":      "#FFD700",
            "Trend & Volatility": "#60a5fa",
            "Momentum":           "#f472b6",
            "Intraday":           "#34d399",
            "Calendar":           "#94a3b8",
            "Macro":              "#a78bfa",
            "Other":              "#64748b",
        }
        bar_colors = [CAT_COLORS.get(c, "#64748b") for c in top["Category"]]

        fig_fi = go.Figure(go.Bar(
            x=top["Importance"], y=top["Label"],
            orientation="h", marker_color=bar_colors,
        ))
        fig_fi.update_layout(**gl("Top 15 Feature Importances", 500))
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

        # Category colour legend rendered as a clean HTML caption
        # below the chart — avoids the overlap issue with in-chart annotations
        legend_html = " &nbsp;·&nbsp; ".join(
            f'<span style="color:{col};">■ {cat}</span>'
            for cat, col in CAT_COLORS.items()
            if cat in top["Category"].values
        )
        st.markdown(
            f'<div style="text-align:center;font-size:11px;'
            f'color:#64748b;margin-top:-10px;">{legend_html}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-STEP FORECAST — full width below the two columns
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("---")
    st.markdown('<div class="s-hdr">📅 Multi-Step Forecast</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
      <strong>How this works:</strong> I use recursive forecasting — the model predicts
      Day 1, then that prediction gets fed back as input to predict Day 2, and so on.
      This is honest about how uncertainty grows: the confidence band deliberately widens
      each day using a √step multiplier, which is the standard statistical approach for
      sequential forecast uncertainty. By Day 7 the band is wider than Day 1 — that's
      intentional and correct, not a bug.
    </div>""", unsafe_allow_html=True)

    fc1, fc2 = st.columns([3, 1], gap="large")
    with fc2:
        horizon = st.select_slider(
            "Forecast Horizon",
            options=[3, 5, 7],
            value=7,
            help="How many trading days ahead to forecast.",
        )
        run_forecast = st.button("📅  Run Forecast", use_container_width=True)

    if run_forecast:
        with st.spinner(f"Running {horizon}-day recursive forecast…"):
            fdf = predict_multi_step(
                model, df_features, feature_names,
                model_type, metrics["rmse"], horizon,
            )
            st.session_state.forecast_df = fdf
            forecast_df = fdf
        st.toast(f"✅ {horizon}-day forecast complete!", icon="📅")

    if forecast_df is not None and not forecast_df.empty:
        # ── Forecast chart ────────────────────────────────────────────────────
        last_n   = 30
        hist_df  = df_raw.tail(last_n)

        fig_fc = go.Figure()

        # Historical price line
        fig_fc.add_trace(go.Scatter(
            x=list(hist_df.index),
            y=hist_df["Close"].tolist(),
            name="Historical Price",
            line=dict(color="#FFD700", width=2),
        ))

        # Bridge point — connects history to forecast cleanly
        bridge_x = [hist_df.index[-1], hist_df.index[-1]]
        bridge_y = [float(hist_df["Close"].iloc[-1]),
                    float(forecast_df["predicted"].iloc[0])]

        # Forecast dates (use strings since they're formatted)
        fc_dates = [f"Day +{r['day']} ({r['date']})" for _, r in forecast_df.iterrows()]

        # Confidence band
        fig_fc.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=forecast_df["upper"].tolist() + forecast_df["lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(255,215,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Band",
            hoverinfo="skip",
        ))

        # Upper bound line
        fig_fc.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["upper"].tolist(),
            line=dict(color="rgba(255,215,0,0.25)", width=1, dash="dot"),
            name="Upper Bound",
            showlegend=False,
        ))

        # Lower bound line
        fig_fc.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["lower"].tolist(),
            line=dict(color="rgba(255,215,0,0.25)", width=1, dash="dot"),
            name="Lower Bound",
            showlegend=False,
        ))

        # Forecast price line
        fig_fc.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df["predicted"].tolist(),
            name="Forecast",
            line=dict(color="#a78bfa", width=2.5, dash="dash"),
            mode="lines+markers",
            marker=dict(color="#a78bfa", size=7, symbol="circle"),
        ))

        layout_fc = gl(f"{horizon}-Day Gold Price Forecast  (USD / troy oz)", 440)
        layout_fc.update({"xaxis": {"tickangle": -30}})
        fig_fc.update_layout(**layout_fc)
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Forecast table ────────────────────────────────────────────────────
        st.markdown('<div class="s-hdr">Forecast Table</div>', unsafe_allow_html=True)

        display_df = forecast_df.copy()
        display_df["Predicted"]  = display_df["predicted"].apply(lambda x: f"${x:,.2f}")
        display_df["Low"]        = display_df["lower"].apply(lambda x: f"${x:,.2f}")
        display_df["High"]       = display_df["upper"].apply(lambda x: f"${x:,.2f}")
        display_df["Change ($)"] = display_df["change_usd"].apply(
            lambda x: f"+${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
        )
        display_df["Change (%)"] = display_df["change_pct"].apply(
            lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%"
        )
        display_df["Day"]  = display_df["day"].apply(lambda x: f"+{x}")
        display_df["Date"] = display_df["date"]

        st.dataframe(
            display_df[["Day", "Date", "Predicted", "Low", "High", "Change ($)", "Change (%)"]],
            hide_index=True,
            use_container_width=True,
        )

        # ── Summary metric cards ──────────────────────────────────────────────
        last_fc   = forecast_df.iloc[-1]
        best_day  = forecast_df.loc[forecast_df["predicted"].idxmax()]
        worst_day = forecast_df.loc[forecast_df["predicted"].idxmin()]
        avg_pred  = forecast_df["predicted"].mean()

        fc_s1, fc_s2, fc_s3, fc_s4 = st.columns(4)
        with fc_s1:
            end_chg = last_fc["change_pct"]
            end_cls = "up" if end_chg >= 0 else "down"
            st.markdown(card(
                f"Day +{int(last_fc['day'])} Forecast",
                f"${last_fc['predicted']:,.2f}",
                f'<span class="{end_cls}">{"+" if end_chg >= 0 else ""}{end_chg:.2f}% from today</span>',
            ), unsafe_allow_html=True)
        with fc_s2:
            st.markdown(card(
                f"Period Average",
                f"${avg_pred:,.2f}",
                f"Across {horizon} trading days",
            ), unsafe_allow_html=True)
        with fc_s3:
            st.markdown(card(
                "Forecast High",
                f"${best_day['predicted']:,.2f}",
                f"Day +{int(best_day['day'])} ({best_day['date']})",
            ), unsafe_allow_html=True)
        with fc_s4:
            st.markdown(card(
                "Forecast Low",
                f"${worst_day['predicted']:,.2f}",
                f"Day +{int(worst_day['day'])} ({worst_day['date']})",
            ), unsafe_allow_html=True)

    else:
        st.info("Set your horizon above and click 'Run Forecast' to generate the forecast.", icon="📅")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP SECTION — full width below the two columns
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("---")
    st.markdown('<div class="s-hdr">🔍 SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
      <strong>What is SHAP?</strong> Feature importance shows which inputs matter
      globally across all predictions. SHAP goes further — it shows exactly how
      much each feature pushed <em>today's specific prediction</em> up or down
      from the model's baseline. Every bar below is a dollar amount contribution.<br><br>
      <strong>How to read it:</strong>
      🟢 Green bars pushed the prediction <strong>higher</strong> &nbsp;·&nbsp;
      🔴 Red bars pushed it <strong>lower</strong> &nbsp;·&nbsp;
      The baseline is the model's average prediction across all training data.
    </div>""", unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.warning("SHAP not installed. Add `shap` to requirements.txt and redeploy.", icon="⚠️")

    elif shap_values is None:
        st.info("Train the model to generate SHAP explanations.", icon="💡")

    else:
        shap_col1, shap_col2 = st.columns([1, 1], gap="large")

        # ── Waterfall chart — today's prediction breakdown ────────────────────
        with shap_col1:
            st.markdown('<div class="s-hdr">Today\'s Prediction — Feature Contributions</div>',
                        unsafe_allow_html=True)
            st.caption(
                "Why did the model predict this specific price today? "
                "Each bar shows one feature's dollar contribution."
            )

            last_row     = df_features[feature_names].iloc[-1:]
            waterfall_df = get_waterfall_data(
                shap_values, feature_names,
                last_row.iloc[0], shap_expected, LABEL_MAP,
            )

            if not waterfall_df.empty:
                final_pred = shap_expected + waterfall_df["shap_value"].sum()

                w_colors = [
                    "#4ade80" if v >= 0 else "#f87171"
                    for v in waterfall_df["shap_value"]
                ]

                fig_w = go.Figure()

                # Baseline marker
                fig_w.add_trace(go.Scatter(
                    x=[shap_expected] * len(waterfall_df),
                    y=waterfall_df["label"],
                    mode="markers",
                    marker=dict(color="rgba(255,215,0,0.2)", size=6, symbol="line-ns"),
                    name="Baseline",
                    showlegend=False,
                ))

                # SHAP contribution bars
                fig_w.add_trace(go.Bar(
                    x=waterfall_df["shap_value"],
                    y=waterfall_df["label"],
                    orientation="h",
                    marker_color=w_colors,
                    marker_line_width=0,
                    name="SHAP contribution",
                    text=[f"{'+' if v >= 0 else ''}{v:.2f}" for v in waterfall_df["shap_value"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                ))

                fig_w.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")

                layout_w = gl("", 440)
                layout_w.update({
                    "xaxis": {
                        **layout_w.get("xaxis", {}),
                        "title": "SHAP Value (USD contribution to prediction)",
                        "tickprefix": "$",
                    },
                    "barmode": "relative",
                    "showlegend": False,
                })
                fig_w.update_layout(**layout_w)
                st.plotly_chart(fig_w, use_container_width=True)

                # Baseline + total callout
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.markdown(card("Baseline Value",
                                     f"${shap_expected:,.2f}",
                                     "Model's avg prediction"), unsafe_allow_html=True)
                with bc2:
                    net = waterfall_df["shap_value"].sum()
                    ns  = "+" if net >= 0 else ""
                    st.markdown(card("Feature Adjustment",
                                     f'<span class="{"up" if net >= 0 else "down"}">'
                                     f'{ns}${net:.2f}</span>',
                                     "Sum of all SHAP contributions"), unsafe_allow_html=True)
                with bc3:
                    st.markdown(card("Explained Prediction",
                                     f"${final_pred:,.2f}",
                                     "Baseline + adjustment"), unsafe_allow_html=True)

        # ── Summary chart — average SHAP impact across all test predictions ───
        with shap_col2:
            st.markdown('<div class="s-hdr">Average SHAP Impact — Test Period</div>',
                        unsafe_allow_html=True)
            st.caption(
                "Average contribution of each feature across all test predictions. "
                "Green = tends to push predictions higher. Red = tends to push lower."
            )

            summary_df = get_summary_data(shap_values, feature_names, LABEL_MAP)

            if not summary_df.empty:
                s_colors = [
                    "#4ade80" if v >= 0 else "#f87171"
                    for v in summary_df["mean_shap"]
                ]

                fig_s = go.Figure(go.Bar(
                    x=summary_df["mean_shap"],
                    y=summary_df["label"],
                    orientation="h",
                    marker_color=s_colors,
                    marker_line_width=0,
                    text=[f"{'+' if v >= 0 else ''}{v:.2f}"
                          for v in summary_df["mean_shap"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                ))

                fig_s.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")

                layout_s = gl("", 440)
                layout_s.update({
                    "xaxis": {
                        **layout_s.get("xaxis", {}),
                        "title": "Mean SHAP Value (USD)",
                        "tickprefix": "$",
                    },
                    "showlegend": False,
                })
                fig_s.update_layout(**layout_s)
                st.plotly_chart(fig_s, use_container_width=True)

                # Key insights callout
                top_pos = summary_df[summary_df["mean_shap"] > 0].iloc[-1] \
                          if len(summary_df[summary_df["mean_shap"] > 0]) > 0 else None
                top_neg = summary_df[summary_df["mean_shap"] < 0].iloc[0] \
                          if len(summary_df[summary_df["mean_shap"] < 0]) > 0 else None

                if top_pos is not None or top_neg is not None:
                    st.markdown("""
                    <div class="info-pill" style="margin-top:12px;">
                      <strong>💡 How to read this:</strong> Features with large positive
                      values tend to push gold price predictions higher on average —
                      e.g. a high VIX typically signals market fear which drives gold
                      demand. Large negative values push predictions lower — e.g. a
                      strong USD typically suppresses gold prices.
                    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card("RMSE",     f"${metrics['rmse']:.2f}",
                               "Avg error in dollars"),          unsafe_allow_html=True)
    with c2: st.markdown(card("MAE",      f"${metrics['mae']:.2f}",
                               "Mean absolute error"),           unsafe_allow_html=True)
    with c3: st.markdown(card("R² Score", f"{metrics['r2']:.4f}",
                               f"Explains {metrics['r2']*100:.1f}% of variance"),
                          unsafe_allow_html=True)
    with c4: st.markdown(card("MAPE",     f"{metrics['mape']:.2f}%",
                               "% error relative to price"),    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="s-hdr">Actual vs Predicted Price — Test Period</div>',
                unsafe_allow_html=True)

    actual = np.array(y_test)
    errors = actual - predictions
    dates  = test_df.index

    fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.68, 0.32], vertical_spacing=0.05,
                          subplot_titles=("Price: Actual vs Predicted", "Prediction Error ($)"))
    fig_p.add_trace(go.Scatter(x=dates, y=actual,      name="Actual",
                               line=dict(color="#FFD700", width=2)),      row=1, col=1)
    fig_p.add_trace(go.Scatter(x=dates, y=predictions, name="Predicted",
                               line=dict(color="#60a5fa", width=2, dash="dot")), row=1, col=1)
    fig_p.add_trace(go.Scatter(x=dates, y=errors,      name="Error",
                               fill="tozeroy",
                               fillcolor="rgba(248,113,113,0.12)",
                               line=dict(color="#f87171", width=1)),      row=2, col=1)
    fig_p.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)", row=2, col=1)
    fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,22,0.8)",
                        font=dict(color="#94a3b8"), height=500,
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        margin=dict(l=8, r=8, t=40, b=8))
    fig_p.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
    fig_p.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown('<div class="s-hdr">Error Analysis</div>', unsafe_allow_html=True)
    ea1, ea2 = st.columns(2)
    with ea1:
        fig_h = go.Figure(go.Histogram(x=errors, nbinsx=40,
                                       marker_color="#FFD700", opacity=0.75))
        fig_h.add_vline(x=0, line_dash="dash", line_color="#f87171",
                        annotation_text="Zero error")
        fig_h.update_layout(**gl("Error Distribution", 300))
        st.plotly_chart(fig_h, use_container_width=True)
    with ea2:
        mn = min(actual.min(), predictions.min())
        mx = max(actual.max(), predictions.max())
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=actual, y=predictions, mode="markers",
                                   marker=dict(color="#FFD700", opacity=0.4, size=4)))
        fig_s.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                   name="Perfect fit",
                                   line=dict(color="#4ade80", dash="dash", width=2)))
        fig_s.update_layout(**gl("Actual vs Predicted Scatter", 300))
        st.plotly_chart(fig_s, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MONITORING
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    left_m, right_m = st.columns([1, 1], gap="large")

    with left_m:
        st.markdown('<div class="s-hdr">📊 PSI Drift Detection</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-pill">
          <strong>What is drift?</strong> When the distribution of incoming gold price
          data shifts significantly from what the model was trained on, predictions
          start becoming less reliable. I use two tests here:<br><br>
          <strong>PSI (Population Stability Index)</strong> — a financial industry standard
          for measuring distribution shift. Originally developed for credit risk models.<br>
          <strong>KS Test</strong> — a statistical hypothesis test for whether two
          distributions are significantly different (p &lt; 0.05 = significant).<br><br>
          🟢 PSI &lt; 0.10 Stable &nbsp;·&nbsp;
          🟡 0.10–0.25 Monitor &nbsp;·&nbsp;
          🔴 ≥ 0.25 Retrain
        </div>""", unsafe_allow_html=True)

        if st.button("🔍  Run Drift Analysis", use_container_width=True):
            with st.spinner("Running PSI + KS tests across all features…"):
                results = run_full_monitoring(
                    X_train, X_test, y_train, y_test,
                    train_preds, predictions,
                )
                st.session_state.monitoring_results = results
            st.success("✅  Monitoring complete!")
            st.rerun()

        if st.session_state.monitoring_results:
            res = st.session_state.monitoring_results
            status = res["overall_status"]
            emoji  = res["overall_emoji"]
            color_cls  = {"Stable": "drift-green", "Monitor": "drift-yellow",
                          "Retrain": "drift-red"}.get(status, "drift-green")
            txt_cls    = {"Stable": "txt-green",   "Monitor": "txt-yellow",
                          "Retrain": "txt-red"}.get(status, "txt-green")

            st.markdown(f"""
            <div class="drift-banner {color_cls}">
              <div class="db-label">Overall Drift Status</div>
              <div class="db-status {txt_cls}">{emoji} {status.upper()}</div>
              <div class="db-sub">
                {res['n_drifted']} of {res['n_total']} features flagged
                &nbsp;·&nbsp; Max PSI: {res['max_psi']:.4f}
              </div>
            </div>""", unsafe_allow_html=True)

            pred_s, _, pred_e = psi_status(res["prediction_psi"])
            err_s,  _, err_e  = psi_status(res["error_psi"])

            pm1, pm2 = st.columns(2)
            with pm1: st.markdown(card("Prediction PSI",
                                       f"{pred_e} {res['prediction_psi']:.4f}", pred_s),
                                  unsafe_allow_html=True)
            with pm2: st.markdown(card("Residual PSI",
                                       f"{err_e} {res['error_psi']:.4f}", err_s),
                                  unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Feature-Level Results:**")
            st.dataframe(
                res["summary_df"],
                hide_index=True,
                use_container_width=True,
                height=320,
            )

            # PSI bar chart
            psi_vals  = [v["psi"]    for v in res["feature_results"].values()]
            psi_feats = list(res["feature_results"].keys())
            psi_colors = [
                "#4ade80" if p < 0.10 else ("#fbbf24" if p < 0.25 else "#f87171")
                for p in psi_vals
            ]
            fig_psi = go.Figure(go.Bar(
                x=psi_vals, y=psi_feats, orientation="h",
                marker_color=psi_colors,
            ))
            fig_psi.add_vline(x=0.10, line_dash="dot", line_color="#fbbf24",
                              annotation_text="0.10")
            fig_psi.add_vline(x=0.25, line_dash="dot", line_color="#f87171",
                              annotation_text="0.25")
            fig_psi.update_layout(**gl("PSI by Feature", 400))
            fig_psi.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_psi, use_container_width=True)
        else:
            st.info("Click 'Run Drift Analysis' to check for feature drift.", icon="ℹ️")

    with right_m:
        # ── MLflow history ────────────────────────────────────────────────────
        st.markdown('<div class="s-hdr">📋 MLflow Experiment History</div>',
                    unsafe_allow_html=True)
        st.caption("Every training run is automatically logged here — compare models, "
                   "track RMSE over time, and audit what changed between runs.")

        history = get_run_history()
        if history.empty:
            st.info("No runs yet. Train the model to start logging.", icon="ℹ️")
        else:
            best = get_best_run()
            if best:
                b1, b2 = st.columns(2)
                with b1: st.markdown(card("Best Model", best["model_type"],
                                          f"RMSE ${best['rmse']:.2f}"),
                                     unsafe_allow_html=True)
                with b2: st.markdown(card("Best R²", f"{best['r2']:.4f}",
                                          f"MAE ${best['mae']:.2f}"),
                                     unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            st.dataframe(history, hide_index=True, use_container_width=True)

            if "RMSE ($)" in history.columns and len(history) > 1:
                colors_map = {"Random Forest": "#60a5fa", "XGBoost": "#f472b6"}
                b_colors   = [colors_map.get(str(m), "#FFD700")
                              for m in history.get("Model", [])]
                x_axis = history["Timestamp"].tolist() \
                         if "Timestamp" in history.columns else list(range(len(history)))
                fig_r = go.Figure(go.Bar(x=x_axis, y=history["RMSE ($)"],
                                         marker_color=b_colors))
                fig_r.update_layout(**gl("RMSE Across Training Runs", 250))
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")
        backend = st.session_state.mlflow_backend
        if backend == "dagshub":
            duser = _read_secret("mlflow", "dagshub_username")
            drepo = _read_secret("mlflow", "dagshub_repo")
            if duser and drepo:
                st.markdown(
                    f"📡 **[Open DagsHub MLflow UI](https://dagshub.com/{duser}/{drepo}.mlflow)**"
                    " — runs persist across deployments."
                )
        else:
            st.markdown("""
**💡 View MLflow UI locally:**
```bash
mlflow ui
```
Open **http://localhost:5000** to browse all runs.
            """)



# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — OPTUNA TRIAL HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="s-hdr">🎯 Optuna Hyperparameter Tuning Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">
      <strong>What is Optuna?</strong> Instead of guessing hyperparameters manually,
      Optuna runs a directed search across the parameter space. Each trial tries a
      different combination of settings and records the validation RMSE. The TPE sampler
      learns from each result and focuses subsequent trials on promising regions —
      much smarter than a grid search.<br><br>
      Enable <strong>Optuna Hyperparameter Tuning</strong> in the sidebar before training
      to populate this dashboard.
    </div>""", unsafe_allow_html=True)

    if optuna_study is None:
        st.info(
            "No Optuna study found. Enable 'Optuna Hyperparameter Tuning' in the "
            "sidebar and train the model to see the trial history here.",
            icon="💡",
        )
    else:
        trials_df, params_df = get_trial_history(optuna_study)

        if trials_df.empty:
            st.warning("Trial history is empty — something may have gone wrong during tuning.")
        else:
            best_rmse    = optuna_study.best_value
            best_trial   = optuna_study.best_trial.number + 1
            n_trials_run = len(trials_df)
            worst_rmse   = float(trials_df["RMSE"].max())
            improvement  = round(((worst_rmse - best_rmse) / worst_rmse) * 100, 1)

            # Top 3 trials sorted by RMSE ascending
            top3 = trials_df.nsmallest(3, "RMSE").reset_index(drop=True)
            medals = ["🥇", "🥈", "🥉"]
            medal_trial_map = {
                int(top3.iloc[i]["Trial"]): medals[i]
                for i in range(len(top3))
            }

            # ── Summary cards — top 3 podium ──────────────────────────────────
            st.markdown('<div class="s-hdr">🏆 Podium — Top 3 Trials</div>',
                        unsafe_allow_html=True)

            podium_cols = st.columns(3)
            podium_colors = ["#FFD700", "#94a3b8", "#cd7f32"]
            for i in range(min(3, len(top3))):
                row = top3.iloc[i]
                with podium_cols[i]:
                    st.markdown(
                        f'<div class="g-card" style="border-color:{podium_colors[i]};">'
                        f'<div class="gc-label">{medals[i]} Rank {i+1}</div>'
                        f'<div class="gc-value" style="color:{podium_colors[i]};">'
                        f'${row["RMSE"]:.2f}</div>'
                        f'<div class="gc-sub">Trial {int(row["Trial"])}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary stat cards
            oc1, oc2, oc3, oc4 = st.columns(4)
            with oc1:
                st.markdown(card("Total Trials Run",
                                 str(n_trials_run),
                                 f"Model: {optuna_mtype}"),
                            unsafe_allow_html=True)
            with oc2:
                st.markdown(card("Best Validation RMSE",
                                 f"${best_rmse:.2f}",
                                 f"🥇 Trial {best_trial}"),
                            unsafe_allow_html=True)
            with oc3:
                st.markdown(card("First Trial RMSE",
                                 f"${worst_rmse:.2f}",
                                 "Before tuning"),
                            unsafe_allow_html=True)
            with oc4:
                st.markdown(card("RMSE Improvement",
                                 f"{improvement}%",
                                 "vs first trial"),
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Convergence chart ─────────────────────────────────────────────
            st.markdown('<div class="s-hdr">RMSE Convergence — All Trials</div>',
                        unsafe_allow_html=True)
            st.caption(
                "Each point is one trial. 🥇🥈🥉 mark the top 3 results. "
                "The green line tracks the best RMSE found so far."
            )

            trials_df["Best So Far"] = trials_df["RMSE"].cummin()

            # Assign colour and size per trial based on rank
            top3_trials = set(medal_trial_map.keys())
            dot_colors = []
            dot_sizes  = []
            dot_borders = []
            for _, row in trials_df.iterrows():
                t = int(row["Trial"])
                if t == best_trial:
                    dot_colors.append("#FFD700")
                    dot_sizes.append(14)
                    dot_borders.append(2)
                elif t in top3_trials:
                    dot_colors.append("#94a3b8")
                    dot_sizes.append(11)
                    dot_borders.append(2)
                else:
                    dot_colors.append("#334155")
                    dot_sizes.append(7)
                    dot_borders.append(0)

            # Hover text with medal if in top 3
            hover_texts = []
            for _, row in trials_df.iterrows():
                t = int(row["Trial"])
                medal = medal_trial_map.get(t, "")
                hover_texts.append(
                    f"{medal} Trial {t}<br>RMSE: ${row['RMSE']:.2f}"
                )

            fig_conv = go.Figure()

            fig_conv.add_trace(go.Scatter(
                x=trials_df["Trial"],
                y=trials_df["RMSE"],
                mode="markers",
                name="Trial RMSE",
                marker=dict(
                    color=dot_colors,
                    size=dot_sizes,
                    line=dict(color="#FFD700", width=dot_borders),
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts,
            ))

            fig_conv.add_trace(go.Scatter(
                x=trials_df["Trial"],
                y=trials_df["Best So Far"],
                mode="lines",
                name="Best So Far",
                line=dict(color="#4ade80", width=2, dash="dot"),
            ))

            # Annotate top 3 on the chart
            annotation_offsets = [(30, -30), (-30, -40), (40, -50)]
            for i in range(min(3, len(top3))):
                row = top3.iloc[i]
                ax, ay = annotation_offsets[i]
                fig_conv.add_annotation(
                    x=row["Trial"], y=row["RMSE"],
                    text=f"{medals[i]} ${row['RMSE']:.2f}",
                    showarrow=True, arrowhead=2,
                    arrowcolor=podium_colors[i],
                    font=dict(color=podium_colors[i], size=10),
                    ax=ax, ay=ay,
                )

            layout_conv = gl("", 400)
            layout_conv.update({
                "xaxis": {**layout_conv.get("xaxis", {}),
                          "title": "Trial Number", "dtick": 1},
                "yaxis": {**layout_conv.get("yaxis", {}),
                          "title": "Validation RMSE ($)", "tickprefix": "$"},
            })
            fig_conv.update_layout(**layout_conv)
            st.plotly_chart(fig_conv, use_container_width=True)

            # ── Best hyperparameters ──────────────────────────────────────────
            st.markdown('<div class="s-hdr">Best Hyperparameters Found</div>',
                        unsafe_allow_html=True)
            st.caption(f"Parameters from trial {best_trial} — used to train the final model.")

            best_params = optuna_study.best_params
            bp_cols     = st.columns(min(len(best_params), 4))
            for i, (param, val) in enumerate(best_params.items()):
                with bp_cols[i % len(bp_cols)]:
                    display_val = f"{val:.4f}" if isinstance(val, float) else str(val)
                    st.markdown(
                        card(param.replace("_", " ").title(), display_val),
                        unsafe_allow_html=True,
                    )

            # ── Full trial history table ──────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="s-hdr">Full Trial History</div>',
                        unsafe_allow_html=True)
            st.caption("Every trial Optuna ran. 🥇🥈🥉 mark the top 3 results.")

            display_params = params_df.copy()
            display_params.insert(
                2, "Rank",
                display_params["Trial"].apply(
                    lambda t: medal_trial_map.get(int(t), "")
                )
            )
            st.dataframe(
                display_params,
                hide_index=True,
                use_container_width=True,
                height=320,
            )

            # ── RMSE distribution ─────────────────────────────────────────────
            st.markdown('<div class="s-hdr">RMSE Distribution Across Trials</div>',
                        unsafe_allow_html=True)
            st.caption("Shows how spread out the trial results were. "
                       "A tight cluster near the left means most configurations performed well.")

            fig_dist = go.Figure(go.Histogram(
                x=trials_df["RMSE"],
                nbinsx=max(5, n_trials_run // 3),
                marker_color="#FFD700",
                opacity=0.75,
            ))
            fig_dist.add_vline(
                x=best_rmse, line_dash="dash", line_color="#4ade80",
                annotation_text=f"Best ${best_rmse:.2f}",
                annotation_font_color="#4ade80",
            )
            layout_dist = gl("", 280)
            layout_dist.update({
                "xaxis": {**layout_dist.get("xaxis", {}),
                          "title": "Validation RMSE ($)", "tickprefix": "$"},
                "yaxis": {**layout_dist.get("yaxis", {}), "title": "Count"},
            })
            fig_dist.update_layout(**layout_dist)
            st.plotly_chart(fig_dist, use_container_width=True)
