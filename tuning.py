# tuning.py
# Optuna hyperparameter tuning for Random Forest and XGBoost.
#
# Instead of manually picking hyperparameters (which I did in v1 by trial and error),
# Optuna runs a directed search across the parameter space — it's smarter than a
# grid search because it learns from each trial and focuses on promising regions.
#
# I use a validation split inside the training data (not the test set) to avoid
# any leakage. The test set is only touched once, at final evaluation.
#
# n_trials controls the trade-off between search quality and time:
#   20 trials  → ~2 mins  — good for quick runs and GitHub Actions
#   50 trials  → ~5 mins  — better coverage, worth it for a manual run
#   100 trials → ~12 mins — near-optimal, overkill for most cases
#
# Optuna suppresses its own logging by default here so the Streamlit UI
# doesn't get flooded with trial output.

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Silence Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_random_forest(
    X_train, y_train,
    n_trials: int = 30,
    val_frac: float = 0.15,
) -> tuple:
    """
    Runs an Optuna study to find the best Random Forest hyperparameters.

    I carve out a small validation set from the end of the training data
    (chronological, not random) to score each trial. This mimics how the
    model will actually be used — predicting forward in time.

    Args:
        X_train:   Training features
        y_train:   Training targets
        n_trials:  Number of Optuna trials to run
        val_frac:  Fraction of training data to hold out for validation

    Returns:
        (best_params dict, best_rmse float)
    """
    split = int(len(X_train) * (1 - val_frac))
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      100, 500),
            "max_depth":         trial.suggest_int("max_depth",            4,  20),
            "min_samples_split": trial.suggest_int("min_samples_split",    2,  10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",     1,   5),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
            "random_state":      42,
            "n_jobs":           -1,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, round(study.best_value, 4)


def tune_xgboost(
    X_train, y_train,
    n_trials: int = 30,
    val_frac: float = 0.15,
) -> tuple:
    """
    Runs an Optuna study to find the best XGBoost hyperparameters.

    Same validation approach as Random Forest — chronological split from
    the end of training data.

    Returns:
        (best_params dict, best_rmse float)
    """
    split = int(len(X_train) * (1 - val_frac))
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",          100, 500),
            "max_depth":        trial.suggest_int("max_depth",                3,  10),
            "learning_rate":    trial.suggest_float("learning_rate",       0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample",            0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",    0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight",        1,   7),
            "gamma":            trial.suggest_float("gamma",                0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",            0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",           0.5, 2.0),
            "random_state":     42,
            "verbosity":         0,
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, round(study.best_value, 4)


def run_tuning(model_type: str, X_train, y_train,
               n_trials: int = 30) -> tuple:
    """
    Dispatcher — runs the right tuning function based on model type.

    Returns:
        (best_params dict, best_rmse float)
    """
    if model_type == "Random Forest":
        return tune_random_forest(X_train, y_train, n_trials)
    else:
        return tune_xgboost(X_train, y_train, n_trials)
