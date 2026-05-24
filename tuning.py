# tuning.py
# Optuna hyperparameter tuning for Random Forest and XGBoost.
#
# Instead of manually picking hyperparameters, Optuna runs a directed search
# across the parameter space using a TPE (Tree-structured Parzen Estimator) sampler —
# it learns from each trial and focuses on promising regions rather than searching blindly.
#
# I use a chronological validation split inside the training data to score each trial.
# The test set is never touched during tuning.
#
# n_trials controls the trade-off between search quality and time:
#   10 trials → ~1 min  — quick, decent improvement
#   30 trials → ~3 mins — good coverage
#   50 trials → ~5 mins — near-optimal for most cases
#
# All three tuning functions now return the full study object alongside
# best_params and best_rmse so the UI can display the trial history dashboard.

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_random_forest(
    X_train, y_train,
    n_trials: int = 20,
    val_frac: float = 0.15,
) -> tuple:
    """
    Runs an Optuna study to find the best Random Forest hyperparameters.
    Uses a chronological validation split from the end of training data.

    Returns:
        (best_params dict, best_rmse float, study object)
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

    return study.best_params, round(study.best_value, 4), study


def tune_xgboost(
    X_train, y_train,
    n_trials: int = 20,
    val_frac: float = 0.15,
) -> tuple:
    """
    Runs an Optuna study to find the best XGBoost hyperparameters.
    Uses a chronological validation split from the end of training data.

    Returns:
        (best_params dict, best_rmse float, study object)
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

    return study.best_params, round(study.best_value, 4), study


def run_tuning(model_type: str, X_train, y_train,
               n_trials: int = 20) -> tuple:
    """
    Dispatcher — runs the right tuning function based on model type.

    Returns:
        (best_params dict, best_rmse float, study object)
    """
    if model_type == "Random Forest":
        return tune_random_forest(X_train, y_train, n_trials)
    else:
        return tune_xgboost(X_train, y_train, n_trials)


def get_trial_history(study) -> tuple:
    """
    Extracts trial history from a completed Optuna study into clean DataFrames
    ready for the dashboard charts and table.

    Returns:
        (trials_df, params_df)
        trials_df — trial number, RMSE, is_best flag
        params_df — trial number + all hyperparameter values
    """
    import pandas as pd

    if study is None:
        return pd.DataFrame(), pd.DataFrame()

    trials = []
    params_list = []
    best_val = study.best_value

    for t in study.trials:
        if t.value is None:
            continue
        trials.append({
            "Trial":   t.number + 1,
            "RMSE":    round(t.value, 4),
            "Is Best": t.value == best_val,
        })
        row = {"Trial": t.number + 1, "RMSE": round(t.value, 4)}
        row.update({k: round(v, 4) if isinstance(v, float) else v
                    for k, v in t.params.items()})
        params_list.append(row)

    trials_df = pd.DataFrame(trials)
    params_df = pd.DataFrame(params_list)
    return trials_df, params_df
