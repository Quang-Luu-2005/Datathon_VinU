"""Model training, tuning, blending, and recursive prediction logic."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .constants import DATE_COL, DEFAULT_TUNED_PARAMS, EVENT_COLS
from .cv import ExpandingWindowWalkForward
from .data_io import seasonal_naive_predict
from .features import lag_roll_row_from_history


def tune_lgbm_l1(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    splitter: ExpandingWindowWalkForward,
    n_trials: int,
    seed: int,
    timeout_s: int = 0,
) -> Dict[str, float]:
    y_full = train_df[target_col].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 120),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "max_bin": trial.suggest_int("max_bin", 63, 255),
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "verbose": -1,
            "num_threads": -1,
        }
        maes: List[float] = []
        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(train_df)):
            x_tr = train_df.loc[tr_idx, feature_cols]
            y_tr = np.log1p(train_df.loc[tr_idx, target_col].clip(lower=0.0))
            x_va = train_df.loc[va_idx, feature_cols]
            y_va = y_full[va_idx]

            dtr = lgb.Dataset(x_tr, label=y_tr)
            dva = lgb.Dataset(x_va, label=np.log1p(np.clip(y_va, a_min=0.0, a_max=None)))
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=3000,
                valid_sets=[dva],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
            )
            pred = np.expm1(booster.predict(x_va, num_iteration=booster.best_iteration))
            pred = np.clip(pred, a_min=0.0, a_max=None)
            maes.append(float(mean_absolute_error(y_va, pred)))
            trial.report(float(np.mean(maes)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    if n_trials <= 0:
        return dict(DEFAULT_TUNED_PARAMS)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max(1, splitter.max_splits)),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=(None if timeout_s <= 0 else timeout_s),
        show_progress_bar=False,
    )
    return dict(study.best_params)


def build_l1_params(best_params: Dict[str, float], seed: int) -> Dict[str, float]:
    return {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": float(best_params["learning_rate"]),
        "num_leaves": int(best_params["num_leaves"]),
        "max_depth": int(best_params["max_depth"]),
        "min_data_in_leaf": int(best_params["min_data_in_leaf"]),
        "feature_fraction": float(best_params["feature_fraction"]),
        "bagging_fraction": float(best_params["bagging_fraction"]),
        "bagging_freq": int(best_params["bagging_freq"]),
        "lambda_l1": float(best_params["lambda_l1"]),
        "lambda_l2": float(best_params["lambda_l2"]),
        "min_gain_to_split": float(best_params["min_gain_to_split"]),
        "max_bin": int(best_params["max_bin"]),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "verbose": -1,
        "num_threads": -1,
    }


def build_tweedie_params(best_params: Dict[str, float], seed: int) -> Dict[str, float]:
    return {
        "objective": "tweedie",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "tweedie_variance_power": 1.3,
        "learning_rate": float(best_params["learning_rate"]),
        "num_leaves": int(best_params["num_leaves"]),
        "max_depth": int(best_params["max_depth"]),
        "min_data_in_leaf": int(best_params["min_data_in_leaf"]),
        "feature_fraction": float(best_params["feature_fraction"]),
        "bagging_fraction": float(best_params["bagging_fraction"]),
        "bagging_freq": int(best_params["bagging_freq"]),
        "lambda_l1": float(best_params["lambda_l1"]),
        "lambda_l2": float(best_params["lambda_l2"]),
        "min_gain_to_split": float(best_params["min_gain_to_split"]),
        "max_bin": int(best_params["max_bin"]),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "verbose": -1,
        "num_threads": -1,
    }


def train_prophet(train_dates: pd.Series, train_y: np.ndarray, seed: int) -> Prophet:
    prophet_train = pd.DataFrame({"ds": train_dates.values, "y": train_y})
    model = Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        uncertainty_samples=0,
    )
    model.add_country_holidays(country_name="VN")
    model.fit(prophet_train, seed=seed)
    return model


def predict_prophet_safe(
    train_dates: pd.Series,
    train_y: np.ndarray,
    pred_dates: pd.Series,
    seed: int,
) -> np.ndarray:
    try:
        model = train_prophet(train_dates, train_y, seed=seed)
        fcst = model.predict(pd.DataFrame({"ds": pred_dates.values}))
        return np.clip(fcst["yhat"].to_numpy(dtype=float), a_min=0.0, a_max=None)
    except Exception:
        hist = pd.Series(train_y, index=pd.DatetimeIndex(train_dates))
        return seasonal_naive_predict(hist, pd.DatetimeIndex(pred_dates), seasonal_lag=365)


def generate_oof_predictions(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    splitter: ExpandingWindowWalkForward,
    l1_params: Dict[str, float],
    tweedie_params: Dict[str, float],
    seed: int,
    use_prophet: bool = True,
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    oof = train_df[[DATE_COL, target_col] + EVENT_COLS].copy()
    oof["pred_l1"] = np.nan
    oof["pred_tweedie"] = np.nan
    oof["pred_prophet"] = np.nan

    best_iters_l1: List[int] = []
    best_iters_tw: List[int] = []

    for tr_idx, va_idx in splitter.split(train_df):
        x_tr = train_df.loc[tr_idx, feature_cols]
        y_tr = train_df.loc[tr_idx, target_col].to_numpy()
        x_va = train_df.loc[va_idx, feature_cols]

        y_tr_log = np.log1p(np.clip(y_tr, a_min=0.0, a_max=None))
        dtr_l1 = lgb.Dataset(x_tr, label=y_tr_log)
        dva_l1 = lgb.Dataset(x_va, label=np.log1p(np.clip(train_df.loc[va_idx, target_col].to_numpy(), 0, None)))
        booster_l1 = lgb.train(
            l1_params,
            dtr_l1,
            num_boost_round=3000,
            valid_sets=[dva_l1],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        pred_l1 = np.expm1(booster_l1.predict(x_va, num_iteration=booster_l1.best_iteration))
        oof.loc[va_idx, "pred_l1"] = np.clip(pred_l1, a_min=0.0, a_max=None)
        best_iters_l1.append(int(booster_l1.best_iteration))

        dtr_tw = lgb.Dataset(x_tr, label=y_tr)
        dva_tw = lgb.Dataset(x_va, label=train_df.loc[va_idx, target_col].to_numpy())
        booster_tw = lgb.train(
            tweedie_params,
            dtr_tw,
            num_boost_round=3000,
            valid_sets=[dva_tw],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        pred_tw = booster_tw.predict(x_va, num_iteration=booster_tw.best_iteration)
        oof.loc[va_idx, "pred_tweedie"] = np.clip(pred_tw, a_min=0.0, a_max=None)
        best_iters_tw.append(int(booster_tw.best_iteration))

        if use_prophet:
            pred_prophet = predict_prophet_safe(
                train_dates=train_df.loc[tr_idx, DATE_COL],
                train_y=y_tr,
                pred_dates=train_df.loc[va_idx, DATE_COL],
                seed=seed,
            )
            oof.loc[va_idx, "pred_prophet"] = pred_prophet
        else:
            sn = seasonal_naive_predict(
                pd.Series(y_tr, index=pd.DatetimeIndex(train_df.loc[tr_idx, DATE_COL].values)),
                pd.DatetimeIndex(train_df.loc[va_idx, DATE_COL].values),
                seasonal_lag=365,
            )
            oof.loc[va_idx, "pred_prophet"] = sn

    return oof, best_iters_l1, best_iters_tw


def fit_blend_weights(oof: pd.DataFrame, lam: float = 0.01) -> Tuple[np.ndarray, float]:
    usable = oof.dropna(subset=["pred_l1", "pred_tweedie", "pred_prophet"])
    y = usable["actual"].to_numpy()
    preds = usable[["pred_l1", "pred_tweedie", "pred_prophet"]].to_numpy()

    n_models = preds.shape[1]

    def loss(w: np.ndarray) -> float:
        yhat = preds @ w
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return rmse + lam * float(np.sum(w**2))

    if n_models == 3:
        best_w = np.array([1.0, 0.0, 0.0], dtype=float)
        best_loss = float("inf")
        step = 0.01
        grid = np.arange(0.0, 1.0 + step, step)
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < 0.0:
                    continue
                w = np.array([w0, w1, w2], dtype=float)
                cur = loss(w)
                if cur < best_loss:
                    best_loss = cur
                    best_w = w
        return best_w, best_loss

    init = np.ones(n_models) / n_models
    res = minimize(
        loss,
        init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        per_model_rmse = np.sqrt(np.mean((preds - y.reshape(-1, 1)) ** 2, axis=0))
        inv = 1.0 / np.maximum(per_model_rmse, 1e-6)
        w = inv / inv.sum()
        return w.astype(float), float(loss(w))

    w = np.clip(res.x, 0.0, None)
    if w.sum() <= 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return w, float(loss(w))


def compute_event_multipliers(oof: pd.DataFrame, k: float = 10.0) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for event_col in EVENT_COLS:
        subset = oof[oof[event_col] == 1].dropna(subset=["actual", "pred_blend"])
        if len(subset) >= 3 and subset["pred_blend"].sum() > 0:
            ratio = float(subset["actual"].sum() / subset["pred_blend"].sum())
            n = float(len(subset))
            out[event_col] = float((n * ratio + k * 1.0) / (n + k))
        else:
            out[event_col] = 1.0
    return out


def apply_event_multiplier(pred_value: float, event_row: pd.Series, event_multipliers: Dict[str, float]) -> float:
    m = 1.0
    for event_col, factor in event_multipliers.items():
        if int(event_row.get(event_col, 0)) == 1:
            m *= factor
    return float(pred_value * m)


def fit_residual_sarima(oof: pd.DataFrame) -> Optional[SARIMAX]:
    residual = oof.dropna(subset=["pred_blend"]).copy()
    if residual.empty:
        return None
    residual["resid"] = residual["actual"] - residual["pred_blend"]
    series = residual.set_index(DATE_COL)["resid"].asfreq("D").fillna(0.0)
    if len(series) < 365:
        return None
    lb = acorr_ljungbox(series, lags=[7, 14, 28], return_df=True)
    if not (lb["lb_pvalue"] < 0.05).any():
        return None
    try:
        return SARIMAX(
            series,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception:
        return None


def recursive_forecast_lgb(
    history_series: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    deterministic_features: pd.DataFrame,
    feature_cols: List[str],
    feature_medians: pd.Series,
    model_l1: lgb.Booster,
    model_tw: lgb.Booster,
    prophet_preds: pd.Series,
    blend_weights: np.ndarray,
    event_multipliers: Dict[str, float],
    residual_fcst: Optional[pd.Series],
) -> pd.Series:
    history = history_series.copy()
    out_preds: List[float] = []

    for dt in forecast_dates:
        lag_feats = lag_roll_row_from_history(history)
        row = deterministic_features.loc[dt].to_dict()
        row.update(lag_feats)

        x_row = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_cols}]).fillna(feature_medians)
        pred_l1 = float(np.expm1(model_l1.predict(x_row, num_iteration=model_l1.best_iteration)[0]))
        pred_tw = float(model_tw.predict(x_row, num_iteration=model_tw.best_iteration)[0])
        pred_pr = float(prophet_preds.loc[dt]) if dt in prophet_preds.index else float(pred_tw)

        pred = (
            blend_weights[0] * max(pred_l1, 0.0)
            + blend_weights[1] * max(pred_tw, 0.0)
            + blend_weights[2] * max(pred_pr, 0.0)
        )
        pred = apply_event_multiplier(pred, deterministic_features.loc[dt], event_multipliers)
        if residual_fcst is not None and dt in residual_fcst.index:
            pred += float(residual_fcst.loc[dt])
        pred = max(float(pred), 0.0)

        history.loc[dt] = pred
        out_preds.append(pred)

    return pd.Series(out_preds, index=forecast_dates)

