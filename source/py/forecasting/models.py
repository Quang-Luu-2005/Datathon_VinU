"""Model training, blending, calibration, and recursive prediction logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from scipy.optimize import minimize, nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .constants import DATE_COL, DEFAULT_TUNED_PARAMS, EVENT_COLS, HORIZON_BUCKETS
from .cv import FoldIndices
from .data_io import seasonal_naive_predict
from .features import lag_roll_row_from_history

try:
    from catboost import CatBoostRegressor

    _HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    _HAS_CATBOOST = False


def get_h_bucket(horizon_day: int) -> str:
    for bucket_name, lo, hi in HORIZON_BUCKETS:
        if lo <= horizon_day <= hi:
            return bucket_name
    return HORIZON_BUCKETS[-1][0]


def tune_lgbm_l1(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    short_folds: List[FoldIndices],
    n_trials: int,
    seed: int,
    timeout_s: int = 0,
) -> Dict[str, float]:
    y_full = train_df[target_col].to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
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
        for fold in short_folds:
            tr_idx, va_idx = fold.train_idx, fold.val_idx
            x_tr = train_df.loc[tr_idx, feature_cols]
            y_tr = np.log1p(np.clip(train_df.loc[tr_idx, target_col].to_numpy(dtype=float), 0.0, None))
            x_va = train_df.loc[va_idx, feature_cols]
            y_va = y_full[va_idx]

            dtr = lgb.Dataset(x_tr, label=y_tr)
            dva = lgb.Dataset(x_va, label=np.log1p(np.clip(y_va, 0.0, None)))
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=3000,
                valid_sets=[dva],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
            )
            pred = np.expm1(booster.predict(x_va, num_iteration=booster.best_iteration))
            pred = np.clip(pred, a_min=0.0, a_max=None)
            maes.append(float(mean_absolute_error(y_va, pred)))
            trial.report(float(np.mean(maes)), step=fold.fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    if n_trials <= 0:
        return dict(DEFAULT_TUNED_PARAMS)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max(1, len(short_folds))),
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


def build_l2_params(best_params: Dict[str, float], seed: int) -> Dict[str, float]:
    p = build_l1_params(best_params, seed)
    p["objective"] = "regression"
    p["metric"] = "rmse"
    return p


def build_tweedie_params(best_params: Dict[str, float], seed: int) -> Dict[str, float]:
    p = build_l1_params(best_params, seed)
    p["objective"] = "tweedie"
    p["metric"] = "rmse"
    p["tweedie_variance_power"] = 1.15
    p["max_bin"] = min(int(best_params["max_bin"]), 63)
    return p


def build_quantile_params(best_params: Dict[str, float], seed: int) -> Dict[str, float]:
    p = build_l1_params(best_params, seed)
    p["objective"] = "quantile"
    p["metric"] = "quantile"
    p["alpha"] = 0.5
    return p


def train_prophet(train_dates: pd.Series, train_y: np.ndarray, seed: int) -> Prophet:
    prophet_train = pd.DataFrame({"ds": train_dates.values, "y": train_y})
    model = Prophet(
        growth="linear",
        yearly_seasonality=10,
        weekly_seasonality=3,
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


def _fit_catboost_and_predict(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    seed: int,
) -> Tuple[Optional[CatBoostRegressor], Optional[np.ndarray]]:
    if not _HAS_CATBOOST:
        return None, None
    try:
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            devices="0",
            border_count=128,
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            od_type="Iter",
            od_wait=300,
            use_best_model=True,
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
        )
        ytr = np.log1p(np.clip(y_tr, 0.0, None))
        yva = np.log1p(np.clip(y_va, 0.0, None))
        model.fit(x_tr, ytr, eval_set=(x_va, yva))
        pred = np.expm1(model.predict(x_va))
        return model, np.clip(pred.astype(float), 0.0, None)
    except Exception:
        return None, None


@dataclass
class OOFResult:
    oof: pd.DataFrame
    best_iters: Dict[str, List[int]]
    used_heads: List[str]


def _train_one_lgb_head(
    params: Dict[str, float],
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    log_target: bool,
    num_boost_round: int = 3000,
) -> Tuple[lgb.Booster, np.ndarray]:
    ytr = np.log1p(np.clip(y_tr, 0.0, None)) if log_target else y_tr
    yva = np.log1p(np.clip(y_va, 0.0, None)) if log_target else y_va
    dtr = lgb.Dataset(x_tr, label=ytr)
    dva = lgb.Dataset(x_va, label=yva)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    pred = booster.predict(x_va, num_iteration=booster.best_iteration)
    if log_target:
        pred = np.expm1(pred)
    return booster, np.clip(pred.astype(float), 0.0, None)


def generate_oof_predictions(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    folds: List[FoldIndices],
    l1_params: Dict[str, float],
    l2_params: Optional[Dict[str, float]],
    tweedie_params: Optional[Dict[str, float]],
    quantile_params: Optional[Dict[str, float]],
    seed: int,
    use_prophet: bool = True,
    use_catboost: bool = True,
) -> OOFResult:
    oof = train_df[[DATE_COL, target_col] + EVENT_COLS].copy()
    oof["fold_type"] = ""
    oof["fold_id"] = -1
    oof["horizon_day"] = np.nan
    oof["h_bucket"] = ""
    oof["pred_l1"] = np.nan
    oof["pred_l2"] = np.nan
    oof["pred_tweedie"] = np.nan
    oof["pred_quantile"] = np.nan
    oof["pred_catboost"] = np.nan
    oof["pred_prophet"] = np.nan

    used_heads: List[str] = ["pred_l1", "pred_prophet"]
    if l2_params is not None:
        used_heads.append("pred_l2")
    if tweedie_params is not None:
        used_heads.append("pred_tweedie")
    if quantile_params is not None:
        used_heads.append("pred_quantile")
    if use_catboost and _HAS_CATBOOST:
        used_heads.append("pred_catboost")

    best_iters: Dict[str, List[int]] = {k: [] for k in used_heads if k != "pred_prophet" and k != "pred_catboost"}

    for fold in folds:
        tr_idx, va_idx = fold.train_idx, fold.val_idx
        x_tr = train_df.loc[tr_idx, feature_cols]
        y_tr = train_df.loc[tr_idx, target_col].to_numpy(dtype=float)
        x_va = train_df.loc[va_idx, feature_cols]
        y_va = train_df.loc[va_idx, target_col].to_numpy(dtype=float)

        l1_booster, pred_l1 = _train_one_lgb_head(l1_params, x_tr, y_tr, x_va, y_va, log_target=True)
        oof.loc[va_idx, "pred_l1"] = pred_l1
        best_iters["pred_l1"].append(int(l1_booster.best_iteration))

        if l2_params is not None:
            l2_booster, pred_l2 = _train_one_lgb_head(l2_params, x_tr, y_tr, x_va, y_va, log_target=False)
            oof.loc[va_idx, "pred_l2"] = pred_l2
            best_iters["pred_l2"].append(int(l2_booster.best_iteration))

        if tweedie_params is not None:
            tw_booster, pred_tw = _train_one_lgb_head(tweedie_params, x_tr, y_tr, x_va, y_va, log_target=False)
            oof.loc[va_idx, "pred_tweedie"] = pred_tw
            best_iters["pred_tweedie"].append(int(tw_booster.best_iteration))

        if quantile_params is not None:
            q_booster, pred_q = _train_one_lgb_head(quantile_params, x_tr, y_tr, x_va, y_va, log_target=True)
            oof.loc[va_idx, "pred_quantile"] = pred_q
            best_iters["pred_quantile"].append(int(q_booster.best_iteration))

        if use_catboost:
            _, cat_pred = _fit_catboost_and_predict(x_tr, y_tr, x_va, y_va, seed=seed + fold.fold_id)
            if cat_pred is not None:
                oof.loc[va_idx, "pred_catboost"] = cat_pred

        if use_prophet:
            pred_prophet = predict_prophet_safe(
                train_dates=train_df.loc[tr_idx, DATE_COL],
                train_y=y_tr,
                pred_dates=train_df.loc[va_idx, DATE_COL],
                seed=seed + fold.fold_id,
            )
            oof.loc[va_idx, "pred_prophet"] = pred_prophet
        else:
            sn = seasonal_naive_predict(
                pd.Series(y_tr, index=pd.DatetimeIndex(train_df.loc[tr_idx, DATE_COL].values)),
                pd.DatetimeIndex(train_df.loc[va_idx, DATE_COL].values),
                seasonal_lag=365,
            )
            oof.loc[va_idx, "pred_prophet"] = sn

        horizon_days = np.arange(1, len(va_idx) + 1)
        oof.loc[va_idx, "fold_type"] = fold.fold_type
        oof.loc[va_idx, "fold_id"] = fold.fold_id
        oof.loc[va_idx, "horizon_day"] = horizon_days
        oof.loc[va_idx, "h_bucket"] = [get_h_bucket(int(h)) for h in horizon_days]

    return OOFResult(oof=oof, best_iters=best_iters, used_heads=used_heads)


def fit_simplex_weights(y: np.ndarray, preds: np.ndarray, lam: float = 0.01) -> np.ndarray:
    n_models = preds.shape[1]

    def loss(w: np.ndarray) -> float:
        yhat = preds @ w
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return rmse + lam * float(np.sum(w**2))

    init = np.ones(n_models) / max(n_models, 1)
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
        return w.astype(float)
    w = np.clip(res.x, 0.0, None)
    if w.sum() <= 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return w


def fit_caruana_weights(
    y: np.ndarray,
    preds: np.ndarray,
    bag_runs: int = 15,
    max_iter: int = 60,
    subsample: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    n, m = preds.shape
    if n == 0 or m == 0:
        return np.array([])
    if m == 1:
        return np.array([1.0])

    rng = np.random.default_rng(seed)
    counts = np.zeros(m, dtype=float)

    for run in range(bag_runs):
        take = max(30, int(n * subsample))
        idx = rng.choice(n, size=take, replace=False)
        p = preds[idx]
        t = y[idx]
        ens = np.zeros(take, dtype=float)
        run_counts = np.zeros(m, dtype=float)

        for it in range(max_iter):
            best_j = 0
            best_score = np.inf
            denom = it + 1.0
            for j in range(m):
                cand = (ens * it + p[:, j]) / denom
                rmse = float(np.sqrt(np.mean((t - cand) ** 2)))
                if rmse < best_score:
                    best_score = rmse
                    best_j = j
            ens = (ens * it + p[:, best_j]) / denom
            run_counts[best_j] += 1.0

        if run_counts.sum() > 0:
            run_counts /= run_counts.sum()
        counts += run_counts

    if counts.sum() <= 1e-12:
        return fit_simplex_weights(y, preds)
    return counts / counts.sum()


def fit_ridge_meta_weights(
    y: np.ndarray,
    preds: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    if preds.shape[1] == 1:
        return np.array([1.0], dtype=float)
    w_nnls, _ = nnls(preds, y)
    ridge = Ridge(alpha=alpha, positive=True, fit_intercept=False)
    ridge.fit(preds, y)
    w = 0.5 * w_nnls + 0.5 * ridge.coef_
    w = np.clip(w, 0.0, None)
    if w.sum() <= 1e-12:
        return np.ones(preds.shape[1]) / preds.shape[1]
    return w / w.sum()


def fit_blend_weights(
    oof: pd.DataFrame,
    head_cols: List[str],
    lam: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    usable = oof.dropna(subset=["actual"] + head_cols + ["h_bucket"]).copy()
    if usable.empty:
        n = len(head_cols)
        return np.ones(n) / max(n, 1), {}

    y = usable["actual"].to_numpy(dtype=float)
    p = usable[head_cols].to_numpy(dtype=float)

    w_simplex = fit_simplex_weights(y, p, lam=lam)
    w_caruana = fit_caruana_weights(y, p, seed=seed)
    w_ridge = fit_ridge_meta_weights(y, p)

    def _rmse(w: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - p @ w) ** 2)))

    candidates = [w_simplex, w_caruana, w_ridge]
    scores = [_rmse(w) for w in candidates]
    global_w = candidates[int(np.argmin(scores))]

    bucket_w: Dict[str, np.ndarray] = {}
    for bucket_name, _, _ in HORIZON_BUCKETS:
        sub = usable[usable["h_bucket"] == bucket_name]
        if len(sub) < 30:
            bucket_w[bucket_name] = global_w.copy()
            continue
        yb = sub["actual"].to_numpy(dtype=float)
        pb = sub[head_cols].to_numpy(dtype=float)
        wb = fit_ridge_meta_weights(yb, pb, alpha=1.0)
        # Light prior to global blend.
        n = float(len(sub))
        prior = 30.0
        wb = (n * wb + prior * global_w) / (n + prior)
        wb = np.clip(wb, 0.0, None)
        wb = wb / wb.sum() if wb.sum() > 0 else global_w.copy()
        bucket_w[bucket_name] = wb

    return global_w, bucket_w


def safe_ensemble_row(
    row: pd.Series,
    head_cols: List[str],
    weights: np.ndarray,
    fallback_col: str = "pred_l2",
) -> float:
    vals = []
    wts = []
    for i, c in enumerate(head_cols):
        v = row.get(c, np.nan)
        if pd.notna(v) and np.isfinite(v):
            vals.append(float(v))
            wts.append(float(weights[i]))
    if len(vals) == 0:
        fb = row.get(fallback_col, np.nan)
        if pd.notna(fb) and np.isfinite(fb):
            return float(max(fb, 0.0))
        fb = row.get("pred_l1", np.nan)
        return float(max(float(fb), 0.0)) if pd.notna(fb) else 0.0
    w = np.array(wts, dtype=float)
    if w.sum() <= 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    pred = float(np.dot(np.array(vals, dtype=float), w))
    return max(pred, 0.0)


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


def fit_linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if len(x) < 5:
        return 1.0, 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.var(x))
    if var_x <= 1e-12:
        return 1.0, y_mean - x_mean
    cov = float(np.mean((x - x_mean) * (y - y_mean)))
    a = cov / var_x
    b = y_mean - a * x_mean
    return float(a), float(b)


def fit_quantile_map(y_true: np.ndarray, y_pred: np.ndarray, n_q: int = 99) -> Tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0.01, 0.99, n_q)
    pq = np.quantile(y_pred, q)
    yq = np.quantile(y_true, q)
    delta = yq - pq
    return pq.astype(float), delta.astype(float)


def apply_quantile_map(pred: np.ndarray, pq: np.ndarray, delta: np.ndarray, strength: float = 0.55) -> np.ndarray:
    interp = np.interp(pred, pq, delta, left=delta[0], right=delta[-1])
    return pred + strength * interp


def fit_residual_sarima(oof: pd.DataFrame, pred_col: str = "pred_final_base") -> Optional[SARIMAX]:
    residual = oof.dropna(subset=[pred_col]).copy()
    if residual.empty:
        return None
    residual["resid"] = residual["actual"] - residual[pred_col]
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


def train_final_models(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    best_iters: Dict[str, List[int]],
    l1_params: Dict[str, float],
    l2_params: Optional[Dict[str, float]],
    tweedie_params: Optional[Dict[str, float]],
    quantile_params: Optional[Dict[str, float]],
    use_prophet: bool,
    use_catboost: bool,
    seed: int,
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    x = train_df[feature_cols]
    y = train_df[target_col].to_numpy(dtype=float)

    def _num_round(head: str, default: int = 800) -> int:
        arr = best_iters.get(head, [])
        if not arr:
            return default
        return int(max(300, np.median(arr) * 1.1))

    d_l1 = lgb.Dataset(x, label=np.log1p(np.clip(y, 0.0, None)))
    models["pred_l1"] = lgb.train(l1_params, d_l1, num_boost_round=_num_round("pred_l1"), callbacks=[lgb.log_evaluation(0)])

    if l2_params is not None:
        d_l2 = lgb.Dataset(x, label=y)
        models["pred_l2"] = lgb.train(
            l2_params,
            d_l2,
            num_boost_round=_num_round("pred_l2"),
            callbacks=[lgb.log_evaluation(0)],
        )

    if tweedie_params is not None:
        d_tw = lgb.Dataset(x, label=y)
        models["pred_tweedie"] = lgb.train(
            tweedie_params,
            d_tw,
            num_boost_round=_num_round("pred_tweedie"),
            callbacks=[lgb.log_evaluation(0)],
        )

    if quantile_params is not None:
        d_q = lgb.Dataset(x, label=np.log1p(np.clip(y, 0.0, None)))
        models["pred_quantile"] = lgb.train(
            quantile_params,
            d_q,
            num_boost_round=_num_round("pred_quantile"),
            callbacks=[lgb.log_evaluation(0)],
        )

    if use_catboost and _HAS_CATBOOST:
        model, _ = _fit_catboost_and_predict(x, y, x.iloc[-min(len(x), 30) :], y[-min(len(y), 30) :], seed=seed)
        if model is not None:
            models["pred_catboost"] = model

    if use_prophet:
        try:
            models["pred_prophet"] = train_prophet(train_df[DATE_COL], y, seed=seed)
        except Exception:
            models["pred_prophet"] = None
    else:
        models["pred_prophet"] = None
    return models


def recursive_forecast_multihead(
    history_series: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    deterministic_features: pd.DataFrame,
    feature_cols: List[str],
    feature_medians: pd.Series,
    models: Dict[str, object],
    head_cols: List[str],
    global_weights: np.ndarray,
    bucket_weights: Dict[str, np.ndarray],
    event_multipliers: Dict[str, float],
    residual_fcst: Optional[pd.Series],
) -> Tuple[pd.Series, pd.DataFrame]:
    history = history_series.copy()
    out_preds: List[float] = []
    components: List[Dict[str, float]] = []

    prophet_model = models.get("pred_prophet")
    if prophet_model is not None:
        future_prophet = prophet_model.predict(pd.DataFrame({"ds": forecast_dates.values}))
        prophet_series = pd.Series(np.clip(future_prophet["yhat"].to_numpy(dtype=float), 0.0, None), index=forecast_dates)
    else:
        prophet_series = pd.Series(
            seasonal_naive_predict(history, forecast_dates, seasonal_lag=365),
            index=forecast_dates,
        )

    for step, dt in enumerate(forecast_dates, start=1):
        lag_feats = lag_roll_row_from_history(history)
        row = deterministic_features.loc[dt].to_dict()
        row.update(lag_feats)
        x_row = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}]).fillna(feature_medians)

        pred_map: Dict[str, float] = {}
        if "pred_l1" in head_cols and "pred_l1" in models:
            m = models["pred_l1"]
            pred_map["pred_l1"] = float(np.expm1(m.predict(x_row, num_iteration=m.best_iteration)[0]))
        if "pred_l2" in head_cols and "pred_l2" in models:
            m = models["pred_l2"]
            pred_map["pred_l2"] = float(m.predict(x_row, num_iteration=m.best_iteration)[0])
        if "pred_tweedie" in head_cols and "pred_tweedie" in models:
            m = models["pred_tweedie"]
            pred_map["pred_tweedie"] = float(m.predict(x_row, num_iteration=m.best_iteration)[0])
        if "pred_quantile" in head_cols and "pred_quantile" in models:
            m = models["pred_quantile"]
            pred_map["pred_quantile"] = float(np.expm1(m.predict(x_row, num_iteration=m.best_iteration)[0]))
        if "pred_catboost" in head_cols and "pred_catboost" in models:
            m = models["pred_catboost"]
            pred_map["pred_catboost"] = float(np.expm1(m.predict(x_row)[0]))
        if "pred_prophet" in head_cols:
            pred_map["pred_prophet"] = float(prophet_series.loc[dt])

        h_bucket = get_h_bucket(step)
        w = bucket_weights.get(h_bucket, global_weights)
        row_for_blend = pd.Series(pred_map)
        for c in head_cols:
            if c not in row_for_blend:
                row_for_blend[c] = np.nan
        pred = safe_ensemble_row(row_for_blend, head_cols, w)
        pred = apply_event_multiplier(pred, deterministic_features.loc[dt], event_multipliers)
        if residual_fcst is not None and dt in residual_fcst.index:
            pred += float(residual_fcst.loc[dt])
        pred = max(float(pred), 0.0)

        history.loc[dt] = pred
        out_preds.append(pred)
        comp = {"date": dt, "h_bucket": h_bucket, "pred_blend_raw": pred}
        comp.update(pred_map)
        components.append(comp)

    forecast = pd.Series(out_preds, index=forecast_dates)
    comp_df = pd.DataFrame(components)
    return forecast, comp_df
