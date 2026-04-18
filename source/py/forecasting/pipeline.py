"""Orchestrates end-to-end training and forecasting for both targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .constants import DATE_COL, LAGS, TARGETS
from .cv import ExpandingWindowWalkForward
from .data_io import read_inputs, seasonal_naive_predict
from .features import build_deterministic_feature_frame, make_lag_roll_features
from .metrics import evaluate
from .models import (
    apply_event_multiplier,
    build_l1_params,
    build_tweedie_params,
    compute_event_multipliers,
    fit_blend_weights,
    fit_residual_sarima,
    generate_oof_predictions,
    recursive_forecast_lgb,
    train_prophet,
    tune_lgbm_l1,
)
from .types import ForecastArtifacts, PipelineConfig


def run_target_pipeline(
    sales: pd.DataFrame,
    deterministic_features: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    target_col: str,
    n_trials: int,
    max_splits: int,
    seed: int,
    disable_optuna: bool,
    disable_prophet: bool,
    optuna_timeout: int,
) -> Tuple[pd.Series, ForecastArtifacts]:
    frame = sales[[DATE_COL, target_col]].copy().sort_values(DATE_COL).reset_index(drop=True)
    target_series = pd.Series(
        frame[target_col].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame[DATE_COL]),
        name=target_col,
    )

    lag_feats = make_lag_roll_features(target_series, horizon=1)
    train_df = deterministic_features.loc[target_series.index].copy().join(lag_feats)
    train_df[DATE_COL] = train_df.index
    train_df[target_col] = target_series.values

    min_needed = [f"lag_{max(LAGS)}", "rmean_90", "yoy_lag", "yoy_roll28"]
    train_df = train_df.dropna(subset=min_needed).reset_index(drop=True)
    feature_cols = [c for c in train_df.columns if c not in (DATE_COL, target_col)]
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    feature_medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(feature_medians)

    splitter = ExpandingWindowWalkForward(
        initial_train_days=365 * 5,
        val_days=180,
        step_days=180,
        gap_days=28,
        max_splits=max_splits,
    )
    if splitter.get_n_splits(train_df) <= 0:
        raise RuntimeError("Not enough rows to create walk-forward folds.")

    trials = 0 if disable_optuna else n_trials
    best_params = tune_lgbm_l1(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        splitter=splitter,
        n_trials=trials,
        seed=seed,
        timeout_s=optuna_timeout,
    )
    l1_params = build_l1_params(best_params, seed=seed)
    tw_params = build_tweedie_params(best_params, seed=seed)

    oof, fold_iters_l1, fold_iters_tw = generate_oof_predictions(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        splitter=splitter,
        l1_params=l1_params,
        tweedie_params=tw_params,
        seed=seed,
        use_prophet=not disable_prophet,
    )
    oof = oof.rename(columns={target_col: "actual"})

    blend_weights, blend_objective = fit_blend_weights(oof, lam=0.01)
    oof["pred_blend"] = (
        blend_weights[0] * oof["pred_l1"]
        + blend_weights[1] * oof["pred_tweedie"]
        + blend_weights[2] * oof["pred_prophet"]
    )

    event_multipliers = compute_event_multipliers(oof)
    oof["pred_blend_event"] = oof.apply(
        lambda row: apply_event_multiplier(float(row["pred_blend"]), row, event_multipliers)
        if pd.notna(row["pred_blend"])
        else np.nan,
        axis=1,
    )

    residual_model = fit_residual_sarima(oof)

    n_rounds_l1 = int(max(300, np.median(fold_iters_l1) * 1.1))
    n_rounds_tw = int(max(300, np.median(fold_iters_tw) * 1.1))
    dtrain_l1 = lgb.Dataset(train_df[feature_cols], label=np.log1p(np.clip(train_df[target_col], 0, None)))
    dtrain_tw = lgb.Dataset(train_df[feature_cols], label=train_df[target_col])
    model_l1 = lgb.train(l1_params, dtrain_l1, num_boost_round=n_rounds_l1, callbacks=[lgb.log_evaluation(0)])
    model_tw = lgb.train(tw_params, dtrain_tw, num_boost_round=n_rounds_tw, callbacks=[lgb.log_evaluation(0)])

    prophet_model = None
    if not disable_prophet:
        try:
            prophet_model = train_prophet(train_df[DATE_COL], train_df[target_col].to_numpy(), seed=seed)
        except Exception:
            prophet_model = None
    if prophet_model is not None:
        future_prophet = prophet_model.predict(pd.DataFrame({"ds": forecast_dates.values}))
        prophet_preds = pd.Series(np.clip(future_prophet["yhat"].to_numpy(), 0.0, None), index=forecast_dates)
    else:
        prophet_preds = pd.Series(
            seasonal_naive_predict(target_series, forecast_dates, seasonal_lag=365),
            index=forecast_dates,
        )

    residual_fcst = None
    if residual_model is not None:
        try:
            residual_fcst = pd.Series(
                residual_model.get_forecast(steps=len(forecast_dates)).predicted_mean.to_numpy(dtype=float),
                index=forecast_dates,
            )
        except Exception:
            residual_fcst = None

    forecast_series = recursive_forecast_lgb(
        history_series=target_series,
        forecast_dates=forecast_dates,
        deterministic_features=deterministic_features,
        feature_cols=feature_cols,
        feature_medians=feature_medians,
        model_l1=model_l1,
        model_tw=model_tw,
        prophet_preds=prophet_preds,
        blend_weights=blend_weights,
        event_multipliers=event_multipliers,
        residual_fcst=residual_fcst,
    )

    eval_rows = oof.dropna(subset=["pred_l1", "pred_tweedie", "pred_prophet", "pred_blend", "pred_blend_event"])
    metrics = {
        "lgb_l1": evaluate(eval_rows["actual"].to_numpy(), eval_rows["pred_l1"].to_numpy()),
        "lgb_tweedie": evaluate(eval_rows["actual"].to_numpy(), eval_rows["pred_tweedie"].to_numpy()),
        "prophet_or_sn": evaluate(eval_rows["actual"].to_numpy(), eval_rows["pred_prophet"].to_numpy()),
        "blend": evaluate(eval_rows["actual"].to_numpy(), eval_rows["pred_blend"].to_numpy()),
        "blend_event": evaluate(eval_rows["actual"].to_numpy(), eval_rows["pred_blend_event"].to_numpy()),
        "blend_objective": {"value": blend_objective},
    }

    artifacts = ForecastArtifacts(
        target=target_col,
        oof_frame=oof,
        metrics=metrics,
        blend_weights={
            "lgb_l1": float(blend_weights[0]),
            "lgb_tweedie": float(blend_weights[1]),
            "prophet_or_sn": float(blend_weights[2]),
        },
        event_multipliers=event_multipliers,
        best_params=best_params,
    )
    return forecast_series, artifacts


def run_full_pipeline(config: PipelineConfig) -> Path:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    sales, horizon_dates = read_inputs(config.data_dir)
    start_date = pd.Timestamp(sales[DATE_COL].min())
    end_date = pd.Timestamp(horizon_dates.max())
    deterministic_features = build_deterministic_feature_frame(start_date, end_date)

    results = {}
    forecasts = {}
    for i, target in enumerate(TARGETS):
        fcst, artifacts = run_target_pipeline(
            sales=sales,
            deterministic_features=deterministic_features,
            forecast_dates=horizon_dates,
            target_col=target,
            n_trials=config.n_trials,
            max_splits=config.max_splits,
            seed=config.seed + i,
            disable_optuna=config.disable_optuna,
            disable_prophet=config.disable_prophet,
            optuna_timeout=config.optuna_timeout,
        )
        forecasts[target] = fcst
        results[target] = artifacts

    submission = pd.DataFrame(
        {
            "Date": horizon_dates.strftime("%Y-%m-%d"),
            "Revenue": np.round(forecasts["revenue"].to_numpy(), 2),
            "COGS": np.round(forecasts["cogs"].to_numpy(), 2),
        }
    )
    submission_path = config.out_dir / "submission_modeling.csv"
    submission.to_csv(submission_path, index=False)
    results["revenue"].oof_frame.to_csv(config.out_dir / "oof_revenue.csv", index=False)
    results["cogs"].oof_frame.to_csv(config.out_dir / "oof_cogs.csv", index=False)

    summary = {
        "config": {
            "data_dir": str(config.data_dir),
            "out_dir": str(config.out_dir),
            "n_trials": config.n_trials,
            "max_splits": config.max_splits,
            "disable_prophet": config.disable_prophet,
            "disable_optuna": config.disable_optuna,
            "optuna_timeout": config.optuna_timeout,
        },
        "revenue": {
            "blend_weights": results["revenue"].blend_weights,
            "event_multipliers": results["revenue"].event_multipliers,
            "metrics": results["revenue"].metrics,
            "best_params": results["revenue"].best_params,
        },
        "cogs": {
            "blend_weights": results["cogs"].blend_weights,
            "event_multipliers": results["cogs"].event_multipliers,
            "metrics": results["cogs"].metrics,
            "best_params": results["cogs"].best_params,
        },
        "outputs": {
            "submission": str(submission_path),
            "oof_revenue": str(config.out_dir / "oof_revenue.csv"),
            "oof_cogs": str(config.out_dir / "oof_cogs.csv"),
        },
    }
    summary_path = config.out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_path

