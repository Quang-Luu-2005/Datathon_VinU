"""Orchestrates end-to-end training and forecasting for targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .constants import DATE_COL, HORIZON_BUCKETS, LAGS, TARGETS
from .cv import HybridWalkForwardCV
from .data_io import read_inputs
from .features import (
    add_changepoint_features,
    build_deterministic_feature_frame,
    make_lag_roll_features,
)
from .metrics import block_bootstrap_ci, evaluate, evaluate_full
from .models import (
    OOFResult,
    apply_event_multiplier,
    apply_quantile_map,
    build_l1_params,
    build_l2_params,
    build_quantile_params,
    build_tweedie_params,
    compute_event_multipliers,
    fit_blend_weights,
    fit_linear_calibration,
    fit_quantile_map,
    fit_residual_sarima,
    generate_oof_predictions,
    recursive_forecast_multihead,
    safe_ensemble_row,
    train_final_models,
    tune_lgbm_l1,
)
from .types import ForecastArtifacts, PipelineConfig


def _map_blend_weights(head_cols: list[str], weights: np.ndarray) -> Dict[str, float]:
    return {head_cols[i]: float(weights[i]) for i in range(len(head_cols))}


def _pick_available_heads(oof: pd.DataFrame, candidate_heads: list[str]) -> list[str]:
    out = []
    for c in candidate_heads:
        if c in oof.columns and oof[c].notna().any():
            out.append(c)
    return out


def _apply_oof_blend(
    oof: pd.DataFrame,
    head_cols: list[str],
    global_weights: np.ndarray,
    bucket_weights: Dict[str, np.ndarray],
) -> pd.Series:
    preds = []
    for _, row in oof.iterrows():
        bucket = str(row.get("h_bucket", ""))
        w = bucket_weights.get(bucket, global_weights)
        preds.append(safe_ensemble_row(row, head_cols, w))
    return pd.Series(preds, index=oof.index, dtype=float)


def _fit_calibration_stack(
    oof_short: pd.DataFrame,
    target_col: str,
) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], float, float]:
    cal_rows = oof_short.dropna(subset=["actual", "pred_blend_event"]).copy()
    if cal_rows.empty:
        return 1.0, 0.0, None, None, 0.0, np.inf

    y = cal_rows["actual"].to_numpy(dtype=float)
    p = cal_rows["pred_blend_event"].to_numpy(dtype=float)
    a, b = fit_linear_calibration(y, p)
    p_cal = np.clip(a * p + b, 0.0, None)
    pq, dq = fit_quantile_map(y, p_cal)
    clip_lo = float(np.quantile(y, 0.01))
    clip_hi = float(np.quantile(y, 0.99))

    if target_col == "cogs_ratio":
        clip_lo = max(0.01, clip_lo)
        clip_hi = min(3.0, max(clip_lo + 1e-3, clip_hi))

    return a, b, pq, dq, clip_lo, clip_hi


def run_target_pipeline(
    sales: pd.DataFrame,
    deterministic_features: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    target_col: str,
    config: PipelineConfig,
    seed: int,
) -> Tuple[pd.Series, ForecastArtifacts, pd.DataFrame]:
    frame = sales[[DATE_COL, target_col]].copy().sort_values(DATE_COL).reset_index(drop=True)
    target_series = pd.Series(
        frame[target_col].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame[DATE_COL]),
        name=target_col,
    )

    det_target = deterministic_features.loc[target_series.index].copy()
    det_target = add_changepoint_features(det_target, target_series, n_bkps=3)
    lag_feats = make_lag_roll_features(target_series, horizon=1)
    train_df = det_target.join(lag_feats)
    train_df[DATE_COL] = train_df.index
    train_df[target_col] = target_series.values

    min_needed = [f"lag_{max(LAGS)}", "rmean_90", "yoy_lag", "yoy_roll28"]
    train_df = train_df.dropna(subset=min_needed).reset_index(drop=True)
    feature_cols = [c for c in train_df.columns if c not in (DATE_COL, target_col)]
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    feature_medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(feature_medians)

    cv = HybridWalkForwardCV(
        short_initial_train_days=config.short_cv_initial_train_days,
        short_val_days=config.short_cv_val_days,
        short_step_days=config.short_cv_step_days,
        short_gap_days=config.short_cv_gap_days,
        short_max_splits=config.short_cv_max_splits,
        realistic_probe_enabled=config.realistic_probe_enabled,
        realistic_probe_initial_train_days=config.realistic_probe_initial_train_days,
        realistic_probe_val_days=config.realistic_probe_val_days,
        realistic_probe_gap_days=config.realistic_probe_gap_days,
    )
    short_folds = cv.split_short(train_df)
    probe_folds = cv.split_realistic_probe(train_df)
    all_folds = short_folds + probe_folds
    if len(short_folds) <= 0:
        raise RuntimeError("Not enough rows to create short_cv folds.")

    trials = 0 if config.disable_optuna else config.n_trials
    best_params = tune_lgbm_l1(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        short_folds=short_folds,
        n_trials=trials,
        seed=seed,
        timeout_s=config.optuna_timeout,
    )
    l1_params = build_l1_params(best_params, seed=seed)
    l2_params = build_l2_params(best_params, seed=seed) if config.use_l2_head else None
    tw_params = build_tweedie_params(best_params, seed=seed) if config.use_tweedie_head else None
    q_params = build_quantile_params(best_params, seed=seed) if config.use_quantile_head else None

    oof_result: OOFResult = generate_oof_predictions(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        folds=all_folds,
        l1_params=l1_params,
        l2_params=l2_params,
        tweedie_params=tw_params,
        quantile_params=q_params,
        seed=seed,
        use_prophet=not config.disable_prophet,
        use_catboost=config.use_catboost_head,
    )
    oof = oof_result.oof.rename(columns={target_col: "actual"})
    head_cols = _pick_available_heads(oof, oof_result.used_heads)

    oof_short = oof[oof["fold_type"] == "short_cv"].copy()

    global_w, bucket_w = fit_blend_weights(oof_short, head_cols=head_cols, lam=0.01, seed=seed)
    oof["pred_blend"] = _apply_oof_blend(oof, head_cols, global_w, bucket_w)

    oof_short = oof[oof["fold_type"] == "short_cv"].copy()
    oof_probe = oof[oof["fold_type"] == "realistic_probe"].copy()
    event_multipliers = compute_event_multipliers(oof_short)
    oof["pred_blend_event"] = oof.apply(
        lambda row: apply_event_multiplier(float(row["pred_blend"]), row, event_multipliers)
        if pd.notna(row["pred_blend"])
        else np.nan,
        axis=1,
    )

    oof_short = oof[oof["fold_type"] == "short_cv"].copy()
    oof_probe = oof[oof["fold_type"] == "realistic_probe"].copy()
    cal_a, cal_b, qm_pq, qm_delta, clip_lo, clip_hi = _fit_calibration_stack(oof_short, target_col=target_col)
    p = np.clip(cal_a * oof["pred_blend_event"].to_numpy(dtype=float) + cal_b, 0.0, None)
    if qm_pq is not None and qm_delta is not None:
        p = apply_quantile_map(p, qm_pq, qm_delta, strength=0.55)
    p = np.clip(p, clip_lo, clip_hi)
    if target_col == "cogs_ratio":
        p = np.clip(p, 0.01, 3.0)
    oof["pred_final_base"] = p

    oof_short = oof[oof["fold_type"] == "short_cv"].copy()
    oof_probe = oof[oof["fold_type"] == "realistic_probe"].copy()
    residual_model = fit_residual_sarima(oof_short, pred_col="pred_final_base")
    residual_fcst = None
    if residual_model is not None:
        try:
            residual_fcst = pd.Series(
                residual_model.get_forecast(steps=len(forecast_dates)).predicted_mean.to_numpy(dtype=float),
                index=forecast_dates,
            )
        except Exception:
            residual_fcst = None

    final_models = train_final_models(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        best_iters=oof_result.best_iters,
        l1_params=l1_params,
        l2_params=l2_params,
        tweedie_params=tw_params,
        quantile_params=q_params,
        use_prophet=not config.disable_prophet,
        use_catboost=config.use_catboost_head,
        seed=seed,
    )

    target_history = pd.Series(
        train_df[target_col].to_numpy(dtype=float),
        index=pd.DatetimeIndex(train_df[DATE_COL]),
    )
    forecast_series, component_frame = recursive_forecast_multihead(
        history_series=target_history,
        forecast_dates=forecast_dates,
        deterministic_features=deterministic_features,
        feature_cols=feature_cols,
        feature_medians=feature_medians,
        models=final_models,
        head_cols=head_cols,
        global_weights=global_w,
        bucket_weights=bucket_w,
        event_multipliers=event_multipliers,
        residual_fcst=residual_fcst,
    )
    forecast_final = np.clip(cal_a * forecast_series.to_numpy(dtype=float) + cal_b, 0.0, None)
    if qm_pq is not None and qm_delta is not None:
        forecast_final = apply_quantile_map(forecast_final, qm_pq, qm_delta, strength=0.55)
    forecast_final = np.clip(forecast_final, clip_lo, clip_hi)
    if target_col == "cogs_ratio":
        forecast_final = np.clip(forecast_final, 0.01, 3.0)
    forecast_series = pd.Series(forecast_final, index=forecast_dates)

    insample = target_history.to_numpy(dtype=float)
    eval_rows = oof_short.dropna(subset=["actual", "pred_blend", "pred_blend_event", "pred_final_base"]).copy()
    metrics: Dict[str, Dict[str, float]] = {}
    metric_map = [
        ("l1", "pred_l1"),
        ("l2_raw", "pred_l2"),
        ("tweedie", "pred_tweedie"),
        ("quantile", "pred_quantile"),
        ("catboost", "pred_catboost"),
        ("prophet_or_sn", "pred_prophet"),
        ("blend", "pred_blend"),
        ("blend_event", "pred_blend_event"),
        ("final_calibrated", "pred_final_base"),
    ]
    for name, col in metric_map:
        if col in eval_rows.columns and eval_rows[col].notna().any():
            metrics[name] = evaluate_full(
                eval_rows["actual"].to_numpy(dtype=float),
                eval_rows[col].to_numpy(dtype=float),
                insample,
            )

    bucket_metrics = {}
    for bucket_name, _, _ in HORIZON_BUCKETS:
        sub = eval_rows[eval_rows["h_bucket"] == bucket_name]
        if len(sub) < 20:
            continue
        bucket_metrics[bucket_name] = evaluate_full(
            sub["actual"].to_numpy(dtype=float),
            sub["pred_final_base"].to_numpy(dtype=float),
            insample,
        )
    metrics["bucket_final"] = bucket_metrics

    if "final_calibrated" in metrics and len(eval_rows) > 0:
        yt = eval_rows["actual"].to_numpy(dtype=float)
        yp = eval_rows["pred_final_base"].to_numpy(dtype=float)
        metrics["final_calibrated_ci"] = {
            "MAE": block_bootstrap_ci(
                yt,
                yp,
                metric_fn=lambda a, b: float(np.mean(np.abs(a - b))),
                block_days=config.block_bootstrap_block_days,
                n_boot=config.block_bootstrap_n,
                seed=seed,
            ),
            "RMSE": block_bootstrap_ci(
                yt,
                yp,
                metric_fn=lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))),
                block_days=config.block_bootstrap_block_days,
                n_boot=config.block_bootstrap_n,
                seed=seed + 7,
            ),
            "R2": block_bootstrap_ci(
                yt,
                yp,
                metric_fn=lambda a, b: float(1.0 - np.sum((a - b) ** 2) / (np.sum((a - np.mean(a)) ** 2) + 1e-12)),
                block_days=config.block_bootstrap_block_days,
                n_boot=config.block_bootstrap_n,
                seed=seed + 17,
            ),
        }

    realistic_probe_metrics: Dict[str, float] = {}
    probe_rows = oof_probe.dropna(subset=["actual", "pred_final_base"])
    if not probe_rows.empty:
        realistic_probe_metrics = evaluate(
            probe_rows["actual"].to_numpy(dtype=float),
            probe_rows["pred_final_base"].to_numpy(dtype=float),
        )

    artifacts = ForecastArtifacts(
        target=target_col,
        oof_frame=oof,
        metrics=metrics,
        blend_weights=_map_blend_weights(head_cols, global_w),
        blend_weights_bucket={
            b: _map_blend_weights(head_cols, w) for b, w in bucket_w.items()
        },
        event_multipliers=event_multipliers,
        best_params=best_params,
        realistic_probe_metrics=realistic_probe_metrics,
        used_heads=head_cols,
    )
    return forecast_series, artifacts, component_frame


def _optimize_cogs_alpha(result_map: Dict[str, ForecastArtifacts]) -> Tuple[float, Dict[str, float]]:
    cogs_oof = result_map["cogs"].oof_frame[[DATE_COL, "actual", "pred_final_base"]].rename(
        columns={"actual": "actual_cogs", "pred_final_base": "pred_cogs_direct"}
    )
    rev_oof = result_map["revenue"].oof_frame[[DATE_COL, "pred_final_base"]].rename(
        columns={"pred_final_base": "pred_revenue"}
    )
    ratio_oof = result_map["cogs_ratio"].oof_frame[[DATE_COL, "pred_final_base"]].rename(
        columns={"pred_final_base": "pred_ratio"}
    )
    merged = cogs_oof.merge(rev_oof, on=DATE_COL, how="inner").merge(ratio_oof, on=DATE_COL, how="inner")
    merged = merged.dropna(subset=["actual_cogs", "pred_cogs_direct", "pred_revenue", "pred_ratio"])
    if merged.empty:
        return 1.0, {"RMSE": np.nan, "MAE": np.nan}

    merged["pred_cogs_from_ratio"] = np.clip(merged["pred_revenue"] * merged["pred_ratio"], 0.0, None)
    y = merged["actual_cogs"].to_numpy(dtype=float)
    p_dir = merged["pred_cogs_direct"].to_numpy(dtype=float)
    p_rat = merged["pred_cogs_from_ratio"].to_numpy(dtype=float)
    best_alpha = 1.0
    best_rmse = float("inf")
    best_mae = float("inf")
    for alpha in np.arange(0.0, 1.0001, 0.01):
        pred = alpha * p_dir + (1.0 - alpha) * p_rat
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = float(np.mean(np.abs(y - pred)))
            best_alpha = float(alpha)
    return best_alpha, {"RMSE": best_rmse, "MAE": best_mae}


def run_full_pipeline(config: PipelineConfig) -> Path:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    sales, horizon_dates = read_inputs(config.data_dir)
    if "cogs_ratio" not in sales.columns:
        sales["cogs_ratio"] = (
            pd.to_numeric(sales["cogs"], errors="coerce").fillna(0.0)
            / pd.to_numeric(sales["revenue"], errors="coerce").fillna(0.0).clip(lower=1.0)
        ).clip(0.01, 3.0)

    start_date = pd.Timestamp(sales[DATE_COL].min())
    end_date = pd.Timestamp(horizon_dates.max())
    deterministic_features = build_deterministic_feature_frame(start_date, end_date)

    results: Dict[str, ForecastArtifacts] = {}
    forecasts: Dict[str, pd.Series] = {}
    component_frames: Dict[str, pd.DataFrame] = {}
    for i, target in enumerate(TARGETS):
        fcst, artifacts, comp = run_target_pipeline(
            sales=sales,
            deterministic_features=deterministic_features,
            forecast_dates=horizon_dates,
            target_col=target,
            config=config,
            seed=config.seed + i,
        )
        forecasts[target] = fcst
        results[target] = artifacts
        component_frames[target] = comp

    alpha, alpha_stats = _optimize_cogs_alpha(results)
    cogs_from_ratio = np.clip(
        forecasts["revenue"].to_numpy(dtype=float) * forecasts["cogs_ratio"].to_numpy(dtype=float),
        0.0,
        None,
    )
    cogs_final = np.clip(
        alpha * forecasts["cogs"].to_numpy(dtype=float) + (1.0 - alpha) * cogs_from_ratio,
        0.0,
        None,
    )

    submission = pd.DataFrame(
        {
            "Date": horizon_dates.strftime("%Y-%m-%d"),
            "Revenue": np.round(forecasts["revenue"].to_numpy(dtype=float), 2),
            "COGS": np.round(cogs_final, 2),
        }
    )
    submission_path = config.out_dir / "submission_modeling.csv"
    submission.to_csv(submission_path, index=False)

    components = pd.DataFrame(
        {
            "Date": horizon_dates.strftime("%Y-%m-%d"),
            "Revenue_forecast": forecasts["revenue"].to_numpy(dtype=float),
            "COGS_direct_forecast": forecasts["cogs"].to_numpy(dtype=float),
            "COGS_ratio_forecast": forecasts["cogs_ratio"].to_numpy(dtype=float),
            "COGS_from_ratio_forecast": cogs_from_ratio,
            "COGS_final_blend": cogs_final,
        }
    )
    components_path = config.out_dir / "forecast_components.csv"
    components.to_csv(components_path, index=False)

    for target in TARGETS:
        results[target].oof_frame.to_csv(config.out_dir / f"oof_{target}.csv", index=False)

    core_metrics_rows = []
    for target in TARGETS:
        final_metrics = results[target].metrics.get("final_calibrated", {})
        core_metrics_rows.append(
            {
                "target": target,
                "MAE": float(final_metrics.get("MAE", np.nan)),
                "RMSE": float(final_metrics.get("RMSE", np.nan)),
                "R2": float(final_metrics.get("R2", np.nan)),
            }
        )
    core_metrics_df = pd.DataFrame(core_metrics_rows)
    core_metrics_csv = config.out_dir / "model_metrics_core.csv"
    core_metrics_json = config.out_dir / "model_metrics_core.json"
    core_metrics_df.to_csv(core_metrics_csv, index=False)
    with core_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(core_metrics_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "config": {
            "data_dir": str(config.data_dir),
            "out_dir": str(config.out_dir),
            "n_trials": config.n_trials,
            "seed": config.seed,
            "disable_prophet": config.disable_prophet,
            "disable_optuna": config.disable_optuna,
            "optuna_timeout": config.optuna_timeout,
            "use_l2_head": config.use_l2_head,
            "use_tweedie_head": config.use_tweedie_head,
            "use_quantile_head": config.use_quantile_head,
            "use_catboost_head": config.use_catboost_head,
            "short_cv": {
                "max_splits": config.short_cv_max_splits,
                "initial_train_days": config.short_cv_initial_train_days,
                "val_days": config.short_cv_val_days,
                "step_days": config.short_cv_step_days,
                "gap_days": config.short_cv_gap_days,
            },
            "realistic_probe": {
                "enabled": config.realistic_probe_enabled,
                "initial_train_days": config.realistic_probe_initial_train_days,
                "val_days": config.realistic_probe_val_days,
                "gap_days": config.realistic_probe_gap_days,
            },
        },
        "targets": {},
        "cogs_hybrid": {
            "alpha_direct": alpha,
            "alpha_ratio": (1.0 - alpha),
            "oof_stats": alpha_stats,
        },
        "outputs": {
            "submission": str(submission_path),
            "forecast_components": str(components_path),
            "core_metrics_csv": str(core_metrics_csv),
            "core_metrics_json": str(core_metrics_json),
        },
    }

    for target in TARGETS:
        summary["targets"][target] = {
            "best_params": results[target].best_params,
            "used_heads": results[target].used_heads,
            "blend_weights_global": results[target].blend_weights,
            "blend_weights_bucket": results[target].blend_weights_bucket,
            "event_multipliers": results[target].event_multipliers,
            "metrics": results[target].metrics,
            "realistic_probe_metrics": results[target].realistic_probe_metrics,
        }

    summary_path = config.out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_path
