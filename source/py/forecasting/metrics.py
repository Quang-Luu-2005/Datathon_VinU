"""Metrics and utility helpers."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_full(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = float(np.mean(np.abs(np.diff(insample)))) if len(insample) > 1 else np.nan
    mase = mae / denom if np.isfinite(denom) and denom > 1e-12 else np.nan
    smape = float(
        200.0
        * np.mean(
            np.abs(y_pred - y_true)
            / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
        )
    )
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE": smape,
        "MASE": float(mase) if np.isfinite(mase) else np.nan,
    }


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    block_days: int = 30,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"lo": np.nan, "hi": np.nan}
    if n <= block_days:
        score = metric_fn(y_true, y_pred)
        return {"lo": float(score), "hi": float(score)}

    rng = np.random.default_rng(seed)
    vals = []
    n_blocks = int(np.ceil(n / block_days))
    starts = np.arange(0, n - block_days + 1)
    for _ in range(n_boot):
        idx_parts = []
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        for s in chosen:
            idx_parts.append(np.arange(s, min(s + block_days, n)))
        idx = np.concatenate(idx_parts)[:n]
        vals.append(float(metric_fn(y_true[idx], y_pred[idx])))
    q_lo, q_hi = np.quantile(vals, [0.025, 0.975])
    return {"lo": float(q_lo), "hi": float(q_hi)}
