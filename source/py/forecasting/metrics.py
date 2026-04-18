"""Metrics and utility helpers."""

from __future__ import annotations

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

