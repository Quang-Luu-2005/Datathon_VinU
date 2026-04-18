"""Input/Output and baseline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .constants import DATE_COL


def read_inputs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    sales = pd.read_csv(data_dir / "sales.csv")
    sales.columns = [c.strip().lower() for c in sales.columns]
    sales[DATE_COL] = pd.to_datetime(sales[DATE_COL])
    sales = sales.sort_values(DATE_COL).reset_index(drop=True)

    sub = pd.read_csv(data_dir / "sample_submission.csv")
    date_col = "Date" if "Date" in sub.columns else "date"
    horizon_dates = pd.to_datetime(sub[date_col]).sort_values().unique()
    return sales, pd.DatetimeIndex(horizon_dates)


def seasonal_naive_predict(history: pd.Series, forecast_dates: pd.DatetimeIndex, seasonal_lag: int = 365) -> np.ndarray:
    hist = history.copy()
    preds: List[float] = []
    for dt in forecast_dates:
        ref = dt - pd.Timedelta(days=seasonal_lag)
        if ref in hist.index:
            pred = float(hist.loc[ref])
        else:
            pred = float(hist.iloc[-7:].mean())
        preds.append(max(pred, 0.0))
        hist.loc[dt] = pred
    return np.array(preds, dtype=float)

