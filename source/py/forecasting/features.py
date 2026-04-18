"""Feature engineering for deterministic calendar/event and autoregressive features."""

from __future__ import annotations

from typing import Dict

import holidays
import numpy as np
import pandas as pd

from .constants import DATE_COL, EWM_ALPHAS, LAGS, ROLL_WINDOWS, TET_DATES


def add_basic_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out[DATE_COL].dt
    out["day_of_week"] = d.dayofweek
    out["day_of_month"] = d.day
    out["day_of_year"] = d.dayofyear
    out["week_of_year"] = d.isocalendar().week.astype(int)
    out["month"] = d.month
    out["quarter"] = d.quarter
    out["year"] = d.year
    out["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    out["is_month_start"] = d.is_month_start.astype("int8")
    out["is_month_end"] = d.is_month_end.astype("int8")
    out["is_quarter_start"] = d.is_quarter_start.astype("int8")
    out["is_quarter_end"] = d.is_quarter_end.astype("int8")
    out["is_year_start"] = d.is_year_start.astype("int8")
    out["is_year_end"] = d.is_year_end.astype("int8")
    out["time_idx"] = np.arange(len(out), dtype=np.int32)
    return out


def add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, period in [
        ("day_of_week", 7),
        ("month", 12),
        ("day_of_year", 365.25),
        ("day_of_month", 31),
    ]:
        out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out


def add_tet_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out[DATE_COL].values.astype("datetime64[D]")
    tet = TET_DATES.values.astype("datetime64[D]")
    nxt = np.clip(np.searchsorted(tet, d, side="left"), 0, len(tet) - 1)
    prv = np.clip(nxt - 1, 0, len(tet) - 1)
    days_to = (tet[nxt] - d).astype(int)
    days_since = (d - tet[prv]).astype(int)
    out["days_to_tet"] = days_to
    out["days_since_tet"] = days_since
    out["is_tet_day"] = (days_to == 0).astype("int8")
    out["is_tet_week"] = ((days_to <= 3) | (days_since <= 3)).astype("int8")
    out["is_pre_tet_14d"] = ((days_to >= 1) & (days_to <= 14)).astype("int8")
    out["is_pre_tet_30d"] = ((days_to >= 1) & (days_to <= 30)).astype("int8")
    out["tet_proximity"] = np.exp(-np.minimum(days_to, days_since) / 7.0)
    return out


def _black_friday(year: int) -> pd.Timestamp:
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    return nov[nov.dayofweek == 4][3]


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    years = sorted(out[DATE_COL].dt.year.unique().tolist())
    vn_holidays = holidays.country_holidays("VN", years=years)
    out["is_vn_holiday"] = out[DATE_COL].isin(vn_holidays).astype("int8")

    mega_days = {
        "dd_3_3": (3, 3),
        "dd_6_6": (6, 6),
        "dd_9_9": (9, 9),
        "dd_10_10": (10, 10),
        "dd_11_11": (11, 11),
        "dd_12_12": (12, 12),
        "womens_day": (3, 8),
    }
    for name, (month, day) in mega_days.items():
        out[f"is_{name}"] = (
            (out[DATE_COL].dt.month == month) & (out[DATE_COL].dt.day == day)
        ).astype("int8")
        this_year = pd.to_datetime(
            dict(year=out[DATE_COL].dt.year, month=month, day=day),
            errors="coerce",
        )
        next_year = pd.to_datetime(
            dict(year=out[DATE_COL].dt.year + 1, month=month, day=day),
            errors="coerce",
        )
        dt = (this_year - out[DATE_COL]).dt.days
        dt = dt.where(dt >= 0, (next_year - out[DATE_COL]).dt.days)
        out[f"days_to_{name}"] = dt.clip(lower=0, upper=180)

    black_fridays = pd.to_datetime([_black_friday(y) for y in range(min(years), max(years) + 2)])
    out["is_black_friday"] = out[DATE_COL].isin(black_fridays).astype("int8")
    out["is_cyber_monday"] = out[DATE_COL].isin(black_fridays + pd.Timedelta(days=3)).astype("int8")

    eom = out[DATE_COL] + pd.offsets.MonthEnd(0)
    out["is_mid_month_pay"] = (out[DATE_COL].dt.day == 15).astype("int8")
    out["is_eom_pay"] = (out[DATE_COL] == eom).astype("int8")
    out["is_payday_window"] = (
        (out[DATE_COL].dt.day.between(14, 17)) | ((eom - out[DATE_COL]).dt.days <= 2)
    ).astype("int8")
    return out


def build_deterministic_feature_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    base = pd.DataFrame({DATE_COL: pd.date_range(start_date, end_date, freq="D")})
    base = add_basic_calendar(base)
    base = add_cyclical(base)
    base = add_tet_features(base)
    base = add_event_features(base)
    return base.set_index(DATE_COL)


def make_lag_roll_features(series: pd.Series, horizon: int = 1) -> pd.DataFrame:
    s = series.astype(float).copy()
    out = pd.DataFrame(index=series.index)

    for lag in LAGS:
        out[f"lag_{lag}"] = s.shift(lag)
    for lag in [7, 14, 21, 28, 35, 49]:
        out[f"dow_lag_{lag}"] = s.shift(lag)

    base = s.shift(horizon)
    for w in ROLL_WINDOWS:
        r = base.rolling(w, min_periods=max(3, w // 3))
        out[f"rmean_{w}"] = r.mean()
        out[f"rstd_{w}"] = r.std()
        out[f"rmin_{w}"] = r.min()
        out[f"rmax_{w}"] = r.max()
        out[f"rmed_{w}"] = r.median()
        out[f"rcv_{w}"] = out[f"rstd_{w}"] / (out[f"rmean_{w}"].abs() + 1e-6)

    for alpha in EWM_ALPHAS:
        ewm = base.ewm(alpha=alpha, adjust=False)
        out[f"ewm_a{alpha}"] = ewm.mean()
        out[f"ewm_std_a{alpha}"] = ewm.std()

    out["diff_1"] = s.shift(1) - s.shift(2)
    out["diff_7"] = s.shift(1) - s.shift(8)
    out["yoy_lag"] = s.shift(365)
    out["yoy_diff"] = s.shift(1) - s.shift(365)
    out["yoy_ratio"] = s.shift(1) / (s.shift(365) + 1.0)
    out["yoy_roll28"] = s.shift(365).rolling(28, min_periods=10).mean()
    return out


def lag_roll_row_from_history(history: pd.Series) -> Dict[str, float]:
    h = history.astype(float).dropna()
    feats: Dict[str, float] = {}
    n = len(h)
    arr = h.to_numpy()

    for lag in LAGS:
        feats[f"lag_{lag}"] = float(arr[-lag]) if n >= lag else np.nan
    for lag in [7, 14, 21, 28, 35, 49]:
        feats[f"dow_lag_{lag}"] = float(arr[-lag]) if n >= lag else np.nan

    for w in ROLL_WINDOWS:
        min_periods = max(3, w // 3)
        if n >= min_periods:
            window_vals = arr[-w:]
            feats[f"rmean_{w}"] = float(np.nanmean(window_vals))
            feats[f"rstd_{w}"] = float(np.nanstd(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
            feats[f"rmin_{w}"] = float(np.nanmin(window_vals))
            feats[f"rmax_{w}"] = float(np.nanmax(window_vals))
            feats[f"rmed_{w}"] = float(np.nanmedian(window_vals))
            feats[f"rcv_{w}"] = feats[f"rstd_{w}"] / (abs(feats[f"rmean_{w}"]) + 1e-6)
        else:
            feats[f"rmean_{w}"] = np.nan
            feats[f"rstd_{w}"] = np.nan
            feats[f"rmin_{w}"] = np.nan
            feats[f"rmax_{w}"] = np.nan
            feats[f"rmed_{w}"] = np.nan
            feats[f"rcv_{w}"] = np.nan

    h_series = h.copy()
    for alpha in EWM_ALPHAS:
        feats[f"ewm_a{alpha}"] = float(h_series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]) if n else np.nan
        feats[f"ewm_std_a{alpha}"] = (
            float(h_series.ewm(alpha=alpha, adjust=False).std().iloc[-1]) if n > 1 else 0.0
        )

    feats["diff_1"] = float(arr[-1] - arr[-2]) if n >= 2 else np.nan
    feats["diff_7"] = float(arr[-1] - arr[-8]) if n >= 8 else np.nan
    feats["yoy_lag"] = float(arr[-365]) if n >= 365 else np.nan
    feats["yoy_diff"] = float(arr[-1] - arr[-365]) if n >= 365 else np.nan
    feats["yoy_ratio"] = float(arr[-1] / (arr[-365] + 1.0)) if n >= 365 else np.nan
    feats["yoy_roll28"] = float(np.nanmean(arr[-(365 + 28) : -365])) if n >= (365 + 10) else np.nan
    return feats

