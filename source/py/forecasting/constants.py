"""Central constants used across the forecasting pipeline."""

from __future__ import annotations

import pandas as pd

SEED = 42
DATE_COL = "date"
TARGETS = ("revenue", "cogs", "cogs_ratio")

# Used when lunardate is unavailable.
TET_DATES_FALLBACK = pd.to_datetime(
    [
        "2012-01-23",
        "2013-02-10",
        "2014-01-31",
        "2015-02-19",
        "2016-02-08",
        "2017-01-28",
        "2018-02-16",
        "2019-02-05",
        "2020-01-25",
        "2021-02-12",
        "2022-02-01",
        "2023-01-22",
        "2024-02-10",
        "2025-01-29",
        "2026-02-17",
        "2027-02-06",
        "2028-01-26",
    ]
)

LAGS = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365]
ROLL_WINDOWS = [7, 14, 28, 56, 90]
EWM_ALPHAS = [0.05, 0.1, 0.2, 0.4]

EVENT_COLS = [
    "is_tet_week",
    "is_dd_9_9",
    "is_dd_10_10",
    "is_dd_11_11",
    "is_dd_12_12",
    "is_black_friday",
    "is_payday_window",
]

# v4 bucket design: merged long horizon bucket.
HORIZON_BUCKETS = (
    ("h01_030", 1, 30),
    ("h031_090", 31, 90),
    ("h091_180", 91, 180),
    ("h181_plus", 181, 9999),
)

DEFAULT_TUNED_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.01,
    "lambda_l2": 0.1,
    "min_gain_to_split": 0.0,
    "max_bin": 63,
}
