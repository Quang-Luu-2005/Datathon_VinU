"""Time-series cross-validation splitters."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class ExpandingWindowWalkForward:
    """Calendar-order walk-forward splitter with gap embargo."""

    def __init__(
        self,
        initial_train_days: int = 365 * 5,
        val_days: int = 180,
        step_days: int = 180,
        gap_days: int = 28,
        max_splits: int = 8,
    ) -> None:
        self.initial_train_days = initial_train_days
        self.val_days = val_days
        self.step_days = step_days
        self.gap_days = gap_days
        self.max_splits = max_splits

    def get_n_splits(self, X: pd.DataFrame) -> int:
        n = len(X)
        k = max(
            0,
            (n - self.initial_train_days - self.gap_days - self.val_days) // self.step_days + 1,
        )
        return min(k, self.max_splits) if self.max_splits else k

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        idx = np.arange(n)
        train_end = self.initial_train_days
        fold = 0
        while True:
            val_start = train_end + self.gap_days
            val_end = val_start + self.val_days
            if val_end > n:
                break
            if self.max_splits and fold >= self.max_splits:
                break
            yield idx[:train_end], idx[val_start:val_end]
            train_end += self.step_days
            fold += 1

