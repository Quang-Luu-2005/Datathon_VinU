"""Time-series cross-validation splitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import pandas as pd


@dataclass
class FoldIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold_id: int
    fold_type: str


class ExpandingWindowWalkForward:
    """Calendar-order walk-forward splitter with gap embargo."""

    def __init__(
        self,
        initial_train_days: int = 365 * 2,
        val_days: int = 365,
        step_days: int = 180,
        gap_days: int = 28,
        max_splits: int = 3,
        fold_type: str = "short_cv",
    ) -> None:
        self.initial_train_days = initial_train_days
        self.val_days = val_days
        self.step_days = step_days
        self.gap_days = gap_days
        self.max_splits = max_splits
        self.fold_type = fold_type

    def get_n_splits(self, X: pd.DataFrame) -> int:
        n = len(X)
        if self.step_days <= 0:
            k = int(n >= (self.initial_train_days + self.gap_days + self.val_days))
            return min(k, self.max_splits) if self.max_splits else k
        k = max(
            0,
            (n - self.initial_train_days - self.gap_days - self.val_days) // self.step_days + 1,
        )
        return min(k, self.max_splits) if self.max_splits else k

    def split(self, X: pd.DataFrame) -> Iterator[FoldIndices]:
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
            yield FoldIndices(
                train_idx=idx[:train_end],
                val_idx=idx[val_start:val_end],
                fold_id=fold,
                fold_type=self.fold_type,
            )
            train_end += max(self.step_days, 1)
            fold += 1


class HybridWalkForwardCV:
    """Hybrid CV: short folds for tuning + optional realistic probe fold."""

    def __init__(
        self,
        short_initial_train_days: int = 730,
        short_val_days: int = 365,
        short_step_days: int = 180,
        short_gap_days: int = 28,
        short_max_splits: int = 3,
        realistic_probe_enabled: bool = True,
        realistic_probe_initial_train_days: int = 2555,
        realistic_probe_val_days: int = 548,
        realistic_probe_gap_days: int = 28,
    ) -> None:
        self.short_splitter = ExpandingWindowWalkForward(
            initial_train_days=short_initial_train_days,
            val_days=short_val_days,
            step_days=short_step_days,
            gap_days=short_gap_days,
            max_splits=short_max_splits,
            fold_type="short_cv",
        )
        self.realistic_probe_enabled = realistic_probe_enabled
        self.realistic_probe_initial_train_days = realistic_probe_initial_train_days
        self.realistic_probe_val_days = realistic_probe_val_days
        self.realistic_probe_gap_days = realistic_probe_gap_days

    def split_short(self, X: pd.DataFrame) -> List[FoldIndices]:
        return list(self.short_splitter.split(X))

    def split_realistic_probe(self, X: pd.DataFrame) -> List[FoldIndices]:
        if not self.realistic_probe_enabled:
            return []
        n = len(X)
        needed = (
            self.realistic_probe_initial_train_days
            + self.realistic_probe_gap_days
            + self.realistic_probe_val_days
        )
        if n < needed:
            # Fallback: use the latest feasible split for realistic estimate.
            val_end = n
            val_start = max(0, val_end - self.realistic_probe_val_days)
            train_end = max(0, val_start - self.realistic_probe_gap_days)
            if train_end <= 0 or val_start >= val_end:
                return []
            idx = np.arange(n)
            return [
                FoldIndices(
                    train_idx=idx[:train_end],
                    val_idx=idx[val_start:val_end],
                    fold_id=0,
                    fold_type="realistic_probe",
                )
            ]

        idx = np.arange(n)
        train_end = self.realistic_probe_initial_train_days
        val_start = train_end + self.realistic_probe_gap_days
        val_end = val_start + self.realistic_probe_val_days
        if val_end > n:
            # Align to tail while keeping fixed val length.
            val_end = n
            val_start = max(0, val_end - self.realistic_probe_val_days)
            train_end = max(0, val_start - self.realistic_probe_gap_days)
        if train_end <= 0 or val_start >= val_end:
            return []
        return [
            FoldIndices(
                train_idx=idx[:train_end],
                val_idx=idx[val_start:val_end],
                fold_id=0,
                fold_type="realistic_probe",
            )
        ]

    def split_all(self, X: pd.DataFrame) -> List[FoldIndices]:
        return self.split_short(X) + self.split_realistic_probe(X)
