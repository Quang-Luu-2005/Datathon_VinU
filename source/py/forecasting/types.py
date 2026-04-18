"""Typed containers for configuration and model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from .constants import SEED


@dataclass
class PipelineConfig:
    data_dir: Path = Path("dataset")
    out_dir: Path = Path("source/outputs")
    n_trials: int = 35
    max_splits: int = 8
    seed: int = SEED
    disable_prophet: bool = False
    disable_optuna: bool = False
    optuna_timeout: int = 0


@dataclass
class ForecastArtifacts:
    target: str
    oof_frame: pd.DataFrame
    metrics: Dict[str, Dict[str, float]]
    blend_weights: Dict[str, float]
    event_multipliers: Dict[str, float]
    best_params: Dict[str, float]

