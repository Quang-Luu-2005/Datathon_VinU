"""Typed containers for configuration and model artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .constants import SEED


@dataclass
class PipelineConfig:
    data_dir: Path = Path("dataset")
    out_dir: Path = Path("source/outputs")
    n_trials: int = 35
    seed: int = SEED
    disable_prophet: bool = False
    disable_optuna: bool = False
    optuna_timeout: int = 0
    use_l2_head: bool = True
    use_tweedie_head: bool = True
    use_quantile_head: bool = True
    use_catboost_head: bool = True
    short_cv_max_splits: int = 3
    short_cv_initial_train_days: int = 730
    short_cv_val_days: int = 365
    short_cv_step_days: int = 180
    short_cv_gap_days: int = 28
    realistic_probe_enabled: bool = True
    realistic_probe_initial_train_days: int = 2555
    realistic_probe_val_days: int = 548
    realistic_probe_gap_days: int = 28
    block_bootstrap_n: int = 500
    block_bootstrap_block_days: int = 30


@dataclass
class ForecastArtifacts:
    target: str
    oof_frame: pd.DataFrame
    metrics: Dict[str, Dict[str, float]]
    blend_weights: Dict[str, float]
    blend_weights_bucket: Dict[str, Dict[str, float]]
    event_multipliers: Dict[str, float]
    best_params: Dict[str, float]
    realistic_probe_metrics: Dict[str, float] = field(default_factory=dict)
    used_heads: List[str] = field(default_factory=list)
