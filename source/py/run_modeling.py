#!/usr/bin/env python
"""CLI entrypoint for the modular forecasting system."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from forecasting.constants import SEED
from forecasting.metrics import set_seed
from forecasting.pipeline import run_full_pipeline
from forecasting.types import PipelineConfig

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build forecasting model and submission.")
    parser.add_argument("--data-dir", default="dataset", type=str)
    parser.add_argument("--out-dir", default="source/outputs", type=str)
    parser.add_argument("--n-trials", default=35, type=int, help="Optuna trials per target.")
    parser.add_argument("--max-splits", default=8, type=int, help="Walk-forward fold count.")
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--disable-prophet", action="store_true")
    parser.add_argument("--disable-optuna", action="store_true")
    parser.add_argument("--optuna-timeout", default=0, type=int, help="Seconds. 0 disables timeout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        n_trials=args.n_trials,
        max_splits=args.max_splits,
        seed=args.seed,
        disable_prophet=args.disable_prophet,
        disable_optuna=args.disable_optuna,
        optuna_timeout=args.optuna_timeout,
    )
    set_seed(config.seed)
    summary_path = run_full_pipeline(config)
    print(f"Saved metrics to: {summary_path}")
    print(f"Saved submission to: {config.out_dir / 'submission_modeling.csv'}")


if __name__ == "__main__":
    main()

