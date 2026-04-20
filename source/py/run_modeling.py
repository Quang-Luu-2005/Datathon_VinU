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
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--disable-prophet", action="store_true")
    parser.add_argument("--disable-optuna", action="store_true")
    parser.add_argument("--optuna-timeout", default=0, type=int, help="Seconds. 0 disables timeout.")
    parser.add_argument("--short-cv-max-splits", default=3, type=int)
    parser.add_argument("--short-cv-initial-train-days", default=730, type=int)
    parser.add_argument("--short-cv-val-days", default=365, type=int)
    parser.add_argument("--short-cv-step-days", default=180, type=int)
    parser.add_argument("--short-cv-gap-days", default=28, type=int)
    parser.add_argument("--disable-realistic-probe", action="store_true")
    parser.add_argument("--realistic-probe-initial-train-days", default=2555, type=int)
    parser.add_argument("--realistic-probe-val-days", default=548, type=int)
    parser.add_argument("--realistic-probe-gap-days", default=28, type=int)
    parser.add_argument("--disable-l2-head", action="store_true")
    parser.add_argument("--disable-tweedie-head", action="store_true")
    parser.add_argument("--disable-quantile-head", action="store_true")
    parser.add_argument("--disable-catboost-head", action="store_true")
    parser.add_argument("--bootstrap-n", default=500, type=int)
    parser.add_argument("--bootstrap-block-days", default=30, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        n_trials=args.n_trials,
        seed=args.seed,
        disable_prophet=args.disable_prophet,
        disable_optuna=args.disable_optuna,
        optuna_timeout=args.optuna_timeout,
        use_l2_head=(not args.disable_l2_head),
        use_tweedie_head=(not args.disable_tweedie_head),
        use_quantile_head=(not args.disable_quantile_head),
        use_catboost_head=(not args.disable_catboost_head),
        short_cv_max_splits=args.short_cv_max_splits,
        short_cv_initial_train_days=args.short_cv_initial_train_days,
        short_cv_val_days=args.short_cv_val_days,
        short_cv_step_days=args.short_cv_step_days,
        short_cv_gap_days=args.short_cv_gap_days,
        realistic_probe_enabled=(not args.disable_realistic_probe),
        realistic_probe_initial_train_days=args.realistic_probe_initial_train_days,
        realistic_probe_val_days=args.realistic_probe_val_days,
        realistic_probe_gap_days=args.realistic_probe_gap_days,
        block_bootstrap_n=args.bootstrap_n,
        block_bootstrap_block_days=args.bootstrap_block_days,
    )
    set_seed(config.seed)
    summary_path = run_full_pipeline(config)
    print(f"Saved metrics to: {summary_path}")
    print(f"Saved submission to: {config.out_dir / 'submission_modeling.csv'}")


if __name__ == "__main__":
    main()
