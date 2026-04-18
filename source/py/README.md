# Python Package (`source/py`)

Production-style modular layout for the forecasting pipeline.

## Structure

- `run_modeling.py`: CLI entrypoint.
- `forecasting/constants.py`: constants/config defaults.
- `forecasting/types.py`: typed dataclasses.
- `forecasting/cv.py`: expanding walk-forward splitter.
- `forecasting/features.py`: calendar/event + lag/rolling features.
- `forecasting/data_io.py`: dataset loaders + seasonal naive helper.
- `forecasting/models.py`: tuning/training/blend/residual/recursive logic.
- `forecasting/pipeline.py`: orchestration for Revenue + COGS.
- `forecasting/metrics.py`: metrics + seed helper.

## Run

```bash
python source/py/run_modeling.py --data-dir dataset --out-dir source/outputs
```

