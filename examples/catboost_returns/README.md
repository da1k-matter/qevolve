# CatBoost Bitcoin Return Forecasting Example

This example demonstrates how to use **OpenEvolve** to evolve a CatBoost model
for predicting future Bitcoin returns. The dataset `BTC_30.csv` contains OHLCV
records sampled every 30 minutes. The goal is to minimize validation RMSE when
forecasting returns `horizon` bars into the future.

## Files

- `initial_program.py` – Starting CatBoost implementation inside an EVOLVE-BLOCK.
- `evaluator.py` – Evaluation script computing RMSE and a `combined_score`.
- `config.yaml` – OpenEvolve configuration with a `horizon` parameter.
- `requirements.txt` – Python dependencies.

## Running the Example

```bash
cd examples/catboost_returns
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

OpenEvolve will evolve the code inside the EVOLVE-BLOCK to reduce RMSE on the
validation split. The `horizon` value in `config.yaml` controls how many bars
ahead to predict.
