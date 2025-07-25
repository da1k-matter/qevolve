# EVOLVE-BLOCK-START
"""
CatBoost model for Bitcoin return prediction.

This program loads BTC_30.csv and trains a CatBoostRegressor to predict
future returns. The horizon parameter controls how many bars ahead the
model forecasts returns.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


def train_model(
    horizon: int = 1,
    test_fraction: float = 0.2,
    iterations: int = 200,
    depth: int = 6,
    learning_rate: float = 0.1,
):
    """Train CatBoost on BTC_30.csv and return validation RMSE."""
    df = pd.read_csv("BTC_30.csv")
    df["return"] = df["close"].pct_change().shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    features = df[["open", "high", "low", "close", "volume"]]
    target = df["return"]

    split_idx = int(len(df) * (1 - test_fraction))
    X_train, X_valid = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_valid = target.iloc[:split_idx], target.iloc[split_idx:]

    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

    preds = model.predict(X_valid)
    rmse = float(np.sqrt(np.mean((preds - y_valid) ** 2)))
    return {"rmse": rmse}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    metrics = train_model()
    print(metrics)
