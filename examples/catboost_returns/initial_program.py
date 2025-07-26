# EVOLVE-BLOCK-START
"""
CatBoost model for Bitcoin return prediction.

This program loads BTC_30.csv and trains a CatBoostClassifier 
The function now also returns a `combined_score` so that OpenEvolve has the feature dimension it expects.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score


# You can use TA-lib


def train_model(
    horizon: int = 20,
    test_fraction: float = 0.2,
    iterations: int = 200,
    depth: int = 6,
    learning_rate: float = 0.1,
):
    try:
        df = pd.read_csv("BTC_30.csv", parse_dates=["timestamp"])
        df["return"] = df["close"].pct_change().shift(-horizon)
        df["label"] = (df["return"] > 0).astype(int)
        df = df.dropna().reset_index(drop=True)

        features = df[["open", "high", "low", "close", "volume"]]
        target = df["label"]

        split_idx = int(len(df) * (1 - test_fraction))
        X_train, X_valid = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_valid = target.iloc[:split_idx], target.iloc[split_idx:]

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            verbose=True,
            random_seed=42,
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=True)

        # EVOLVE-BLOCK-END

        preds = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
    except Exception as e:
        print(f"Error running program: {e}")
        return {
            "combined_score": 0.0,
            "error": 1,
        }

    return {
        "combined_score": acc,
        "error": 0,
    }


if __name__ == "__main__":
    metrics = train_model()
    print(metrics)
