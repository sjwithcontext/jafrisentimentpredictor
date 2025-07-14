from pathlib import Path
from datetime import datetime

import pandas as pd

from .data import fetch_price
from .features import moving_average, relative_strength_index, realized_volatility
from .model import train_baseline, train_boosted, evaluate, save_model

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["ma2"] = moving_average(df["close"], 2)
    features["ma10"] = moving_average(df["close"], 10)
    features["ma30"] = moving_average(df["close"], 30)
    features["rsi15"] = relative_strength_index(df["close"], 15)
    features["volatility5"] = realized_volatility(df["close"], 5)
    return features


def prepare_target(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) > df["close"]).astype(int)


def run(ticker: str = "AAPL") -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    prices = fetch_price(ticker, "2019-01-01", datetime.now().strftime("%Y-%m-%d"))
    prices.to_csv(DATA_DIR / "prices.csv")

    features = build_features(prices)
    target = prepare_target(prices)
    dataset = features.dropna()
    target = target.loc[dataset.index]

    X_train = dataset.loc[:"2023-12-31"]
    y_train = target.loc[:"2023-12-31"]
    X_test = dataset.loc["2024-01-01":]
    y_test = target.loc["2024-01-01":]

    baseline = train_baseline(X_train, y_train)
    boosted = train_boosted(X_train, y_train)

    print("Baseline:", evaluate(baseline, X_test, y_test))
    print("Boosted:", evaluate(boosted, X_test, y_test))

    save_model(boosted, MODEL_DIR / "model.joblib")


if __name__ == "__main__":
    run()
