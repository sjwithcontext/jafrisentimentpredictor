from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


@dataclass
class ModelResults:
    accuracy: float
    f1: float
    mcc: float


def train_baseline(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def train_boosted(X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05)
    model.fit(X, y)
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> ModelResults:
    pred = model.predict(X)
    return ModelResults(
        accuracy=accuracy_score(y, pred),
        f1=f1_score(y, pred),
        mcc=matthews_corrcoef(y, pred),
    )


def save_model(model, path: Path) -> None:
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)
