from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .model import load_model
from .pipeline import build_features
from .data import fetch_price

app = FastAPI()
MODEL_PATH = Path("models/model.joblib")
model = None


class Prediction(BaseModel):
    date: str
    prediction: int


@app.on_event("startup")
def load() -> None:
    global model
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)


@app.get("/predict", response_model=Prediction)
def predict() -> Prediction:
    if model is None:
        raise RuntimeError("Model not loaded")

    latest = fetch_price("AAPL", start="2023-12-01", end=None).iloc[-1:]
    features = build_features(latest).iloc[-1:]
    pred = int(model.predict(features)[0])
    return Prediction(date=str(latest.index[-1].date()), prediction=pred)
