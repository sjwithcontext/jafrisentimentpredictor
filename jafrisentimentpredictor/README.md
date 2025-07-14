# JAFRI Sentiment Predictor

This repository contains a minimal implementation of the project described in the prompt.
It fetches Apple (AAPL) price data, derives technical and sentiment features and trains
both a logistic regression baseline and a LightGBM classifier. A small FastAPI service
can serve the latest prediction using the trained model.

## Requirements

* Python 3.10+
* Packages listed in `pyproject.toml`

Install dependencies with:

```bash
pip install -e .
```

## Usage

Run the full pipeline which downloads data, builds features, trains the models and
persists the boosted model under `models/model.joblib`:

```bash
python -m jafrisentimentpredictor.pipeline
```

Start the API after training:

```bash
uvicorn jafrisentimentpredictor.server:app --reload
```

The `/predict` endpoint returns a simple JSON document with the date of the latest price
observation and the model's prediction for the next day's direction.
