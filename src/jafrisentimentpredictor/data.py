import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def fetch_price(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV price data using yfinance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns=str.lower)
    return df


def score_sentiment(texts: list[str]) -> pd.DataFrame:
    """Score a list of texts with VADER."""
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t) for t in texts]
    return pd.DataFrame(scores)
