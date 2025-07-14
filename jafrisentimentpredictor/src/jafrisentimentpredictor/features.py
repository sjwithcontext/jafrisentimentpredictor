import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def relative_strength_index(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window).mean()
    loss = down.rolling(window).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def realized_volatility(series: pd.Series, window: int) -> pd.Series:
    log_returns = series.pct_change().apply(lambda x: np.log(1 + x))
    return log_returns.rolling(window).std() * (window ** 0.5)
