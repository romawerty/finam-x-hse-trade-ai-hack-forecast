from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class DataPaths:
    candles_csv: str
    news_csv: str | None = None


class DataLoader:
    def __init__(self, date_col: str = "begin", ticker_col: str = "ticker") -> None:
        self.date_col = date_col
        self.ticker_col = ticker_col

    def load_candles(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.ticker_col, self.date_col])
        return df

    def load_news(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["publish_date"] = pd.to_datetime(df["publish_date"])
        # extract first ticker per row (if list-like); keep simple
        df["ticker"] = df["tickers"].astype(str).str.split(",").str[0].str.strip()
        # normalize to date for join and shift later
        df["news_date"] = df["publish_date"].dt.date
        return df
