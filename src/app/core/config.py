from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 1337
    n_jobs: int = 4
    max_news_per_day_per_ticker: int = 1000  # cap before aggregation
    tfidf_max_features: int = 20000
    price_lags: int = 5
    roll_windows: tuple[int, ...] = (5, 10, 20)

    # columns
    date_col: str = "begin"
    ticker_col: str = "ticker"

    # artifacts
    artifacts_dir: Path = Path("artifacts")

