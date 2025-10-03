from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def add_price_features(df: pd.DataFrame, date_col: str, ticker_col: str, lags: int, windows: tuple[int, ...]) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = np.log(df["close"]).diff()
    # per ticker lags/rolls
    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        # daily log-return
        g["r1"] = np.log(g["close"]).diff()
        # future targets for training
        g["target_r1"] = g["r1"].shift(-1)
        # 20-day cumulative future return (log-sum)
        g["target_R20"] = g["r1"].rolling(20).sum().shift(-20)
        # directional labels
        g["label_up_1"] = (g["target_r1"] > 0).astype(float)
        g["label_up_20"] = (g["target_R20"] > 0).astype(float)
        # ATR-like range
        g["range"] = (g["high"] - g["low"]).replace(0, np.nan)
        # lags
        for k in range(1, lags + 1):
            g[f"r1_lag{k}"] = g["r1"].shift(k)
        # rolling stats
        for w in windows:
            g[f"r1_roll_mean_{w}"] = g["r1"].rolling(w).mean()
            g[f"r1_roll_std_{w}"] = g["r1"].rolling(w).std()
            g[f"range_roll_mean_{w}"] = g["range"].rolling(w).mean()
        return g

    df = df.groupby(ticker_col, group_keys=False).apply(_per_ticker)
    return df


def build_news_matrix(
    news: pd.DataFrame,
    price_index: pd.DataFrame,
    date_col: str,
    ticker_col: str,
    max_features: int = 20000,
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    # aggregate per (ticker, date) with one-day lag (use only ≤ t-1)
    news = news.copy()
    news["date"] = pd.to_datetime(news["news_date"])  # normalized date
    news["date"] = news["date"] + pd.Timedelta(days=1)  # shift to align as available at t

    news["text"] = (news["title"].fillna("") + " \n" + news["publication"].fillna(""))
    # limit rows per day/ticker to cap extreme volume (optional)
    news = news.groupby(["ticker", news["date"]]).head(1000)

    # join keys present in price_index to reduce vocab
    keys = price_index[[ticker_col, date_col]].copy()
    keys = keys.rename(columns={date_col: "date"})

    merged = news.merge(keys, on=["ticker", "date"], how="inner")
    if merged.empty:
        # no news aligned; return empty frame aligned with keys
        mat = pd.DataFrame(index=pd.MultiIndex.from_frame(keys[["ticker", "date"]]), data={})
        return mat.reset_index(), TfidfVectorizer()

    # vectorize
    tfidf = TfidfVectorizer(max_features=max_features, lowercase=True, ngram_range=(1, 2))
    X = tfidf.fit_transform(merged["text"].astype(str))
    # average per (ticker, date)
    groups = merged[["ticker", "date"]].reset_index(drop=True)
    df_sparse = pd.DataFrame.sparse.from_spmatrix(X)
    df_sparse["ticker"] = groups["ticker"].values
    df_sparse["date"] = groups["date"].values
    agg = df_sparse.groupby(["ticker", "date"], as_index=False).mean(numeric_only=True)

    # align to price_index
    feat = keys.merge(agg, on=["ticker", "date"], how="left").fillna(0.0)
    feat = feat.rename(columns={"date": date_col})
    return feat, tfidf
