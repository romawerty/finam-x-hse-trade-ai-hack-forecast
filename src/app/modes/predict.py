from __future__ import annotations

from typing import Optional

import pandas as pd

from ..core.config import Config
from ..core.model import Artifacts
from ..domain.data import DataLoader
from ..domain.features import add_price_features, build_news_matrix


def predict(
    candles_csv: str,
    news_csv: Optional[str],
    artifacts_dir: str,
    outfile: str,
    cfg: Config = Config(),
) -> None:
    dl = DataLoader(cfg.date_col, cfg.ticker_col)
    candles = dl.load_candles(candles_csv)
    feat_price = add_price_features(candles, cfg.date_col, cfg.ticker_col, cfg.price_lags, cfg.roll_windows)

    # keep only rows where we can predict (non-NaN in lags)
    base_num_cols = [
        *[f"r1_lag{k}" for k in range(1, cfg.price_lags + 1)],
        *[f"r1_roll_mean_{w}" for w in cfg.roll_windows],
        *[f"r1_roll_std_{w}" for w in cfg.roll_windows],
        *[f"range_roll_mean_{w}" for w in cfg.roll_windows],
        "volume",
    ]
    feat_price = feat_price.dropna(subset=base_num_cols).copy()

    if news_csv:
        news = dl.load_news(news_csv)
        news_feat, _ = build_news_matrix(news, feat_price[[cfg.ticker_col, cfg.date_col]], cfg.date_col, cfg.ticker_col, cfg.tfidf_max_features)
        X = feat_price.merge(news_feat, on=[cfg.ticker_col, cfg.date_col], how="left").fillna(0.0)
    else:
        X = feat_price.copy()

    # load artifacts & meta
    art = Artifacts.load(artifacts_dir)
    meta = pd.read_json(f"{artifacts_dir}/meta.json", typ="series")
    num_cols = list(meta["num_cols"])  # type: ignore
    cat_cols = list(meta["cat_cols"])  # type: ignore

    # ensure columns present (fill missing TF-IDF cols with 0)
    for c in num_cols:
        if c not in X.columns:
            X[c] = 0.0

    # predictions
    r1_pred = art.pipe_r1.predict(X[[*num_cols, *cat_cols]])
    R20_pred = art.pipe_R20.predict(X[[*num_cols, *cat_cols]])
    p_up_1 = art.pipe_up1.predict_proba(X[[*num_cols, *cat_cols]])[:, 1]
    p_up_20 = art.pipe_up20.predict_proba(X[[*num_cols, *cat_cols]])[:, 1]

    out = pd.DataFrame({
        "date": X[cfg.date_col].dt.date,
        "ticker": X[cfg.ticker_col],
        "r1_pred": r1_pred,
        "R20_pred": R20_pred,
        "p_up_1": p_up_1,
        "p_up_20": p_up_20,
    })
    out = out.sort_values(["date", "ticker"])  # stable order
    out.to_csv(outfile, index=False)

