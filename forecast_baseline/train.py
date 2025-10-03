from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .data import DataLoader
from .features import add_price_features, build_news_matrix
from .model import Artifacts, build_pipelines
from .utils import save_json, set_seed


def train(
    candles_csv: str,
    news_csv: Optional[str],
    outdir: str,
    t0: str,
    t1: str,
    cfg: Config = Config(),
) -> None:
    set_seed(cfg.seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    dl = DataLoader(cfg.date_col, cfg.ticker_col)
    candles = dl.load_candles(candles_csv)

    # price features & targets
    feat_price = add_price_features(candles, cfg.date_col, cfg.ticker_col, cfg.price_lags, cfg.roll_windows)

    # restrict to available window for train/val
    feat_price["date_only"] = pd.to_datetime(feat_price[cfg.date_col]).dt.date
    m_train = (feat_price[cfg.date_col] <= pd.to_datetime(t0))
    m_val = (feat_price[cfg.date_col] > pd.to_datetime(t0)) & (feat_price[cfg.date_col] <= pd.to_datetime(t1))

    # news features aligned to available price rows
    if news_csv:
        news = dl.load_news(news_csv)
        news_feat_train, tfidf_train = build_news_matrix(news, feat_price.loc[m_train | m_val, [cfg.ticker_col, cfg.date_col]], cfg.date_col, cfg.ticker_col, cfg.tfidf_max_features)
        # join back
        feat_all = feat_price.merge(news_feat_train, on=[cfg.ticker_col, cfg.date_col], how="left").fillna(0.0)
    else:
        feat_all = feat_price.copy()

    # select columns
    target_cols = ["target_r1", "target_R20", "label_up_1", "label_up_20"]
    base_num_cols = [
        *[f"r1_lag{k}" for k in range(1, cfg.price_lags + 1)],
        *[f"r1_roll_mean_{w}" for w in cfg.roll_windows],
        *[f"r1_roll_std_{w}" for w in cfg.roll_windows],
        *[f"range_roll_mean_{w}" for w in cfg.roll_windows],
        "volume",
    ]
    # news tf-idf columns (if any): everything numeric beyond known cols
    non_feature_cols = set([cfg.date_col, cfg.ticker_col, "open", "high", "low", "close", "ret", "r1", "range", "date_only", *target_cols])
    num_cols = [c for c in feat_all.columns if c not in non_feature_cols and feat_all[c].dtype != "O"]
    # ensure base features included
    for c in base_num_cols:
        if c not in num_cols and c in feat_all.columns:
            num_cols.append(c)
    cat_cols = [cfg.ticker_col]

    # drop rows with NaNs in lags
    clean = feat_all.dropna(subset=base_num_cols).copy()

    # train/val split masks
    tr = clean[clean[cfg.date_col] <= pd.to_datetime(t0)]
    va = clean[(clean[cfg.date_col] > pd.to_datetime(t0)) & (clean[cfg.date_col] <= pd.to_datetime(t1))]

    artifacts = build_pipelines(num_cols=num_cols, cat_cols=cat_cols)

    # fit models
    artifacts.pipe_r1.fit(tr[[*num_cols, *cat_cols]], tr["target_r1"])
    artifacts.pipe_R20.fit(tr[[*num_cols, *cat_cols]], tr["target_R20"])
    artifacts.pipe_up1.fit(tr[[*num_cols, *cat_cols]], tr["label_up_1"])
    artifacts.pipe_up20.fit(tr[[*num_cols, *cat_cols]], tr["label_up_20"])

    # simple val metrics (optional)
    # ... (add if needed)

    # save
    artifacts.save(outdir)
    meta = {
        "t0": t0,
        "t1": t1,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "config": Config().__dict__,
    }
    save_json(f"{outdir}/meta.json", meta)
