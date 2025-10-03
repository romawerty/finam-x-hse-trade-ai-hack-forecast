# FORECAST — Minimal, Scalable Baseline

**Goal**: simple & fast baseline with strict time scheme (no leakage), two horizons (t+1 daily return, 20-day cumulative return), and up-probabilities.

## Quickstart

```bash
python -m src.app train \
  --candles data/raw/task_1_candles.csv \
  --news data/raw/task_5_news.csv \
  --outdir artifacts/ \
  --t0 2023-06-30 --t1 2023-12-31

python -m src.app predict \
  --candles data/raw/task_1_candles.csv \
  --news data/raw/task_5_news.csv \
  --artifacts artifacts/ \
  --outfile outputs/submission.csv

python -m src.app evaluate \
  --pred outputs/submission.csv \
  --truth data/processed/ground_truth.csv
```

## Data format (given by organizers)
- `task_1_candles.csv`: columns `begin, ticker, open, high, low, close, volume`.
- `task_5_news.csv`: columns `publish_date, title, publication, tickers`.

## Submission format
CSV with columns:
`date,ticker,r1_pred,R20_pred,p_up_1,p_up_20`

## Reproducibility
- Fixed seed, sklearn RNG.
- Logged config + versions.

## Model
- Linear models (Ridge) for `r_{t+1}`, `R_{t+20}`.
- LogisticRegression for `p_up` on both horizons.
- Global models across tickers with one-hot ticker.

## Features
- Price features: lag returns, rolling mean/std, z-score, ATR-like range.
- News features: TF-IDF over title+publication, aggregated by (ticker, date), shifted to `t-1`.

## Notes
- Replace TF-IDF with Hashing if speed/footprint is critical.
- Replace linear models with tree-based (LGBM/CatBoost) if needed; interface unchanged.
