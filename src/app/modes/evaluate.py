from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.sign(y_true) == np.sign(y_pred)).astype(float)))


def evaluate(pred_csv: str, truth_csv: str) -> dict[str, float]:
    pred = pd.read_csv(pred_csv)
    truth = pd.read_csv(truth_csv)
    # expected columns in truth: date, ticker, r1_true, R20_true
    df = truth.merge(pred, on=["date", "ticker"], how="inner")

    m1 = mae(df["r1_true"].values, df["r1_pred"].values)
    m20 = mae(df["R20_true"].values, df["R20_pred"].values)
    b1 = brier((df["r1_true"] > 0).astype(float).values, df["p_up_1"].values)
    b20 = brier((df["R20_true"] > 0).astype(float).values, df["p_up_20"].values)
    da1 = directional_accuracy(df["r1_true"].values, df["r1_pred"].values)
    da20 = directional_accuracy(df["R20_true"].values, df["R20_pred"].values)

    return {
        "MAE_1": m1,
        "MAE_20": m20,
        "Brier_1": b1,
        "Brier_20": b20,
        "DA_1": da1,
        "DA_20": da20,
    }

