from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Artifacts:
    pipe_r1: Pipeline
    pipe_R20: Pipeline
    pipe_up1: Pipeline
    pipe_up20: Pipeline
    tfidf_vocab_path: Optional[str] = None

    def save(self, outdir: str) -> None:
        joblib.dump(self.pipe_r1, f"{outdir}/pipe_r1.joblib")
        joblib.dump(self.pipe_R20, f"{outdir}/pipe_R20.joblib")
        joblib.dump(self.pipe_up1, f"{outdir}/pipe_up1.joblib")
        joblib.dump(self.pipe_up20, f"{outdir}/pipe_up20.joblib")

    @staticmethod
    def load(outdir: str) -> "Artifacts":
        return Artifacts(
            pipe_r1=joblib.load(f"{outdir}/pipe_r1.joblib"),
            pipe_R20=joblib.load(f"{outdir}/pipe_R20.joblib"),
            pipe_up1=joblib.load(f"{outdir}/pipe_up1.joblib"),
            pipe_up20=joblib.load(f"{outdir}/pipe_up20.joblib"),
        )


def _make_column_transformer(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def build_pipelines(num_cols: list[str], cat_cols: list[str]) -> Artifacts:
    ct = _make_column_transformer(num_cols, cat_cols)
    pipe_r1 = Pipeline([
        ("ct", ct),
        ("reg", Ridge(alpha=1.0, random_state=0))
    ])
    pipe_R20 = Pipeline([
        ("ct", ct),
        ("reg", Ridge(alpha=1.0, random_state=0))
    ])
    pipe_up1 = Pipeline([
        ("ct", ct),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    pipe_up20 = Pipeline([
        ("ct", ct),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    return Artifacts(pipe_r1=pipe_r1, pipe_R20=pipe_R20, pipe_up1=pipe_up1, pipe_up20=pipe_up20)
