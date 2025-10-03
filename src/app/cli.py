from __future__ import annotations

import argparse
import json

from .core.config import Config
from .modes.evaluate import evaluate
from .modes.predict import predict
from .modes.train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="forecast-baseline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--candles", required=True)
    p_train.add_argument("--news", required=False)
    p_train.add_argument("--outdir", required=True)
    p_train.add_argument("--t0", required=True, help="last date of train (inclusive)")
    p_train.add_argument("--t1", required=True, help="last date of val (inclusive)")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--candles", required=True)
    p_pred.add_argument("--news")
    p_pred.add_argument("--artifacts", required=True)
    p_pred.add_argument("--outfile", required=True)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--pred", required=True)
    p_eval.add_argument("--truth", required=True)

    args = parser.parse_args()
    cfg = Config()

    if args.cmd == "train":
        train(
            candles_csv=args.candles,
            news_csv=args.news,
            outdir=args.outdir,
            t0=args.t0,
            t1=args.t1,
            cfg=cfg,
        )
    elif args.cmd == "predict":
        predict(
            candles_csv=args.candles,
            news_csv=args.news,
            artifacts_dir=args.artifacts,
            outfile=args.outfile,
            cfg=cfg,
        )
    elif args.cmd == "evaluate":
        res = evaluate(args.pred, args.truth)
        print(json.dumps(res, indent=2))
    else:
        parser.error("Unknown command")

