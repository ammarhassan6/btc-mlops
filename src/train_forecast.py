import argparse
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ARIMA_ORDER = (5, 1, 0)
TEST_DAYS = 7
TRAIN_WINDOW = 730

logger = logging.getLogger(__name__)


def fit_arima(prices):
    return ARIMA(np.asarray(prices, dtype=float), order=ARIMA_ORDER).fit()


def evaluate(prices):
    train, test = prices[:-TEST_DAYS], prices[-TEST_DAYS:]
    preds = fit_arima(train).forecast(TEST_DAYS)
    errors = test - preds
    return {
        "mae": round(float(np.mean(np.abs(errors))), 2),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 2),
        "mape": round(float(np.mean(np.abs(errors / test)) * 100), 2),
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def train_and_evaluate(in_path, model_dir):
    df = pd.read_csv(in_path, parse_dates=["Date"]).sort_values("Date")
    df = df.tail(TRAIN_WINDOW)
    prices = df["Close"].to_numpy()

    metrics = evaluate(prices)
    threshold = float(os.getenv("MAPE_THRESHOLD", "10"))
    metrics["passed"] = metrics["mape"] <= threshold
    metrics["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics["model_version"] = None

    os.makedirs(model_dir, exist_ok=True)
    if metrics["passed"]:
        version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
        bundle = {
            "model": fit_arima(prices),
            "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "version": version,
        }
        tmp_path = os.path.join(model_dir, "model.pkl.tmp")
        joblib.dump(bundle, tmp_path)
        os.replace(tmp_path, os.path.join(model_dir, "model.pkl"))
        metrics["model_version"] = version
        logger.info("Model %s saved (MAPE %.2f%%)", version, metrics["mape"])
    else:
        logger.warning(
            "MAPE %.2f%% above threshold %s, model not saved", metrics["mape"], threshold
        )

    save_json(metrics, os.path.join(model_dir, "metrics.json"))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/btc_features.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.input, args.model_dir), indent=2))
