import argparse
import logging
import os

import pandas as pd

FEATURE_COLUMNS = ["Date", "Close", "Volume", "return_1d", "ma_7", "ma_30", "volatility_7"]

logger = logging.getLogger(__name__)


def add_features(df):
    df["return_1d"] = df["Close"].pct_change()
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["volatility_7"] = df["return_1d"].rolling(7).std()
    return df


def preprocess(in_path, out_path):
    df = pd.read_csv(in_path, parse_dates=["Date"])
    df = df.drop_duplicates(subset="Date").sort_values("Date")

    df = add_features(df)
    # the first 30 rows have no ma_30 yet
    df = df[FEATURE_COLUMNS].dropna()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows with features to %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/btc_raw.csv")
    parser.add_argument("--out", default="data/processed/btc_features.csv")
    args = parser.parse_args()
    preprocess(args.input, args.out)
