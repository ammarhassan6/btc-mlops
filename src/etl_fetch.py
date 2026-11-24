import argparse
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

TICKER = "BTC-USD"
START_DATE = "2018-01-01"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

logger = logging.getLogger(__name__)


def fetch_yfinance():
    df = yf.Ticker(TICKER).history(start=START_DATE, interval="1d")
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def fetch_binance():
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
    response = requests.get(BINANCE_URL, params=params, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json()).iloc[:, :6]
    df.columns = COLUMNS
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    return df


def clean(df):
    df = df[COLUMNS].copy()
    df[COLUMNS[1:]] = df[COLUMNS[1:]].astype(float)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # today's candle is not finished yet, so keep only complete days
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = df[df["Date"] < today]

    df = df.drop_duplicates(subset="Date").sort_values("Date")
    return df.dropna()


def fetch_data(out_path):
    try:
        df = fetch_yfinance()
        if df.empty:
            raise ValueError("no rows returned")
        logger.info("Downloaded %d rows from yfinance", len(df))
    except Exception as e:
        logger.warning("yfinance failed (%s), using Binance instead", e)
        df = fetch_binance()
        logger.info("Downloaded %d rows from Binance", len(df))

    df = clean(df)
    if df.empty:
        raise RuntimeError("No price data after cleaning")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    first_date, last_date = df["Date"].iloc[0], df["Date"].iloc[-1]
    logger.info("Saved %d rows (%s to %s) to %s", len(df), first_date, last_date, out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/btc_raw.csv")
    args = parser.parse_args()
    fetch_data(args.out)
