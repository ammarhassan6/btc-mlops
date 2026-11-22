import json
import logging
import os
import time
from datetime import date, datetime, timedelta

import joblib
from flask import Flask, jsonify, request

from src.train_forecast import fit_arima

DEFAULT_DAYS = 7
MAX_DAYS = 30
MIN_HISTORY = 30

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("btc-api")

app = Flask(__name__)
START_TIME = time.time()
cache = {"bundle": None, "path": None, "mtime": None}


def model_dir():
    return os.getenv("MODEL_DIR", "models")


def get_model():
    path = os.path.join(model_dir(), "model.pkl")
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    if path != cache["path"] or mtime != cache["mtime"]:
        try:
            cache["bundle"] = joblib.load(path)
            cache["path"], cache["mtime"] = path, mtime
            logger.info("Loaded model %s", cache["bundle"]["version"])
        except Exception:
            logger.exception("Could not load model from %s", path)
    return cache["bundle"]


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_payload():
    if not request.data:
        data = {}
    else:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")

    days = data.get("forecast_days", DEFAULT_DAYS)
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= MAX_DAYS:
        raise ValueError(f"forecast_days must be an integer between 1 and {MAX_DAYS}")

    history = data.get("history")
    if history is not None:
        if not isinstance(history, list) or len(history) < MIN_HISTORY:
            raise ValueError(f"history must be a list of at least {MIN_HISTORY} prices")
        if not all(is_number(p) and p > 0 for p in history):
            raise ValueError("history must contain only positive numbers")
    return days, history


@app.get("/health")
def health():
    bundle = get_model()
    return jsonify(
        status="ok",
        uptime_seconds=round(time.time() - START_TIME, 1),
        model_version=bundle["version"] if bundle else None,
        model_loaded=bundle is not None,
    )


@app.post("/predict")
def predict():
    try:
        days, history = parse_payload()
    except ValueError as e:
        return jsonify(error=str(e)), 400

    try:
        if history is not None:
            preds = fit_arima(history).forecast(days)
            start = date.today()
            version = "custom-history"
        else:
            bundle = get_model()
            if bundle is None:
                return jsonify(error="Model not loaded"), 503
            preds = bundle["model"].forecast(days)
            start = datetime.strptime(bundle["last_date"], "%Y-%m-%d").date()
            version = bundle["version"]

        forecast = [
            {
                "date": (start + timedelta(days=i + 1)).isoformat(),
                "predicted_close": round(float(p), 2),
            }
            for i, p in enumerate(preds)
        ]
        return jsonify(model_version=version, forecast_days=days, forecast=forecast)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify(error="Internal server error"), 500


@app.get("/metrics")
def metrics():
    path = os.path.join(model_dir(), "metrics.json")
    if not os.path.exists(path):
        return jsonify(error="No metrics available yet"), 404
    try:
        with open(path) as f:
            return jsonify(json.load(f))
    except Exception:
        logger.exception("Could not read metrics")
        return jsonify(error="Internal server error"), 500


@app.post("/reload")
def reload_model():
    api_key = os.getenv("API_KEY")
    if not api_key or request.headers.get("X-API-Key") != api_key:
        return jsonify(error="Unauthorized"), 401

    cache["mtime"] = None
    bundle = get_model()
    if bundle is None:
        return jsonify(error="Model not found"), 503
    return jsonify(status="reloaded", model_version=bundle["version"])


@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(error="Method not allowed"), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
