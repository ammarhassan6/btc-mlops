import numpy as np
import pandas as pd
import pytest

from app.app import app
from src.train_forecast import train_and_evaluate

API_KEY = "test-key"


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(42)
    prices = 50000 + np.cumsum(rng.normal(0, 200, 300))
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    csv_path = folder / "features.csv"
    pd.DataFrame({"Date": dates, "Close": prices}).to_csv(csv_path, index=False)
    train_and_evaluate(str(csv_path), str(folder))
    return folder


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("API_KEY", API_KEY)
    return app.test_client()


def test_training_creates_metrics(trained_model_dir):
    assert (trained_model_dir / "model.pkl").exists()
    assert (trained_model_dir / "metrics.json").exists()


def test_health(client):
    res = client.get("/health")
    body = res.get_json()
    assert res.status_code == 200
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "uptime_seconds" in body
    assert "model_stage" in body


def test_predict_default_returns_7_days(client):
    res = client.post("/predict")
    body = res.get_json()
    assert res.status_code == 200
    assert len(body["forecast"]) == 7
    assert {"date", "predicted_close"} == set(body["forecast"][0])


def test_predict_custom_days(client):
    res = client.post("/predict", json={"forecast_days": 3})
    assert res.status_code == 200
    assert len(res.get_json()["forecast"]) == 3


def test_predict_with_history(client):
    history = [60000 + i * 10 for i in range(60)]
    res = client.post("/predict", json={"history": history})
    assert res.status_code == 200
    assert len(res.get_json()["forecast"]) == 7


@pytest.mark.parametrize("days", [0, 31, "seven", 2.5, True])
def test_predict_invalid_days(client, days):
    res = client.post("/predict", json={"forecast_days": days})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_short_history(client):
    res = client.post("/predict", json={"history": [1, 2, 3]})
    assert res.status_code == 400


def test_predict_bad_history_values(client):
    res = client.post("/predict", json={"history": ["a"] * 40})
    assert res.status_code == 400


def test_predict_invalid_json(client):
    res = client.post("/predict", data="not json", content_type="application/json")
    assert res.status_code == 400


def test_predict_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    res = app.test_client().post("/predict")
    assert res.status_code == 503


def test_metrics(client):
    res = client.get("/metrics")
    body = res.get_json()
    assert res.status_code == 200
    for key in ["mae", "rmse", "mape", "passed"]:
        assert key in body


def test_reload_requires_api_key(client):
    assert client.post("/reload").status_code == 401
    assert client.post("/reload", headers={"X-API-Key": "wrong"}).status_code == 401


def test_reload_with_api_key(client):
    res = client.post("/reload", headers={"X-API-Key": API_KEY})
    assert res.status_code == 200
    assert res.get_json()["status"] == "reloaded"


def test_unknown_route(client):
    assert client.get("/nothing").status_code == 404
