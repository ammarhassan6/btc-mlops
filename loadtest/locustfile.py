import random

from locust import HttpUser, between, task


def make_history(length=200, start=60000.0):
    prices = [start]
    for _ in range(length - 1):
        prices.append(round(prices[-1] * (1 + random.uniform(-0.02, 0.02)), 2))
    return prices


class ForecastUser(HttpUser):
    wait_time = between(1, 2)

    # sending a history makes the API refit ARIMA, which uses a lot of CPU
    @task(3)
    def predict_with_history(self):
        payload = {"history": make_history(), "forecast_days": 7}
        self.client.post("/predict", json=payload, name="/predict (history)")

    @task(1)
    def health(self):
        self.client.get("/health")
