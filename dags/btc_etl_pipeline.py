import logging
import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models")
API_URL = os.getenv("API_URL", "http://api:5000")
API_KEY = os.getenv("API_KEY", "")

RAW_PATH = os.path.join(DATA_DIR, "raw", "btc_raw.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "processed", "btc_features.csv")

logger = logging.getLogger(__name__)


# the src imports are inside the functions so the scheduler can parse this file quickly
def fetch_task():
    from src.etl_fetch import fetch_data

    fetch_data(RAW_PATH)


def preprocess_task():
    from src.preprocess import preprocess

    preprocess(RAW_PATH, FEATURES_PATH)


def train_task():
    from src.train_forecast import train_and_evaluate

    metrics = train_and_evaluate(FEATURES_PATH, MODEL_DIR)
    logger.info("Metrics: %s", metrics)
    return metrics


def reload_task():
    response = requests.post(f"{API_URL}/reload", headers={"X-API-Key": API_KEY}, timeout=30)
    response.raise_for_status()
    logger.info("API response: %s", response.json())


default_args = {
    "owner": "btc-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="btc_etl_pipeline",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 11, 20),
    catchup=False,
    tags=["btc", "mlops"],
) as dag:
    fetch_data = PythonOperator(task_id="fetch_data", python_callable=fetch_task)
    preprocess = PythonOperator(task_id="preprocess", python_callable=preprocess_task)
    train_evaluate = PythonOperator(task_id="train_evaluate", python_callable=train_task)
    reload_api = PythonOperator(task_id="reload_api", python_callable=reload_task)

    fetch_data >> preprocess >> train_evaluate >> reload_api
