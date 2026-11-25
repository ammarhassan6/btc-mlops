# BTC Forecast MLOps

An end-to-end MLOps pipeline that forecasts the next 7 daily closing prices of Bitcoin (BTC-USD).

- **Data:** daily prices from Yahoo Finance (`yfinance`), with the Binance public API as a fallback
- **Model:** ARIMA(5,1,0) on the closing price, evaluated on the last 7 days (MAE / RMSE / MAPE)
- **Serving:** Flask REST API run by Gunicorn, packaged in a multi-stage Docker image
- **CI/CD:** GitHub Actions lints and tests every PR, and pushes the image to Docker Hub on every merge to `main`
- **Orchestration:** Airflow DAG (daily) and a Kubeflow pipeline
- **Deployment:** Kubernetes (Minikube) with ConfigMap/Secret, PVC, probes, NodePort/ClusterIP Services, HPA and NGINX Ingress

Docker image: [`ammarhassan1/btc-forecast-api`](https://hub.docker.com/r/ammarhassan1/btc-forecast-api)

## Architecture

```
yfinance / Binance
      |
      v
etl_fetch.py -> raw CSV -> preprocess.py -> features CSV -> train_forecast.py -> model.pkl + metrics.json
      (run by: Airflow DAG daily | Kubeflow pipeline | Kubernetes train Job)          |
                                                                                       v
GitHub push -> Actions: flake8, black, pytest -> docker build -> Docker Hub -> K8s Deployment btc-api
                                                                                       ^
client -> Ingress btc-forecast.local -> Service btc-api-service:80 -> pods :5000 (Gunicorn + Flask)
                                         HPA btc-api-hpa: 2-8 pods at 60% CPU
```

## Project structure

```
app/            Flask API (app.py), Dockerfile, runtime requirements
src/            etl_fetch.py, preprocess.py, train_forecast.py (shared by all pipelines)
tests/          pytest suite for the API (no internet needed)
dags/           Airflow DAG btc_etl_pipeline
kubeflow/       KFP v2 pipeline and its compiled YAML
k8s/            ConfigMap/Secret, PVC, train Job, Deployment, Services, HPA, Ingress
loadtest/       Locust load test
docker-compose.yml   Postgres + Airflow + API for local orchestration
.github/workflows/   CI/CD pipeline
```

## Requirements

- Python 3.10
- Docker Desktop
- kubectl and Minikube (for the Kubernetes part)

The commands below are for Windows PowerShell. On macOS/Linux use `source .venv/bin/activate` and `curl` instead of `curl.exe`.

## 1. Run locally

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app/requirements.txt -r requirements-dev.txt

python -m src.etl_fetch --out data/raw/btc_raw.csv
python -m src.preprocess --input data/raw/btc_raw.csv --out data/processed/btc_features.csv
python -m src.train_forecast --input data/processed/btc_features.csv --model-dir models

python -m app.app
```

Run the tests and checks (the same ones CI runs):

```powershell
flake8 .
black --check .
pytest -v
```

## 2. API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, uptime, model version, whether a model is loaded |
| POST | `/predict` | Forecast. Body `{}`, `{"forecast_days": 7}` or `{"history": [...], "forecast_days": 7}` |
| GET | `/metrics` | Evaluation metrics of the current model (`metrics.json`) |
| POST | `/reload` | Reload the model from disk. Needs header `X-API-Key` |

Validation: `forecast_days` must be an integer from 1 to 30, and `history` must be a list of at least 30 positive numbers. Invalid input returns `400`, a missing model `503`, a wrong API key `401`.

Example:

```powershell
curl.exe http://localhost:5000/health
'{"forecast_days": 7}' | curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "@-"
```

The API reloads the model automatically whenever `model.pkl` changes on disk, so all Gunicorn workers and all pods pick up a newly trained model.

## 3. Docker

```powershell
docker build -f app/Dockerfile -t btc-forecast-api:local .
docker run --rm -p 5000:5000 -e API_KEY=demo -v ${PWD}/models:/app/models btc-forecast-api:local
```

The build context is the repo root because the image also contains `src/`. That way the same image serves the API and runs the training steps in the Kubernetes Job and the Kubeflow pipeline. The image is multi-stage, based on `python:3.10-slim`, and runs as the non-root user `appuser`.

## 4. Airflow (docker-compose)

```powershell
docker compose up -d --build
docker compose ps
```

- Airflow UI: http://localhost:8080 (user `admin`, password `admin`)
- API: http://localhost:5000

The DAG `btc_etl_pipeline` runs daily: `fetch_data -> preprocess -> train_evaluate -> reload_api`. Airflow and the API share the `./models` folder. After training, the last task calls `/reload`, so the API serves the new model straight away.

```powershell
docker compose exec airflow airflow dags unpause btc_etl_pipeline
docker compose exec airflow airflow dags trigger btc_etl_pipeline
curl.exe http://localhost:5000/metrics
docker compose down
```

## 5. Kubeflow

The pipeline in `kubeflow/btc_kfp_pipeline.py` has three container components (fetch, preprocess, train), all using the Docker Hub image. Data is passed between steps as typed artifacts. Caching is disabled for the fetch step so that every run downloads fresh prices.

kfp pins older protobuf/urllib3 versions, so it is installed in its own virtual environment:

```powershell
py -3.10 -m venv $HOME\kfp-venv
& $HOME\kfp-venv\Scripts\python.exe -m pip install -r kubeflow/requirements.txt
& $HOME\kfp-venv\Scripts\python.exe kubeflow/btc_kfp_pipeline.py
```

This writes `kubeflow/btc_kfp_pipeline.yaml`, which can be uploaded in the Kubeflow Pipelines UI.

## 6. Kubernetes (Minikube)

```powershell
minikube start --driver=docker --cpus=4 --memory=6144
minikube addons enable metrics-server
minikube addons enable ingress

kubectl apply -f k8s/configmap-secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/train-job.yaml
kubectl wait --for=condition=complete job/btc-train --timeout=600s
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
kubectl get pods,svc,pvc,hpa,ingress
```

| Resource | Purpose |
|---|---|
| `btc-config` / `btc-secret` | Settings and the API key as environment variables |
| `btc-models-pvc` | 1Gi volume at `/app/models` holding the model and data |
| Job `btc-train` | Runs fetch, preprocess and train once to put a model on the PVC |
| Deployment `btc-api` | API pods with liveness/readiness probes on `/health`, CPU 100m-500m, memory 128Mi-512Mi |
| `btc-api-service` / `btc-api-nodeport` | ClusterIP (used by the Ingress) and NodePort 30080 |
| HPA `btc-api-hpa` | 2 to 8 pods, target 60% average CPU |
| Ingress `btc-api-ingress` | Host `btc-forecast.local` -> `btc-api-service:80` |

To reach the Ingress on Windows (Docker driver):

1. Run `minikube tunnel` in an Administrator terminal and keep it open.
2. Add `127.0.0.1 btc-forecast.local` to `C:\Windows\System32\drivers\etc\hosts`.
3. Run `curl.exe http://btc-forecast.local/health`.

Retrain on the cluster:

```powershell
kubectl delete job btc-train
kubectl apply -f k8s/train-job.yaml
```

## 7. Load test and self-healing

```powershell
pip install -r loadtest/requirements.txt
locust -f loadtest/locustfile.py --headless -u 5 -r 1 -t 3m --host http://btc-forecast.local --csv report/loadtest
kubectl get hpa btc-api-hpa --watch
```

Each simulated user mostly sends `/predict` with a 200-value price history. That makes the API refit ARIMA on every request, which puts real CPU load on the pods.

Self-healing: delete a pod and the Deployment replaces it immediately.

```powershell
kubectl get pods -l app=btc-api -w
kubectl delete pod <pod-name>
```

## Results

| | |
|---|---|
| Model evaluation (last 7 days) | MAE 3097, RMSE 3526, MAPE 5.11% |
| Load test, 5 users, 3 min | 327 requests, 0.9% failures, `/predict` median 1.0 s, `/health` avg 20 ms |
| Stress test, 100 users | HPA scaled 2 -> 8 pods (CPU 481% of target); the single 4-CPU node became the limit |
| Self-healing | A deleted pod was replaced and `Ready` in about 11 s, with no downtime through the Ingress |

## CI/CD

`.github/workflows/ci_cd.yml`

- **Pull request to `main`:** `flake8`, `black --check`, `pytest`
- **Push to `main`:** the same checks, then build the image and push it to Docker Hub with tags `latest` and the commit SHA

Docker Hub credentials are stored as GitHub Secrets (`DOCKER_USERNAME`, `DOCKER_PASSWORD`).

## Configuration

| Variable | Default | Used for |
|---|---|---|
| `MODEL_DIR` | `models` (`/app/models` in containers) | Where `model.pkl` and `metrics.json` are stored |
| `DATA_DIR` | `data` | Raw and processed CSV files |
| `MAPE_THRESHOLD` | `10` | A new model is saved only if its MAPE is at or below this value |
| `API_KEY` | none | Required for `/reload` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_URL` | `http://api:5000` | Used by the Airflow DAG to call `/reload` |
