# Credit Scoring

Starter Python project template for credit scoring experiments and services.

## Roadmap

See the project roadmap: [ROADMAP.md](ROADMAP.md)

## Quick Start

```bash
uv sync --group dev
py -3.14 -m uv run pre-commit install
uv run pytest
```

Run hooks manually on all files:

```bash
py -3.14 -m uv run pre-commit run --all-files
```

## Run training

```bash
uv sync --group modeling --group dev
uv run train-model --max-iter 300
uv run train-model --max-iter 300 --enable-mlflow --mlflow-experiment credit-scoring
```

## MLflow

MLflow is an experiment tracking tool for machine learning projects.

Use MLflow in this project to:

- track training parameters (for example `max_iter`)
- log evaluation metrics (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`)
- store trained model artifacts for reproducibility and comparison between runs

### How to use it

1. Install modeling dependencies (includes MLflow):

```bash
uv sync --group modeling --group dev
```

2. Start the MLflow UI in a separate terminal (optional for local tracking):

```bash
uv run mlflow ui
```

Then open http://127.0.0.1:5000 in your browser.

3. Run training with MLflow enabled:

```bash
uv run train-model --max-iter 300 --enable-mlflow --mlflow-experiment credit-scoring
```

4. Optionally set an explicit tracking server URI:

```bash
uv run train-model --max-iter 300 --enable-mlflow --mlflow-tracking-uri http://127.0.0.1:5000 --mlflow-experiment credit-scoring --mlflow-run-name baseline-lr
```

### Run MLflow with Docker Compose

You can run a local MLflow tracking server with Docker Compose using `docker-compose.yml`.
The compose service uses an external MLflow image (`ghcr.io/mlflow/mlflow:latest`).
Data is persisted in a named volume (`mlflow_data`).

1. Start MLflow:

```bash
docker compose up -d mlflow
```

2. Open MLflow UI:

Open http://127.0.0.1:5000 in your browser.

3. Run training against the Docker MLflow server:

```bash
uv run train-model --max-iter 300 --enable-mlflow --mlflow-tracking-uri http://127.0.0.1:5000 --mlflow-experiment credit-scoring
```

4. Stop MLflow:

```bash
docker compose down
```

To remove persisted MLflow data as well:

```bash
docker compose down -v
```

## Airflow DAGs

Airflow DAGs available in this repository:

- `dags/credit_scoring_training_dag.py` (`credit_scoring_training`): validates input data and runs training.
- `dags/credit_scoring_data_quality_dag.py` (`credit_scoring_data_quality`): validates schema and missing-data ratio.
- `dags/credit_scoring_retraining_dag.py` (`credit_scoring_retraining`): scheduled model retraining.

### Optional environment variables

- `CREDIT_SCORING_DATASET`: override dataset path (default: `data/raw/UCI_Credit_Card.csv`)
- `CREDIT_SCORING_MAX_ITER`: override training iterations (default: `300`)

### Local usage notes

1. Copy the DAG into your Airflow `dags` folder (or point Airflow to this repository's `dags/` directory).
2. Ensure your Airflow worker/scheduler has access to this repository source and data paths.
3. Trigger `credit_scoring_training_example` from the Airflow UI.

### Run Airflow with Docker Compose

This repository includes a basic Airflow stack (`postgres`, `airflow-init`, `airflow-webserver`, `airflow-scheduler`) in `docker-compose.yml`.

Before starting Airflow, create a local `.env` file and set credentials (minimum: `AIRFLOW_ADMIN_PASSWORD`):

```bash
cp .env.example .env
```

Then edit `.env` and set a strong password, for example:

```env
AIRFLOW_ADMIN_PASSWORD=use-a-strong-unique-password
```

1. Start Airflow:

```bash
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

If you use Podman Compose:

```bash
podman compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

2. Open Airflow UI:

Open http://127.0.0.1:8080 in your browser.

3. Log in with credentials:

- username: `admin`
- password: `AIRFLOW_ADMIN_PASSWORD` from `.env`

4. Enable and trigger DAGs:

- `credit_scoring_data_quality`
- `credit_scoring_training`
- `credit_scoring_retraining`

5. Stop Airflow:

```bash
docker compose down
```

With Podman Compose:

```bash
podman compose down
```

## Project Structure

- `src/credit_scoring/`: application package
- `dags/`: Airflow DAGs
- `tests/`: test suite
- `pyproject.toml`: project metadata and tooling config
