# Credit Scoring

Starter Python project template for credit scoring experiments and services.

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

## Project Structure

- `src/credit_scoring/`: application package
- `tests/`: test suite
- `pyproject.toml`: project metadata and tooling config
