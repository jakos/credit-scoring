# Credit Scoring

A credit-risk ML platform with reproducible baseline training, evaluation, optional
MLflow tracking, and scheduled Airflow data-quality and retraining workflows.
Serving, model registry integration, and production monitoring remain future work.

## Roadmap

See the project roadmap: [ROADMAP.md](ROADMAP.md)

## Quick Start

Install Python 3.14, [uv](https://docs.astral.sh/uv/), and
[just](https://github.com/casey/just) before starting.
The `.python-version` file pins Python to 3.14. The committed `uv.lock` keeps
dependency and tool versions reproducible; plain `uv sync` installs both the dev
and modeling groups needed for the offline test suite.

```bash
uv sync
uv run prek install
just check
```

`just check` runs formatting checks, lint, and tests in order using the locked
environment. Installing prek enables automatic whitespace/configuration checks
and Ruff fixes/formatting before commits, plus automatic tests before pushes.
Ruff and prek are managed in the dev group and locked with the project dependencies.

Run hooks manually on all files:

```bash
uv run prek run --all-files
```

Run the pre-push tests manually:

```bash
uv run --locked prek run --all-files --hook-stage pre-push
```

## Run training

```bash
uv sync --group modeling --group dev
uv run train-model --max-iter 300
uv run python -m credit_scoring.training.main --help
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

### Task configuration

Each task constructs fresh typed settings and configures application logging when it
runs; DAG import does not validate the environment. The training DAG uses 300
iterations unless `CREDIT_SCORING_MAX_ITER` overrides it. Retraining uses its separate
`CREDIT_SCORING_RETRAIN_MAX_ITER` setting (default 500), and a nonempty tracking URI
enables MLflow for retraining. Both subprocesses execute
`python -m credit_scoring.training.main` with checked execution and inherit the task
process environment, so dataset overrides reach the actual training input.

Data-quality tasks require `ID`, `LIMIT_BAL`, `AGE`, and
`default.payment.next.month`, and reject an overall missing ratio above the configured
threshold. Offline tests use Airflow module stubs and mocked subprocesses; they do
not install Airflow or test a running scheduler.

### Local usage notes

1. Copy the DAG into your Airflow `dags` folder (or point Airflow to this repository's `dags/` directory).
2. Ensure your Airflow worker/scheduler has access to this repository source and data paths.
3. Trigger `credit_scoring_training` from the Airflow UI.

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

## Configuration and logging

Settings read the process environment at CLI or task execution time, without a global
cache or implicit `.env` loading. Export variables in your shell for local CLI use;
copying `.env.example` alone does not configure the CLI. For example, in a POSIX shell:

```bash
export CREDIT_SCORING_DATASET=data/raw/UCI_Credit_Card.csv
export CREDIT_SCORING_MAX_ITER=400
uv run train-model
```

Explicit CLI flags take precedence over environment values, followed by caller
defaults. `--enable-mlflow` remains the CLI opt-in: a tracking URI alone does not enable
it. `--mlflow-run-name` remains optional and has no environment alias. `--help` exits
before constructing settings, training, or contacting services.

| Environment alias | Default | Meaning |
| --- | --- | --- |
| `APP_ENV` | `development` | Application environment label |
| `LOG_LEVEL` | `INFO` | Case-insensitive `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, or `NOTSET` |
| `CREDIT_SCORING_DATASET` | unset | Repository-relative `data/raw/UCI_Credit_Card.csv`; relative overrides use the process working directory |
| `CREDIT_SCORING_MAX_ITER` | unset | Positive iteration override; CLI/library default 1000, training DAG default 300 |
| `CREDIT_SCORING_RETRAIN_MAX_ITER` | `500` | Positive retraining DAG iteration count, independent of ordinary iterations |
| `CREDIT_SCORING_MAX_MISSING_RATIO` | `0.10` | Inclusive 0..1 data-quality threshold |
| `CREDIT_SCORING_MLFLOW_TRACKING_URI` | unset | Optional tracking URI; retraining enables MLflow when set |
| `CREDIT_SCORING_MLFLOW_EXPERIMENT` | `credit-scoring` | Experiment name; CLI override: `--mlflow-experiment` |

The library `train` takes explicit arguments and does not read environment variables.
Its optional keyword-only `dataset_path` selects input when `df` is absent; an explicit
DataFrame takes precedence. The unset default remains repository-relative after the
package move, including source-mounted Airflow's `/opt/airflow/project/data` path.

Compose reads `.env` for interpolation, including `AIRFLOW_ADMIN_PASSWORD`. The
existing service environment declarations do not forward every application setting;
application settings must be explicitly provided to the task process environment.
All three Airflow services install `pydantic-settings==2.12.0` alongside their existing
modeling requirements. The image tag `apache/airflow:2.10.5` reports Python 3.12.9
(`PYTHON_BASE_IMAGE=python:3.12-slim-bookworm`) in its published linux/amd64 image
configuration ([published tag](https://hub.docker.com/r/apache/airflow/tags?name=2.10.5)),
digest `sha256:b6e72f8ddc684ec9f9133042168afaab3d37a64d0dfe29e4c86727aa6c788c85`.
The [settings release metadata](https://github.com/pydantic/pydantic-settings/blob/v2.12.0/pyproject.toml)
declares Python >=3.10, Python 3.14 support, and Pydantic >=2.7; the
[Airflow 2.10.5 Python 3.12 constraints](https://github.com/apache/airflow/blob/constraints-2.10.5/constraints-3.12.txt)
include Pydantic 2.10.6. These requirements fit the existing image runtime without an
Airflow upgrade. The project lock and tests verify settings under Python 3.14.

CLI status and metric text remain on stdout. Application logs are JSON lines on
stderr with `timestamp` (UTC), `severity`, `logger`, and `event`. Training completion
includes `sample_count` and all six `metrics`; successful tracking emits
`mlflow_logging_completed`. DAG events report dataset/schema validation, missing
ratio and threshold, or completed training/retraining with `max_iter`. Exception
records include formatted `exception` information. Setup affects only the
`credit_scoring` logger namespace, is idempotent, and preserves root and Airflow
handlers. Application events exclude dataset records, complete settings, credentials,
and tracking URIs.

## Project Structure

- `src/credit_scoring/data/loader.py`: CSV loading and repository-relative default path
- `src/credit_scoring/features/preprocess.py`: feature selection, splitting, and preprocessing
- `src/credit_scoring/training/`: algorithm, CLI orchestration, and optional MLflow tracking
- `src/credit_scoring/evaluation/metrics.py`: accuracy, precision, recall, F1, ROC AUC, and average precision (`pr_auc`)
- `src/credit_scoring/config/settings.py`: typed environment configuration
- `src/credit_scoring/logging.py`: standard-library JSON application logging
- `src/credit_scoring/modeling/`: thin compatibility wrappers for existing exports and module imports
- `dags/`: Airflow DAGs
- `tests/`: test suite
- `pyproject.toml`: project metadata and tooling config


The `train-model` console script targets `credit_scoring.training.main:main`.
Existing `credit_scoring.modeling` imports and
`python -m credit_scoring.modeling.main` continue to delegate to the new packages.
Tests mirror package ownership and cover compatibility, numerical regression,
configuration, logging, and DAG behavior without external services.
