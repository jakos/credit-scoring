# Serving credit-risk models

The scoring service runs a complete stored sklearn pipeline, including inherited
numeric preprocessing. It provides single and batch class-1 default probabilities,
risk bands, liveness, and readiness. See the [project overview](../README.md) and
[roadmap](../ROADMAP.md) for the surrounding training and registry workflows.

## Install and select a model

Run from the repository root with Python 3.14:

```bash
uv sync --locked
uv run --locked serve --help
uv run --locked python -m credit_scoring.serving.main --help
```

The default groups are `dev`, `modeling`, and `serving`. The serving group adds
FastAPI, Pydantic, and base Uvicorn; modeling supplies MLflow, pandas, NumPy, and
sklearn. HTTPX in dev supports offline TestClient tests.

Registry mode uses the shared tracking settings and registered model name. The
registry must already contain a compatible `champion`, for example from the
[training lifecycle](../README.md#registry-lifecycle). In a POSIX shell:

```bash
export CREDIT_SCORING_SERVING_MODEL_SOURCE=registry
export CREDIT_SCORING_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export CREDIT_SCORING_REGISTERED_MODEL_NAME=credit-scoring
uv run --locked serve --host 127.0.0.1 --port 8000
```

At startup the adapter resolves `champion` to an exact version and loads that
version once. No alias mutation, polling, fallback, or automatic refresh occurs.
Restart the service to adopt a new champion or retry a failed load.

Local mode loads a trusted **MLflow sklearn model directory**, containing `MLmodel`
and its serialized pipeline. A raw pickle or joblib file is not a supported source.
To prepare one using the existing training CLI and the locked MLflow stack, run
these commands from the repository root in a POSIX shell. The default CSV must
be present. Use a fresh `mlflow-data/local-model` destination:

```bash
mkdir -p mlflow-data
uv run --locked train-model --max-iter 300 --enable-mlflow --mlflow-tracking-uri sqlite:///mlflow-data/tracking.db --mlflow-experiment serving-local
uv run --locked python - <<'PY'
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow-data/tracking.db")
experiment = mlflow.get_experiment_by_name("serving-local")
models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id], output_format="list"
)
latest = max(models, key=lambda model: model.creation_timestamp)
pipeline = mlflow.sklearn.load_model(f"models:/{latest.model_id}")
mlflow.sklearn.save_model(pipeline, "mlflow-data/local-model", pip_requirements=[])
PY
export CREDIT_SCORING_SERVING_MODEL_SOURCE=local
export CREDIT_SCORING_SERVING_MODEL_PATH=mlflow-data/local-model
uv run --locked serve
```

This chooses the most recently logged model in the dedicated `serving-local`
experiment. Keep generated model directories, tracking databases, and artifact
stores out of commits. Configure only trusted artifacts: loading sklearn models
executes serialized code. Local mode never contacts the registry, and registry
mode never falls back to a local model.

## Configuration

Settings read the process environment when the application is created. CLI
`--host` and `--port` override environment values, which override defaults.
There is no implicit `.env` loading. Copying `.env.example` alone does not set
CLI configuration. `--help` exits before settings construction or model loading.
The application factory accepts explicit settings and a loader for tests.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `CREDIT_SCORING_SERVING_MODEL_SOURCE` | `registry` | Exactly `registry` or `local` |
| `CREDIT_SCORING_SERVING_MODEL_PATH` | unset | Existing MLflow sklearn directory required in local mode; relative to the working directory |
| `CREDIT_SCORING_SERVING_HOST` | `127.0.0.1` | Bind address; CLI override `--host` |
| `CREDIT_SCORING_SERVING_PORT` | `8000` | Port 1–65535; CLI override `--port` |
| `CREDIT_SCORING_RISK_MEDIUM_THRESHOLD` | `0.20` | Inclusive lower boundary of medium risk |
| `CREDIT_SCORING_RISK_HIGH_THRESHOLD` | `0.50` | Inclusive lower boundary of high risk |
| `CREDIT_SCORING_MLFLOW_TRACKING_URI` | unset | Nonempty URI required in registry mode |
| `CREDIT_SCORING_REGISTERED_MODEL_NAME` | `credit-scoring` | Shared registry model name; must not be blank |
| `CREDIT_SCORING_MLFLOW_EXPERIMENT` | `credit-scoring` | Shared adapter experiment setting |
| `LOG_LEVEL` | `INFO` | Inherited JSON application logging level |

Thresholds must be finite and satisfy `0 < medium < high < 1`. Invalid settings
stop startup. An existing local directory whose artifact is broken instead leaves
the application live but unready. Neither source requires registry lifecycle
enablement through `CREDIT_SCORING_REGISTRY_ENABLED`.

## Applicant contract

Every applicant contains a `features` object. Supply every input feature required
by the loaded artifact, with no additional names. Feature names come from fitted
`feature_names_in_` metadata; incompatible or absent metadata prevents readiness.
`ID` and `default.payment.next.month` are never serving inputs. The default UCI
training pipeline uses the 23 feature names in the example below. A model trained
on a different dataset can require a different contract.

Values must be finite JSON numbers or explicit `null`; numeric strings and
booleans are rejected. `null` is accepted only for features covered by the
inherited fitted median imputer with a finite training statistic. Missing keys
are always errors, even when explicit null is supported. Extra envelope fields
are forbidden. JSON key order does not matter: scoring constructs a DataFrame
in trained feature order and runs the complete pipeline without fitting it.

## Single scoring

`POST /v1/score` accepts this complete synthetic applicant. The same payload is
published in OpenAPI and exercised by the offline fixture tests:

<!-- single-request -->
```json
{
  "features": {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 10000,
    "BILL_AMT2": 9000,
    "BILL_AMT3": 8000,
    "BILL_AMT4": 7000,
    "BILL_AMT5": 6000,
    "BILL_AMT6": 5000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000
  }
}
```

Save that JSON as `applicant.json` in your working directory, then submit it:

```bash
curl -X POST http://127.0.0.1:8000/v1/score -H 'Content-Type: application/json' --data-binary @applicant.json
```

A successful response is HTTP 200. This output is illustrative; the probability
depends on the loaded model:

```json
{"probability_of_default": 0.31, "risk_band": "medium"}
```

`probability_of_default` is the `predict_proba` column for class label `1`
(default next month), located through `classes_`, rather than an assumed column
index. The artifact must have two distinct numeric classes including `1`.
Malformed output shape, non-finite output, or any probability outside `[0, 1]`
produces a prediction error; invalid output is never clipped.

| Probability | Default band |
| --- | --- |
| `0 <= p < 0.20` | `low` |
| `0.20 <= p < 0.50` | `medium` |
| `0.50 <= p <= 1` | `high` |

Risk bands partition model output. They establish neither probability calibration
nor lending eligibility, fairness, or regulatory approval.

## Batch scoring

`POST /v1/score/batch` accepts 1–1000 applicants. Here is a complete one-applicant
batch; add more applicant objects to the array as needed:

<!-- batch-request -->
```json
{
  "applicants": [
    {
      "features": {
        "LIMIT_BAL": 50000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 10000,
        "BILL_AMT2": 9000,
        "BILL_AMT3": 8000,
        "BILL_AMT4": 7000,
        "BILL_AMT5": 6000,
        "BILL_AMT6": 5000,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 1000,
        "PAY_AMT3": 1000,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1000
      }
    }
  ]
}
```

Save that JSON as `batch.json`, then submit it:

```bash
curl -X POST http://127.0.0.1:8000/v1/score/batch -H 'Content-Type: application/json' --data-binary @batch.json
```

The HTTP 200 response preserves request order (illustrative probability):

```json
{"scores": [{"probability_of_default": 0.31, "risk_band": "medium"}]}
```

One vectorized prediction call handles the complete batch. Any invalid applicant
rejects the whole request with HTTP 422; no partial results are returned.

## Errors and availability

| Endpoint | Ready | Unready |
| --- | --- | --- |
| `GET /healthz` | 200 `{"status":"live"}` | 200 `{"status":"live"}` |
| `GET /readyz` | 200 `{"status":"ready"}` | 503 `{"status":"unready"}` |
| Both scoring endpoints | 200 on valid input | 503 `{"detail":"Model is not ready"}` |

Loading occurs once during lifespan startup. Loading or metadata validation
failure emits the sanitized `serving_model_load_failed` application event and
leaves the process live but unready. Repair the source and restart. A successful
startup holds its snapshot until shutdown, when per-app model state is cleared.

Request validation returns HTTP 422 with field locations and batch indexes,
without echoing raw values. For example, if the second batch member is missing
`AGE`, a response includes:

```json
{"detail":[{"loc":["body","applicants",1,"features","AGE"],"msg":"Required model feature is missing","type":"missing"}]}
```

An unknown feature is reported at the `features` object with type
`extra_forbidden`; unsupported null uses `null_not_supported`. Other schema
errors use Pydantic error types. Error details omit raw inputs and validation
context. Invalid model predictions return HTTP 500:

```json
{"detail":"Model prediction failed"}
```

The application logs `serving_prediction_failed` without exception text or
tracebacks. Neither API errors nor application events include applicant records,
registry credentials, or filesystem paths.

## OpenAPI and offline verification

Open http://127.0.0.1:8000/docs for Swagger UI and
http://127.0.0.1:8000/openapi.json for the schema. Both scoring operations declare
request and response models and 422, 500, and 503 errors; readiness declares 503.
The applicant example also works against the synthetic fixture model.

```bash
uv run pytest tests/serving tests/config
just check
uv build
```

Tests train a deterministic small pipeline, save and load an MLflow sklearn
model directory in a temporary directory, and use fake registry calls. They need
no external server, credentials, network, or containers. CLI tests patch Uvicorn
rather than starting a listening server. The lock retains the inherited MLflow
3.10.1/sklearn 1.8.0 stack; the service is verified on Python 3.14.
