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
FastAPI, Pydantic, Prometheus client, and base Uvicorn; modeling supplies MLflow, pandas, NumPy, and
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
version once. Registry construction and loading perform read-only registry operations:
no experiment selection or creation, model logging, tag writes, or alias writes.
Serving credentials need read access to the registry and artifacts. Training selects
its experiment only when logging a model. No polling, fallback, or automatic refresh occurs.
Restart the service to adopt a new champion or retry a failed load.

Promotion assigns the champion alias separately from recording its final card and
tags. A later metadata write failure reports an explicit partial failure: assignment
succeeded and metadata recording failed, with the original cause preserved. The
alias remains assigned and metadata may be partially updated; inspect and repair
it before retrying. There is no automatic rollback. See the
[registry lifecycle](../README.md#registry-lifecycle).

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
| `CREDIT_SCORING_MLFLOW_EXPERIMENT` | `credit-scoring` | Training experiment; not selected during serving |
| `CREDIT_SCORING_SERVING_API_KEY` | unset | Optional nonblank secret without leading or trailing whitespace; unset disables authentication |
| `LOG_LEVEL` | `INFO` | Inherited JSON application logging level |

Thresholds must be finite and satisfy `0 < medium < high < 1`. Invalid settings
stop startup. CLI errors identify the invalid setting names without their values
or a complete settings dump. An existing local directory whose artifact is broken
instead leaves the application live but unready. Neither source requires registry lifecycle
enablement through `CREDIT_SCORING_REGISTRY_ENABLED`.

## Applicant contract

Every applicant contains a `features` object. Supply every input feature required
by the loaded artifact, with no additional names. Feature names come from fitted
`feature_names_in_` metadata; incompatible or absent metadata prevents readiness.
`ID` and `default.payment.next.month` are never serving inputs. The default UCI
training pipeline uses the 23 feature names in the example below. A model trained
on a different dataset can require a different contract.

Values must be finite JSON numbers in the inclusive range **-1e12 through 1e12**
or explicit `null`. The same bound applies to integers and floats and is published
in OpenAPI. Values outside the bound return HTTP 422 before inference in both
endpoints; numeric strings and booleans are rejected. `null` is accepted only for features covered by the
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
{"probability_of_default": 0.31, "risk_band": "medium", "explanation": {"method":"linear_log_odds","baseline_log_odds":-1.0,"top_features":[{"feature":"AGE","contribution":0.2}],"remaining_contribution":0.0}}
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
{"scores": [{"probability_of_default": 0.31, "risk_band": "medium", "explanation": {"method":"linear_log_odds","baseline_log_odds":-1.0,"top_features":[{"feature":"AGE","contribution":0.2}],"remaining_contribution":0.0}}]}
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

Python's configured 4300-digit integer conversion limit is unchanged. A raw JSON
body containing an integer with more than 4300 digits fails parsing with HTTP 400,
before numeric validation. Smaller parsed integers outside the feature bound return
422. This applies to both scoring endpoints.

Unknown feature keys use their zero-based insertion positions beneath `features`,
for example `["body", "features", 23]` or
`["body", "applicants", 1, "features", 23]`. Each unknown key has a distinct position;
its name and value are omitted, including when its value fails schema validation.

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

The application logs `serving_prediction_failed` with `exception_type`,
`cause_type`, `message`, and `cause_message`. Startup failures use the same fields
on `serving_model_load_failed`. Only fixed application diagnostic messages are
retained; arbitrary dependency exception messages become `Failure details redacted`.
Original exception causes remain available internally, but logs omit tracebacks.
Missing champions report `No champion version is available`; incompatible artifacts
report the failed model contract; registry access failures report
`Registry lookup or model loading failed` with the underlying exception type.
Check champion assignment, artifact compatibility, or registry access respectively,
then restart after repairing a load failure. Neither API errors nor application
events include applicant records, registry credentials, or filesystem paths.

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

## Native explanations

Each successful single score and each batch member includes a required `explanation`.
`method` is `linear_log_odds`; `baseline_log_odds` is the fitted intercept oriented
toward default label `1`. Each `top_features` entry contains an input `feature` name
and its signed `contribution`: the fitted imputed and scaled value multiplied by
its oriented logistic coefficient. Positive values increase the default log odds.

Features are ranked by descending absolute contribution, then feature name for
ties. At most five are returned; smaller models return all features.
`remaining_contribution` is the sum of omitted contributions. Baseline plus listed
contributions plus remaining contribution reconstructs the oriented decision within
floating-point tolerance. These are log-odds contributions, not probability changes
or causal effects. They do not establish lending eligibility or fairness.

Only the fitted numeric median-imputer/standard-scaler and binary logistic-regression
pipeline is supported. Feature mapping must be established from fitted metadata;
unsupported models (including HistGradientBoosting champions) and ambiguous mappings
leave `/healthz` live at 200, `/readyz` at 503, and scoring at 503. Runtime invalid
explanations return sanitized 500 errors for the entire request, with no partial batch.

## Authentication and correlation

Authentication is disabled when `CREDIT_SCORING_SERVING_API_KEY` is unset. Explicitly
empty, whitespace-only, or leading/trailing whitespace-padded values are invalid;
keys are never trimmed. Valid Unicode keys are preserved. Set a secret in the process
environment before starting; it takes effect after restart. For example, in Bash:

```bash
read -r -s CREDIT_SCORING_SERVING_API_KEY
export CREDIT_SCORING_SERVING_API_KEY
uv run --locked serve
```

In another shell with the same secret environment variable, the saved request files
from the examples above can be sent using:

```bash
curl http://127.0.0.1:8000/v1/score -H 'Content-Type: application/json' -H "X-API-Key: ${CREDIT_SCORING_SERVING_API_KEY}" --data-binary @applicant.json
curl http://127.0.0.1:8000/v1/score/batch -H 'Content-Type: application/json' -H "X-API-Key: ${CREDIT_SCORING_SERVING_API_KEY}" --data-binary @batch.json
```

Exactly one matching `X-API-Key` header is required. Missing, incorrect, or duplicate
headers return 401 `{"detail":"Unauthorized"}` before scoring. Comparison uses
constant-time UTF-8 bytes. Exactly `/healthz`, `/readyz`, and `/metrics` are public;
all other application paths, including `/docs`, `/redoc`, and `/openapi.json`, require
the key when enabled. Swagger's page request must therefore also carry the header.
OpenAPI declares the header security scheme and scoring requirements only when enabled.

Every response carries `X-Request-ID`. An incoming id is preserved only when it has
1 to 128 ASCII letters, digits, dots, underscores, or hyphens. Missing, duplicate, or
invalid ids are replaced with a UUID. Correlation stays local to each request.

One JSON `serving_access` event is emitted per request, including metrics
scrapes and rejected requests. Alongside the inherited `timestamp`, `severity`,
`logger`, and `event`, fields are `request_id`, `method`, `route`, `status`, and
`duration_seconds` (nonnegative monotonic elapsed seconds). Methods use a bounded
HTTP vocabulary with `OTHER` as fallback; routes use matched templates or `unmatched`.
Logs exclude raw URLs, queries, bodies, headers, credentials, and applicant data.
Uvicorn access logging is disabled to avoid a duplicate unstructured stream.
Rejected request ids are absent from the entire access event. An unexpected failure
before response start produces a sanitized 500. After response start, the middleware
logs `serving_request_failed` at error severity and re-raises to the ASGI caller,
without sending another response. Access logs and request metrics retain the status
already sent; incomplete responses never add successful probability observations.

## Prometheus metrics

Public `GET /metrics` returns the Prometheus client's exposition and content type.
Each application instance owns a separate registry. Aggregation is process-local;
multiple workers require external aggregation and this service does not configure
multiprocess metrics or a Prometheus deployment.

| Metric | Labels | Meaning |
| --- | --- | --- |
| `credit_scoring_http_requests_total` | `method`, `route`, `status` | Completed requests, including errors |
| `credit_scoring_http_request_duration_seconds` | `method`, `route` | Request duration histogram |
| `credit_scoring_probability_of_default` | none | Successful applicant probability histogram |

Latency buckets in seconds are `0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10`.
Probability buckets are `0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1`.
Both include the client's positive-infinity bucket and standard histogram `_bucket`,
`_count`, and `_sum` samples. A successful batch of N applicants adds N probability
observations; failures add none. `/metrics` scrapes do not observe themselves in HTTP
metrics, but do produce access events. Label values never contain request ids, raw
paths, credentials, or applicant values. Unknown methods use `OTHER`, and unmatched
paths use `unmatched`, keeping cardinality bounded.
