# Project Roadmap

Build a reproducible credit-risk ML platform spanning training, governed model
lifecycle, and reliable inference. See the [implemented overview](README.md) and
[architecture records](docs/adr/README.md). Checked items describe repository
implementation; optional services still require separate local provisioning.

## Tooling and foundation

- [x] Python 3.14, uv lock, default dev/modeling groups, and Hatchling (`pyproject.toml`, `uv.lock`).
- [x] Ruff, mypy with pandas stubs, pytest, and sequential locked `just check`.
- [x] Prek commit checks and pre-push tests (`prek.toml`).
- [x] Push/pull-request CI using the shared gate (`.github/workflows/ci.yml`).
- [x] Focused packages, legacy wrappers, typed execution-time settings, and JSON logging (`src/credit_scoring/`).
- [x] Offline numerical, settings, logging, compatibility, and DAG tests (`tests/`).
- [x] Initial architecture records and contributor instructions (`docs/adr/`, `CLAUDE.md`).

See [package layout](docs/adr/0001-package-layout.md) and
[tooling](docs/adr/0002-python-tooling.md).

## Model lifecycle and registry

- [x] Exploratory notebook (`notebooks/01_eda.ipynb`).
- [x] CSV loading, numeric preprocessing, stratified split, logistic regression, and six metrics (`src/credit_scoring/`).
- [x] Optional MLflow parameter/metric/model-artifact tracking (`src/credit_scoring/training/tracking.py`).
- [x] Scheduled validation, training, and retraining (`dags/`); see [orchestration](docs/adr/0003-workflow-orchestration.md).
- [x] Model registry registration, exact version aliases, artifact lineage, and model cards (`src/credit_scoring/registry/`).
- [x] Deterministic logistic regression/HistGradientBoosting comparison, KS/Gini/PSI, pure promotion gates, and automated challenger/champion promotion; see [registry decision](docs/adr/0004-model-registry-promotion.md).
- [ ] Automated rollback and additional approval workflows.

## Scoring service

- [ ] FastAPI prediction endpoint with validated schemas.
- [ ] Governed model loading and consistent inference preprocessing.
- [ ] Health/readiness endpoints and offline service tests.

## Containers

- [x] Local Compose MLflow, Postgres, and Airflow services (`docker-compose.yml`).
- [x] Source/data mounts for existing Airflow workflows.
- [ ] Reproducible training and scoring application images.
- [ ] Engine-selectable recipes, rootless Podman validation, and Docker compatibility checks.
- [ ] Image hardening and pinned production service images.

## Local GCP emulation

- [ ] Local substitutes for required GCP APIs and storage.
- [ ] Offline integration fixtures and documented emulator limitations.

## Spark features

- [ ] Batch feature transformations with Spark.
- [ ] Feature parity tests and reproducible input/output contracts.

## Kubernetes

- [ ] Local kind cluster and registry/image-archive loading.
- [ ] Scoring, MLflow, and monitoring manifests or Helm charts with limits and probes.
- [ ] Autoscaling and rollout/rollback strategy.

## GCP infrastructure

- [ ] Terraform storage, identity, networking, and compute definitions.
- [ ] Validation/plan checks, documented costs, and least-privilege access.
- [ ] Authorized cloud deployment workflow.

## Kubeflow/Vertex pipelines

- [ ] Portable training/evaluation artifacts and pipeline components.
- [ ] Evaluate Kubeflow and Vertex execution boundaries and operational tradeoffs.
- [ ] Integrate a selected pipeline platform with registry gates.

Existing Airflow DAGs remain the implemented scheduler; these platforms are future
extensions described in [ADR 0003](docs/adr/0003-workflow-orchestration.md).

## Monitoring and drift

- [ ] Prometheus metrics and Grafana dashboards.
- [ ] Data/prediction drift checks, alerting, and a drift workflow.
- [ ] Model quality monitoring and governed retraining triggers.

## Fraud detection

- [ ] Fraud data contracts, baseline detection, and evaluation.
- [ ] Fraud scoring and monitoring integrated with platform workflows.

## AI analyst assistant

- [ ] Analyst questions grounded in approved metadata and metrics.
- [ ] Access controls, traceable responses, and offline evaluation with mocked APIs.

## Feast feature-store integration (later, optional)

- [ ] Reusable feature definitions and an offline store.
- [ ] Online retrieval and training/serving consistency tests.
