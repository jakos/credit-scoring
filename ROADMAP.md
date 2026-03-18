# Project Roadmap

This roadmap describes planned improvements for the credit-scoring project.

## Vision

Build a reproducible, testable, and deployable credit-scoring workflow that supports experiment tracking, model governance, and reliable inference.

Legend: ✅ done, ⏳ in progress

## ✅ Phase 1 — Core ML Pipeline
Goal:

- Maintain a reliable baseline flow from data loading to model evaluation.

Features:

- ✅ EDA
- ✅ Preprocessing
- ✅ Baseline model
- ✅ MLflow tracking

## ⏳ Phase 2 — MLOps Foundations
Goal:

- Standardize experiment tracking, run metadata, and model artifact lineage.

Features:

- ✅ Airflow DAGs
- Model registry
- FastAPI scoring service

## Phase 3 — Monitoring & Observability
Goal:

- Detect model/data issues early in production-like environments.

Features:

- Prometheus metrics
- Grafana dashboards
- Drift detection DAG

## Phase 4 — Feature Store Integration
Goal:

- Introduce reusable and consistent feature definitions for training and inference.

Features:

- Feast offline store
- Online store for real‑time features

## ⏳ Phase 5 — CI/CD
Goal:

- Automate quality gates, packaging, and release flow.

Features:

- GitHub Actions
- ✅ Automated testing
- Automated model promotion

## Phase 6 — Kubernetes Deployment
Goal:

- Run inference and supporting services on a scalable Kubernetes platform.

Features:

- k3d / kind cluster
- API deployment
- MLflow deployment
- Monitoring stack deployment

Planned deliverables:

- Containerized inference service with health and readiness probes.
- Kubernetes manifests (or Helm chart) for app + dependencies.
- Autoscaling, resource limits, and rollout strategy (for example canary/blue-green).
