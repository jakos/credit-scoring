# 0003: Workflow orchestration

## Status

Accepted for the existing Airflow workflows; future pipeline platforms are planned.

## Context

The platform has scheduled dataset validation, training, and retraining. These
workflows need task dependencies and scheduling around the existing Python CLI.

## Decision

Use the existing Airflow DAGs in `dags/`: daily `credit_scoring_data_quality`,
daily `credit_scoring_training`, and weekly `credit_scoring_retraining`.
Training and retraining validate dataset existence before invoking
`python -m credit_scoring.training.main` through checked subprocess execution.
Data quality checks required columns and the overall missing-value ratio.
Each task constructs settings and logging at execution time, preserving environment
inheritance without validating settings during DAG import.

Keep the local Compose stack for optional scheduler use. Offline tests stub
Airflow and mock subprocesses. Kubeflow/Vertex pipelines are future work, with no
current integration or migration commitment.

## Alternatives

Cron is lighter but provides fewer dependency and task-management facilities.
Airflow fits scheduled Python workflows but adds scheduler/database operations and
does not itself supply model promotion. Kubeflow adds Kubernetes-native ML pipeline
components with cluster overhead. Vertex pipelines provide managed execution with
cloud coupling and resource costs. These are tradeoffs, not recorded benchmark results.

## Consequences

The same training module serves CLI and scheduled tasks. Airflow operators must
provide source, data, dependencies, and environment settings. Local tests cannot
establish live scheduler correctness. Future pipeline work must define artifact
lineage and operational boundaries before replacing or extending these DAGs.

See the [index](README.md), [Airflow usage](../../README.md#optional-airflow), and
[roadmap](../../ROADMAP.md#kubeflowvertex-pipelines).
