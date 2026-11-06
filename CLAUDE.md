# Contributor instructions

This open-source credit-risk ML platform implements baseline training/evaluation,
optional MLflow tracking, typed settings, JSON logging, and Airflow workflows.
Read [README](README.md), [ROADMAP](ROADMAP.md), and [ADRs](docs/adr/README.md).
Registry promotion, scoring services, cloud execution, and monitoring remain planned.

## Python workflow

Use Python 3.14 from `.python-version`, uv, and the committed `uv.lock`.
`uv sync --locked` installs default dev and modeling groups. Hatchling builds the
package from `src/credit_scoring/`; `uv build` builds distributions. Add dependencies
to a fitting group and update the lock without unrelated upgrades. Optional notebooks
use `uv sync --locked --group notebook`.

Run `just check` before finishing. It runs locked Ruff format checks, lint, mypy,
and pytest sequentially. Preserve failure propagation and Windows PowerShell
support. Mypy covers all application modules including wrappers; keep third-party
exceptions narrow and justified. `uv run prek install` installs commit/push hooks
from `prek.toml`.

## Boundaries and tests

Keep application code in `src/credit_scoring/` and mirror ownership in `tests/`.
`data` loads input, `features` prepares it, `training` fits models and coordinates
CLI/tracking, `evaluation` computes metrics, `config` defines settings, and
`logging.py` owns JSON logging. Keep `modeling` wrappers thin and compatible.

Use pytest with offline fixtures and mocks: no network, cloud credentials,
containers, clusters, or external services. Mock optional MLflow, LLM APIs, GCP,
and Kubernetes boundaries. DAG tests stub Airflow and mock subprocesses; they do
not test a running scheduler. New behavior needs a regression test that fails
without the change. Preserve numerical, settings, logging, CLI, and compatibility
assertions. Do not change algorithms, hyperparameters, metrics, datasets, CLI
flags/output, settings precedence, logging contracts, or DAG behavior without
explicit scope.

## Settings and secrets

Never commit secrets, keys, or real credentials. Document environment configuration
in `.env.example`. Construct fresh settings at CLI/task execution time; avoid global
caches and implicit dotenv loading. Explicit CLI values precede environment values
and caller defaults. Library training uses explicit arguments. Compose reads `.env`
for interpolation but does not forward all application settings. Keep dataset records,
full settings, credentials, and tracking URIs out of logs; configure only the
application logger namespace.

## Containers and infrastructure

Use rootless Podman with podman-compose as the primary local engine and preserve
Docker compatibility. Scripts/recipes take `CONTAINER_ENGINE` (default `podman`)
instead of hard-coding an engine. Compose features must work with podman-compose.
Build images with `--format docker` to retain health checks; transfer them to kind
through a local registry or `kind load image-archive`.

Validate GCP infrastructure only with formatting, validation, plans, Helm lint, and
kubeconform; never create cloud resources. `terraform apply` is allowed only for
`infra/terraform/envs/local` against a local kind cluster when that planned directory
exists. The current repository has no infrastructure directory. Do not deploy,
publish, or create resources as a side effect of tests. Keep infrastructure changes
within authorized scope.

## Documentation

Describe the platform in its own terms. Adding a capability updates README and
ROADMAP; checked roadmap items need concrete implementation evidence. Record
architecture decisions using the ADR index template. Keep commands, settings,
paths, and links consistent with implementation.
