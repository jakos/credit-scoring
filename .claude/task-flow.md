# task-flow configuration

## Profiles

- extra: python, docs
- path: deploy/** -> config
- path: infra/** -> config

## Verification commands

### python
```
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Architecture rules

- Python package lives in `src/credit_scoring/`; tests mirror it under `tests/`.
- Tests run offline: no network calls, no cloud credentials, no running containers or clusters required. External services (MLflow server, LLM APIs, GCP, Kubernetes) are mocked or optional.
- Infrastructure code for GCP is validated only (`terraform fmt/validate/plan`, `helm lint`, `kubeconform`); never create cloud resources. `terraform apply` is allowed only for `infra/terraform/envs/local` against the local kind cluster.
- Never commit secrets, keys or real credentials; configuration comes from environment variables with `.env.example` documenting them.
- Docs and code describe the project in its own terms.
- Every task that adds a capability updates README.md and ROADMAP.md accordingly.

## Git

- base: main
- branch: task/<id>
- worktree: yes

## Finish

- commit: yes
- push: ask
- pr: ask
- remove-worktree: ask
