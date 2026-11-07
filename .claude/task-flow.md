# task-flow configuration

## Profiles

- extra: python, docs
- path: deploy/** -> config
- path: infra/** -> config

## Verification commands

### python
```
just check
```

## Architecture rules

- Python package lives in `src/credit_scoring/`; tests mirror it under `tests/`.
- Tests run offline: no network calls, no cloud credentials, no running containers or clusters required. External services (MLflow server, LLM APIs, GCP, Kubernetes) are mocked or optional.
- Infrastructure code for GCP is validated only (`terraform fmt/validate/plan`, `helm lint`, `kubeconform`); never create cloud resources. `terraform apply` is allowed only for `infra/terraform/envs/local` against the local kind cluster.
- Never commit secrets, keys or real credentials; configuration comes from environment variables with `.env.example` documenting them.
- Docs and code describe the project in its own terms.
- Every task that adds a capability updates README.md and ROADMAP.md accordingly.
- Containers are engine-agnostic: Podman is the primary local engine (rootless by default; `podman compose` via podman-compose) and Docker keeps working. Scripts and recipes take the engine from `CONTAINER_ENGINE` (default `podman`) and never hard-code `docker`; compose files use only features podman-compose supports; images are built with `--format docker` so `HEALTHCHECK` is kept, and run rootless. Images reach kind through the local registry or `kind load image-archive`, never `kind load docker-image`.

## Git

- base: main
- branch: task/<id>
- worktree: yes

## Finish

- commit: yes
- push: ask
- pr: ask
- remove-worktree: ask
