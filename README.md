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

## Project Structure

- `src/credit_scoring/`: application package
- `tests/`: test suite
- `pyproject.toml`: project metadata and tooling config

## Run

```bash
uv run python -m credit_scoring.main
```
