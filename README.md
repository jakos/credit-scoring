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

## Run training

```bash
uv sync --group modeling --group dev
uv run train-model --max-iter 300
```

## Project Structure

- `src/credit_scoring/`: application package
- `tests/`: test suite
- `pyproject.toml`: project metadata and tooling config
