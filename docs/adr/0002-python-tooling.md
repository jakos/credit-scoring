# 0002: Python tooling and shared gate

## Status

Accepted; implemented.

## Context

Development and CI need the same reproducible checks without external services.
The repository targets Python 3.14 and commits its dependency lock.

## Decision

Use Python 3.14 from `.python-version`, uv for dependencies and builds through
Hatchling, and the default dev/modeling groups for development. Keep tooling in
the dev group and resolved versions in `uv.lock`.

`just check` runs locked Ruff formatting checks, Ruff lint, mypy, then pytest,
sequentially. GitHub Actions provisions versioned tools and runs `uv sync --locked`
and that same gate. Action references in the workflow use full commit SHAs.
Prek installs commit checks and pre-push tests from `prek.toml`.

Mypy targets all of `src/credit_scoring/`, including compatibility wrappers, with
checked bodies, required function annotations, and unused-ignore reporting. Tests
and standalone DAGs are outside its initial target. Pandas 2.3 stubs match the
locked pandas 2.3 runtime and execute with mypy on Python 3.14. Scikit-learn lacks
typing metadata, so only its modules have a missing-import override. No application
modules are excluded. Pytest covers CLI, settings, logging, numerical regression,
and DAG behavior offline using stubs and mocks.

## Alternatives

Separate CI commands can drift from local checks. Global missing-import exemptions
would hide application import errors. Adding Airflow or live services to the test
environment would make the shared gate slower and dependent on infrastructure.

## Consequences

Contributors run one gate on Windows or Linux. Lock updates accompany dependency
changes; unrelated upgrades are separate work. Third-party typing remains limited
at the scikit-learn boundary. Offline DAG tests do not validate a live scheduler.

See the [index](README.md) and [development setup](../../README.md#development-setup).
