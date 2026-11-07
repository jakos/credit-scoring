# 0001: Package layout

## Status

Accepted; implemented.

## Context

The baseline pipeline loads CSV data, prepares numeric features, trains logistic
regression, evaluates predictions, and optionally records an MLflow run. Existing
callers import from `credit_scoring.modeling`.

## Decision

Keep the application under `src/credit_scoring/`, with focused `data`, `features`,
`training`, `evaluation`, and `config` packages and shared JSON logging.
`training.main` coordinates the CLI; the `train-model` entry point delegates to it.
`training.train` accepts explicit inputs and does not read environment settings.
CLI and DAG entry points construct fresh settings when they execute.

Retain thin `modeling` wrappers that re-export the original public objects and
delegate the legacy module entry point. Tests mirror package ownership and cover
compatibility and numerical behavior.

## Alternatives

A single modeling package keeps imports simple but mixes ownership. A separate
service or deeper domain-layer hierarchy would add boundaries beyond the current
pipeline's needs. Removing legacy imports would break existing callers.

## Consequences

Changes have clear package owners while callers retain compatibility. Wrappers
must stay thin and covered by regression tests. Serving and registry boundaries
remain future decisions; this layout does not implement them.

See the [index](README.md) and [platform structure](../../README.md#package-structure).
