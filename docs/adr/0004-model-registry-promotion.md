# 0004: Model registry promotion

## Status

Accepted; implemented.

## Context

Training and scheduled retraining need reproducible model comparison and an explicit
promotion policy without requiring a registry service for tests. Existing ordinary
training and optional tracking must remain available under Python 3.14.

## Decision

Keep logistic regression as ordinary training. An explicit registry lifecycle trains
logistic regression and scikit-learn HistGradientBoosting on one deterministic
stratified split with train-only preprocessing. Register full pipelines under one
configured name; select by ROC AUC, then average precision, then lexicographically
smallest algorithm name. Assign the exact selected version to `challenger`.

Load the default TOML policy with importlib.resources and tomllib; validate complete
path overrides strictly. Promotion against a champion requires ROC AUC improvement
>= 0.005, PR AUC decrease <= 0.01, and PSI < 0.10. Bootstrap requires ROC AUC >= 0.70
and PSI < 0.10. Missing/nonfinite required metrics reject. KS and Gini are reporting
only. Recompute champion metrics on the current held-out rows with its stored pipeline.

PSI uses training-derived quantiles (10 bins), deduplicated internal boundaries,
infinite outer coverage, and positive smoothing (0.000001) added to each proportion
before renormalization. Identical constants give zero; a constant reference retains
one internal edge. Empty bins are smoothed; invalid probabilities fail.

Keep decisions pure; inject a small registry interface. The optional MLflow adapter
uses logging-returned model URIs, exact registered versions, readiness checks, and
registered-model alias maps. Bootstrap is allowed when the registered model lookup
reports `RESOURCE_DOES_NOT_EXIST` or the model's alias map has no `champion` entry.
Other lookup errors propagate. Cards record aggregate
dataset identity, target/features, split, parameters, metrics, policy, stability,
lineage, limitations, and outcome, without observations or credentials. Both DAGs
invoke the same CLI lifecycle when enabled and inherit typed execution-time settings.

Recheck champion before assignment and refuse observed changes. Require a single
writer across all entrypoints because alias updates lack transactional compare-and-swap.
Record promotion only after successful alias assignment. Rejection preserves champion.

## Alternatives

Tracking alone lacks promotion policy and aliases. Comparing historical champion
metrics would mix datasets and splits. New boosting dependencies add compatibility
and maintenance costs; scikit-learn already supplies HistGradientBoosting. Distributed
locking or transactional registries require a separate operational design.

## Consequences

Offline fakes test the lifecycle and adapter without services. Deployment connectivity
and permissions require separate validation. Reusing held-out data for selection and
promotion limits independence; separate validation remains future work. These gates
do not establish calibration, fairness, or regulatory approval. Rollback and persistent
drift monitoring remain planned.

See the [index](README.md), [usage and configuration](../../README.md#registry-lifecycle),
and [roadmap](../../ROADMAP.md#model-lifecycle-and-registry).
