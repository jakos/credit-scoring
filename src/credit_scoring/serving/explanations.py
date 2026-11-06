"""Native additive explanations for the fitted numeric logistic pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_scoring.serving.contracts import Explanation, FeatureContribution


@dataclass(frozen=True)
class LinearExplanation:
    pipeline: Any
    names: tuple[str, ...]
    coefficients: Any
    baseline: float
    orientation: int

    def explain(self, frame: pd.DataFrame) -> list[Explanation]:
        transformed = np.asarray(self.pipeline[:-1].transform(frame), dtype=float)
        contributions = transformed * self.coefficients
        decisions = self.orientation * np.asarray(self.pipeline.decision_function(frame))
        if (
            contributions.shape != (len(frame), len(self.names))
            or not np.isfinite(contributions).all()
            or not np.isfinite(decisions).all()
            or not np.allclose(
                self.baseline + contributions.sum(axis=1), decisions, rtol=1e-9, atol=1e-9
            )
        ):
            raise ValueError("Invalid explanation output")
        results = []
        for row in contributions:
            ranked = sorted(zip(self.names, row, strict=True), key=lambda x: (-abs(x[1]), x[0]))
            results.append(
                Explanation(
                    method="linear_log_odds",
                    baseline_log_odds=self.baseline,
                    top_features=[
                        FeatureContribution(feature=n, contribution=float(v)) for n, v in ranked[:5]
                    ],
                    remaining_contribution=float(sum(v for _, v in ranked[5:])),
                )
            )
        return results


def build_explanation(
    pipeline: Any, names: tuple[str, ...], default_column: int
) -> LinearExplanation:
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 2:
        raise ValueError("Unsupported explanation pipeline")
    preprocessor, model = (step[1] for step in pipeline.steps)
    if not isinstance(preprocessor, ColumnTransformer) or not isinstance(model, LogisticRegression):
        raise ValueError("Unsupported explanation model")
    mapped: list[str] = []
    for _, transform, columns in preprocessor.transformers_:
        if isinstance(transform, str) and transform == "drop":
            continue
        if (
            not isinstance(transform, Pipeline)
            or len(transform.steps) != 2
            or not isinstance(transform.steps[0][1], SimpleImputer)
            or transform.steps[0][1].strategy != "median"
            or transform.steps[0][1].add_indicator
            or not isinstance(transform.steps[1][1], StandardScaler)
            or not isinstance(columns, (list, tuple))
            or not all(isinstance(c, str) for c in columns)
        ):
            raise ValueError("Unsupported transformed feature mapping")
        output_names = tuple(transform.get_feature_names_out(columns))
        if output_names != tuple(columns):
            raise ValueError("Ambiguous transformed feature mapping")
        mapped.extend(output_names)
    if len(mapped) != len(set(mapped)) or set(mapped) != set(names):
        raise ValueError("Incomplete transformed feature mapping")
    orientation = 1 if default_column == 1 else -1
    coefficients = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if (
        coefficients.shape != (1, len(mapped))
        or intercept.shape != (1,)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
    ):
        raise ValueError("Invalid linear parameters")
    return LinearExplanation(
        pipeline,
        tuple(mapped),
        orientation * coefficients[0],
        float(orientation * intercept[0]),
        orientation,
    )
