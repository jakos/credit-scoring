import pandas as pd

from credit_scoring.training.train import train


def test_train_model_on_small_sample() -> None:
    df = pd.read_csv("data/raw/UCI_Credit_Card.csv").sample(n=2000, random_state=42)

    artifacts = train(df=df, max_iter=300)

    assert len(artifacts.x_test) > 0
    assert len(artifacts.y_pred) == len(artifacts.y_test)
    assert len(artifacts.y_proba) == len(artifacts.y_test)


def test_pipeline_matches_original_algorithm() -> None:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from credit_scoring.evaluation.metrics import evaluate_predictions

    rng = np.random.default_rng(18)
    df = pd.DataFrame(
        {
            "ID": range(120),
            "a": rng.normal(size=120),
            "b": rng.normal(size=120),
            "text": ["ignored"] * 120,
            "default.payment.next.month": [0, 1] * 60,
        }
    )
    df.loc[::7, "a"] = np.nan
    x = df.drop(columns=["ID", "default.payment.next.month"])
    y = df["default.payment.next.month"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    reference = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            ["a", "b"],
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    reference.fit(x_train, y_train)
    actual = train(df=df)
    pd.testing.assert_frame_equal(actual.x_test, x_test)
    pd.testing.assert_series_equal(actual.y_test, y_test)
    assert actual.pipeline["model"].get_params() == reference["model"].get_params()
    np.testing.assert_allclose(
        actual.pipeline["preprocessor"].transform(x_test),
        reference["preprocessor"].transform(x_test),
    )
    np.testing.assert_array_equal(actual.y_pred, reference.predict(x_test))
    np.testing.assert_allclose(actual.y_proba, reference.predict_proba(x_test)[:, 1])
    expected = {
        "accuracy": accuracy_score(y_test, reference.predict(x_test)),
        "precision": precision_score(y_test, reference.predict(x_test), zero_division=0),
        "recall": recall_score(y_test, reference.predict(x_test), zero_division=0),
        "f1": f1_score(y_test, reference.predict(x_test), zero_division=0),
        "roc_auc": roc_auc_score(y_test, reference.predict_proba(x_test)[:, 1]),
        "pr_auc": average_precision_score(y_test, reference.predict_proba(x_test)[:, 1]),
    }
    assert (
        evaluate_predictions(actual.y_test, pd.Series(actual.y_pred), pd.Series(actual.y_proba))
        == expected
    )


def test_dataset_path_and_dataframe_precedence(tmp_path, monkeypatch) -> None:
    import importlib
    from unittest.mock import Mock

    module = importlib.import_module("credit_scoring.training.train")
    df = pd.DataFrame({"ID": range(40), "a": range(40), "default.payment.next.month": [0, 1] * 20})
    csv = tmp_path / "dataset.csv"
    df.to_csv(csv, index=False)
    loaded = train(dataset_path=csv)
    direct = train(df=df, dataset_path=tmp_path / "absent.csv")
    pd.testing.assert_frame_equal(loaded.x_test, direct.x_test)
    loader = Mock(return_value=df)
    monkeypatch.setattr(module, "load_credit_data", loader)
    train(df=df, dataset_path=csv)
    loader.assert_not_called()
    train(dataset_path=csv)
    loader.assert_called_once_with(csv)
