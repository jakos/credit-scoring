import numpy as np
import pandas as pd

from credit_scoring.modeling.preprocess import (
    TARGET_COLUMN,
    build_preprocessor,
    load_credit_data,
    split_features_target,
    train_test_split_stratified,
)


def test_load_credit_data_from_custom_csv_path(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    source_df = pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": [20000, 120000],
            TARGET_COLUMN: [0, 1],
        }
    )
    source_df.to_csv(csv_path, index=False)

    loaded_df = load_credit_data(csv_path)

    pd.testing.assert_frame_equal(loaded_df, source_df)


def test_split_features_target_drops_target_and_requested_columns() -> None:
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "feature_a": [10, 20, 30],
            "feature_b": [0.1, 0.2, 0.3],
            TARGET_COLUMN: [0, 1, 0],
        }
    )

    x, y = split_features_target(df, drop_columns=["ID"])

    assert list(x.columns) == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0]


def test_train_test_split_stratified_preserves_class_ratio() -> None:
    x = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 80 + [1] * 20)

    x_train, x_test, y_train, y_test = train_test_split_stratified(
        x,
        y,
        test_size=0.25,
        random_state=123,
    )

    assert len(x_train) == 75
    assert len(x_test) == 25
    assert y_train.value_counts().to_dict() == {0: 60, 1: 15}
    assert y_test.value_counts().to_dict() == {0: 20, 1: 5}


def test_build_preprocessor_transforms_and_imputes_numeric_columns() -> None:
    x = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [10.0, np.nan, 30.0, 40.0],
        }
    )

    preprocessor = build_preprocessor(["a", "b"])
    transformed = preprocessor.fit_transform(x)

    assert transformed.shape == (4, 2)
    assert np.isfinite(transformed).all()
