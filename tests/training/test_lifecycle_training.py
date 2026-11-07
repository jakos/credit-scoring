import numpy as np
import pandas as pd

from credit_scoring.training.train import train_lifecycle


def test_both_families_share_split_and_fit_preprocessing_on_training_only() -> None:
    df = pd.DataFrame(
        {"ID": range(100), "feature": np.arange(100.0), "default.payment.next.month": [0, 1] * 50}
    )
    results = train_lifecycle(df, 50)
    assert set(results) == {"logistic_regression", "hist_gradient_boosting"}
    first, second = results.values()
    pd.testing.assert_frame_equal(first.x_test, second.x_test)
    pd.testing.assert_frame_equal(first.x_train, second.x_train)
    assert set(first.x_train.index).isdisjoint(first.x_test.index)
    for result in results.values():
        assert result.train_scores.shape == (80,)
        assert result.test_scores.shape == (20,)
        scaler = (
            result.pipeline.named_steps["preprocessor"]
            .named_transformers_["numeric"]
            .named_steps["scaler"]
        )
        assert scaler.mean_[0] == result.x_train.feature.mean()
    repeated = train_lifecycle(df, 50)
    for name in results:
        np.testing.assert_array_equal(results[name].test_scores, repeated[name].test_scores)


def test_nondefault_seed_is_used_for_split_and_artifacts() -> None:
    from credit_scoring.features.preprocess import train_test_split_stratified

    df = pd.DataFrame({"feature": range(40), "default.payment.next.month": [0, 1] * 20})
    results = train_lifecycle(df, 50, random_state=137)
    expected = train_test_split_stratified(df[["feature"]], df.iloc[:, 1], random_state=137)
    for artifact in results.values():
        assert artifact.split_seed == 137
        pd.testing.assert_frame_equal(artifact.x_train, expected[0])
        pd.testing.assert_frame_equal(artifact.x_test, expected[1])
    assert all(a.split_seed == 42 for a in train_lifecycle(df, 50).values())
