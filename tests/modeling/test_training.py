import pandas as pd

from credit_scoring.modeling.train import train


def test_train_model_on_small_sample() -> None:
    df = pd.read_csv("data/raw/UCI_Credit_Card.csv").sample(n=2000, random_state=42)

    artifacts = train(df=df, max_iter=300)

    assert len(artifacts.x_test) > 0
    assert len(artifacts.y_pred) == len(artifacts.y_test)
    assert len(artifacts.y_proba) == len(artifacts.y_test)
