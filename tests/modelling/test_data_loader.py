import pandas as pd

from credit_scoring.modeling.data_loader import (
    load_credit_data,
)


def test_load_credit_data_from_custom_csv_path(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    source_df = pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": [20000, 120000],
        }
    )
    source_df.to_csv(csv_path, index=False)

    loaded_df = load_credit_data(csv_path)

    pd.testing.assert_frame_equal(loaded_df, source_df)
