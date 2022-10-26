from typing import Tuple

from great_expectations.dataset.pandas_dataset import PandasDataset


def test_data_shape(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data
    assert df_train.shape[0] + df_val.shape[0] == 125
    assert df_train.shape[1] == df_val.shape[1] == 5
    assert df_test.shape == (25, 5)


def test_data_content(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data

    assert df_train.expect_column_values_to_not_be_null(column="sepal length (cm)")["success"]
    assert df_val.expect_column_values_to_not_be_null(column="sepal length (cm)")["success"]
    assert df_test.expect_column_values_to_not_be_null(column="sepal length (cm)")["success"]

    assert df_train.expect_column_values_to_be_in_set("class", [1, 0, 2])["success"]
    assert df_test.expect_column_values_to_be_in_set("class", [1, 0, 2])["success"]