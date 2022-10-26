import xgboost as xgb
from great_expectations.dataset import PandasDataset
from sklearn.metrics import accuracy_score


def test_model_performance(data: PandasDataset):
    df_train, df_val, df_test,  = data
    model = xgb.XGBClassifier()

    X_train, y_train = df_train.loc[:, df_train.columns != 'class'], df_train.loc[:,  df_train.columns == 'class']
    X_test, y_test = df_test.loc[:, df_test.columns != 'class'], df_test.loc[:,  df_test.columns == 'class']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert accuracy_score(y_test, y_pred) == 1.0