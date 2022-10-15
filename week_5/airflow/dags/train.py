from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from datetime import datetime
from airflow.settings import AIRFLOW_HOME
import numpy as np

default_args = {
    "owner": "dpekach",
}


def _load_and_preprocess_data():
    data = pd.read_csv(os.path.join(AIRFLOW_HOME, 'dags', 'data', "Telecom_customer churn.csv"))

    X = data.loc[:, data.columns != 'churn']
    y = data.loc[:, data.columns == 'churn']

    # Preprocessing
    X.drop('Customer_ID', axis=1, inplace=True)
    cat_columns = X.select_dtypes(exclude='number').columns
    num_columns = X.select_dtypes(include='number').columns

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")),
               ("scale", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_columns),
            ("categorical", categorical_pipeline, cat_columns),
        ]
    )

    X_processed = full_processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, stratify=y)

    pickle.dump(X_test, open(os.path.join(AIRFLOW_HOME, 'dags', 'data', 'X_test.sav'), 'wb'))
    pickle.dump(y_test, open(os.path.join(AIRFLOW_HOME, 'dags', 'data', 'y_test.sav'), 'wb'))

    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()


def _train_and_save_model(ti):
    X_train, X_test, y_train, y_test = ti.xcom_pull(task_ids='load_and_preprocess_data')
    X_train, y_train = np.array(X_train), np.array(y_train)
    default_params = {
        'objective': 'binary:logistic'
        , 'gamma': 1  ## def: 0
        , 'booster': 'gbtree'
        , 'learning_rate': 0.1  ## def: 0.1
        , 'max_depth': 3
        , 'min_child_weight': 100  ## def: 1
        , 'n_estimators': 100
        , 'nthread': 48
        , 'random_state': 42
        , 'reg_alpha': 0
        , 'reg_lambda': 0  ## def: 1
        , 'tree_method': 'hist'  # use `gpu_hist` to train on GPU
    }

    # Training
    model = xgb.XGBClassifier(**default_params, use_label_encoder=False)
    model.fit(X_train, y_train)

    # Store model
    filename = 'my_model.sav'
    pickle.dump(model, open(os.path.join(AIRFLOW_HOME, 'dags', 'model', filename), 'wb'))
    print('Pipeline proceeded correctly!')


with DAG('week_5_airflow_train', schedule_interval='@daily', start_date=datetime.now(), catchup=False) as dag:
    load_and_preprocess_data = PythonOperator(
        task_id='load_and_preprocess_data',
        python_callable=_load_and_preprocess_data,
        do_xcom_push=True
    )

    train_and_save_model = PythonOperator(
        task_id='train_and_save_model',
        python_callable=_train_and_save_model,

    )

    load_and_preprocess_data >> train_and_save_model
