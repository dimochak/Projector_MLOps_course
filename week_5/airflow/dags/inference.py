import json

from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import pickle
from datetime import datetime
from airflow.settings import AIRFLOW_HOME
from sklearn.metrics import f1_score

default_args = {
    "owner": "dpekach",
}


def _load_inference_data():
    X_test = pickle.load(open(os.path.join(AIRFLOW_HOME, 'dags', 'data', 'X_test.sav'), 'rb'))
    y_test = pickle.load(open(os.path.join(AIRFLOW_HOME, 'dags', 'data', 'y_test.sav'), 'rb'))
    return X_test.tolist(), y_test.tolist()


def _load_and_inference_model(ti):
    X_test, y_test = ti.xcom_pull(task_ids='load_inference_data')
    model = pickle.load(open(os.path.join(AIRFLOW_HOME, 'dags', 'model', 'my_model.sav'), 'rb'))

    y_pred = model.predict(X_test)
    f1_ = f1_score(y_test, y_pred)
    result = {'F1 score': f1_}
    return result


def _save_results(ti):
    res = ti.xcom_pull(task_ids='load_and_inference_model')
    with open(os.path.join(AIRFLOW_HOME, 'dags', 'results', 'results.json'), 'w') as f:
        json.dump(res, f)
    print('Inference completed successfully')


with DAG('week_5_airflow_inference', schedule_interval='@daily', start_date=datetime.now(), catchup=False) as dag:

    load_inference_data = PythonOperator(
        task_id='load_inference_data',
        python_callable=_load_inference_data,
        do_xcom_push=True
    )

    load_and_inference_model = PythonOperator(
        task_id='load_and_inference_model',
        python_callable=_load_and_inference_model,
        do_xcom_push=True
    )

    save_results = PythonOperator(
        task_id='save_results',
        python_callable=_save_results
    )

    load_inference_data >> load_and_inference_model >> save_results


