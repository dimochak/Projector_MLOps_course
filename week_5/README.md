### Airflow deployment instructions:
_1. On Linux machine (I did it in WSL), run: 
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.4.1/docker-compose.yaml'_ \
_2. Create dockerfile with requirements.txt_ \
_3. In docker-compose uncomment build . for fetching a local dockerfile with additional libs_
_4. mkdir -p ./dags ./logs ./plugins_ \
_5. echo -e "AIRFLOW_UID=$(id -u)" > .env_ \
_6. docker-compose up airflow-init_ \
_7. docker-compose up_ 


Airflow dashboard will be available on http://localhost:8080/home (Username/pass: airflow/airflow)
