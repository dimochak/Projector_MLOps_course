## 1) Deploy minio on docker
_docker build . -t week_2:latest_ \
_docker run -p 9000:9000 -p 9001:9001 week_2:latest_ \
\
MinIO console will be available on 127.0.0.1:9001

## 2) MinIO CRUD python client 
Run _client.py._ \
_Note:_ If a test bucket inside MinIO is not created in advance, it could be created in _insert\_data_ method.

## 3) Pandas loading/saving 
Run _pandas_evaluator.py_

## 4) Multithreading benchmarking
Run _multithreading_benchmark.py_ \
\
_Note: For executing points 2-4 properly you should create a virtual environment with libs installed from requrements.txt_