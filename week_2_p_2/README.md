## 1) Commit data with DVC into github repo 

 In empty repository run the following commands: 
- _dvc init --subdir;_
- put your data to test folder;
- _dvc add .\test_data\test_data.csv_
- _git add test_data\.gitignore test_data\test_data.csv.dvc_
- _git commit -m "Add initial version of test_data"_
- _dvc remote add -d minio s3://test-bucket -f_
- _dvc remote modify minio endpointurl http://127.0.0.1:9000_
- _dvc remote modify minio access_key_id dpekach_
- _dvc remote modify minio secret_access_key bestadmin_
- _git add .dvc\config_
- _git commit -m "Configure remote storage"_
- _dvc push_
- _git push origin/main_

## 2) PR with LabelStudio deploy
- _docker build . -t week_2_p_2:latest_
- _docker run -p 8080:8080 week_2_p_2:latest_

LabelStudio will be available on 127.0.0.1:8080

## 3) PR with LakeFS deploy
- _Take docker compose file from http://compose.lakefs.io/_
- _run docker-compose -f docker-compose.yaml up -d_\

LakeFS will be available on localhost:8000 \

_Note: If a test repository inside LakeFS is not created in advance, it could be created in insert_data method._