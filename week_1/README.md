## Image Build: 
_docker build . -t week_1:latest_
## Docker Launch: 
_docker run -p 8000:8000 week_1:latest_
## Docker image share, Windows 
 - _docker login -u dpekach_
 - Type your password
 - _docker tag week_1:latest dpekach/week_1:latest_
 - _docker push dpekach/week_1:latest_

## Results:
 - Launched server with result of LR will be on localhost:8000
 - Image is shared on DockerHub