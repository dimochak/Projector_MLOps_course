FROM python:3.7-slim-buster
WORKDIR /app
RUN apt-get update && apt-get install make


RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "bash" ]