FROM python:3.7-slim-buster

RUN apt-get update && apt-get install make


RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


RUN ln -s /usr/bin/python3 /usr/bin/python
COPY . .

CMD [ "bash" ]