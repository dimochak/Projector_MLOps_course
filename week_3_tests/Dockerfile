FROM python
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH /app
COPY . .

CMD [ "bash" ]