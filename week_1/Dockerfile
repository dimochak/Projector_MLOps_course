FROM python:3.10

WORKDIR week_1
COPY requirements.txt /week_1/requirements.txt
RUN pip install --no-cache-dir -r /week_1/requirements.txt
COPY . .

# Setting python environment variable to log statements directly to stdout
ENV PYTHONUNBUFFERED=1
CMD python main.py