FROM python:3.7

COPY ./* tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Setting python environment variable to log statements directly to stdout
ENV PYTHONUNBUFFERED=1
CMD python tmp/main.py