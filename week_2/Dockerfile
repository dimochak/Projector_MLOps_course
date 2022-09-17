FROM docker.io/minio/minio:latest

ENV MINIO_ROOT_USER="dpekach"
ENV MINIO_ROOT_PASSWORD="bestadmin"
RUN mkdir "buckets"
EXPOSE 9000 9001

CMD ["minio", "server", "/buckets", "--address", ":9000", "--console-address", ":9001"]