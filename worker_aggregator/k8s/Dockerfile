FROM ubuntu:16.04
LABEL maintainer="nlkey2022@gmail.com"
RUN apt update && \
    apt install -y python3 python3-pip && \
    pip3 install flask requests

COPY worker.py /tmp/worker.py
ENTRYPOINT ["python3", "/tmp/worker.py"]
