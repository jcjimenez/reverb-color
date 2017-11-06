FROM tensorflow/tensorflow:1.3.0-py3
LABEL maintainer="jc.jimenez@microsoft.com"
USER root

RUN apt-get update -y --fix-missing
RUN apt-get install -y --fix-missing \
    curl \
    git \
    libopencv-dev python-opencv \
    vim

ADD . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

