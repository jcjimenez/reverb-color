FROM jupyter/scipy-notebook:281505737f8a
LABEL maintainer="jc.jimenez@microsoft.com"
USER root

RUN apt-get update -y
RUN apt-get install \
    apt-utils \
    curl \
    git \
    python-dev \
    python3-dev \
    libopencv-dev \
    python-opencv \
    wget \
    python3 \
    python2.7 -y

