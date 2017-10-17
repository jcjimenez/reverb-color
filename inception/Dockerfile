FROM tensorflow/tensorflow:1.3.0-py3
LABEL maintainer="jc.jimenez@microsoft.com"
USER root

RUN apt-get update -y
RUN apt-get install \
    apt-utils \
    curl \
    git \
    wget \
    vim \
    openjdk-8-jdk \
    -y

RUN curl https://reverb904af5d223d2.blob.core.windows.net/training/label_image-ubuntu-2017-10-17.tgz | tar xz -C /opt/
RUN curl https://reverb904af5d223d2.blob.core.windows.net/training/model-2017-10-17.tgz | tar xz -C /opt/
