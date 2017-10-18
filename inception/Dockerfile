FROM tensorflow/tensorflow:1.3.0-py3
LABEL maintainer="jc.jimenez@microsoft.com"
USER root

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update -y
RUN apt-get install \
    apt-utils \
    curl \
    git \
    wget \
    vim \
    openjdk-8-jdk \
    bazel \
    -y

RUN curl https://reverb904af5d223d2.blob.core.windows.net/training/tensorflow-ubuntu-2017-10-17.tgz | tar xz -C /opt/
RUN curl https://reverb904af5d223d2.blob.core.windows.net/training/inceptionv3-model-2017-10-17.tgz | tar xz -C /opt/

