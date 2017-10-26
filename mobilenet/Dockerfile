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

RUN pip3 install hug hug_middleware_cors

ADD predict.py /app/server.py
EXPOSE 80
CMD ["hug", "-f", "/app/server.py", "--port", "80"]
