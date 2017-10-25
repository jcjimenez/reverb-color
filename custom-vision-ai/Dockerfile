FROM alpine:3.6

ARG BUILD_DEPENDENCIES="build-base python3-dev"

# Install Python 3.6
RUN apk add --no-cache ${BUILD_DEPENDENCIES} python3 libstdc++ \
  && python3 -m ensurepip \
  && rm -r /usr/lib/python*/ensurepip \
  && pip3 install -U pip setuptools \
  && if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip; fi

# Setup server
ADD requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
ADD predict_custom_vision.py /app/server.py

# Cleanup
RUN apk del ${BUILD_DEPENDENCIES} \
  && rm -r /root/.cache \
  && rm /app/requirements.txt

# Run server
EXPOSE 80
CMD ["python3.6", "/app/server.py", "--port", "80"]
