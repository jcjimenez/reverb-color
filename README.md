# reverb-color #

Machine learning to determine the color finish name of guitar images.

Demo: https://aka.ms/reverb-color

## Machine learning ##

We tried a number of machine learning models to determine the finish name for a
given guitar image.

- [custom-vision-ai](custom-vision-ai/README.md)
- [inception](inception/README.md)
- [mobilenet](mobilenet/README.md)

The Custom Vision AI classifier is the most reliable. Latency is about 1000ms
to 1500ms. Mobilenet is the fastest classifier at about 300ms to 500ms latency.

We tried two main approaches to model building: hierarchical and flat. The flat
models are simply classifiers trained on the full set of labels.  Hierarchical
models are first trained on granular categories like "red" and "blue" and then
within each of the shards a second model gets trained to distinguish between
the more subtle nuances of the specific finish names. The hierachical models
are more accurate if the root color family model performs well. The flat models
perform better if it is not possible to create an accurate root model.

## Data pre-processing ##

We trained custom [sharpmask](sharpmask/README.md) and
[fasterRCNN](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN)
models to detect guitars in images. This enables us to crop out non-guitar
parts of images which we hypothesize will increase classifier performance. We
used the [VOTT tool](https://github.com/CatalystCode/VOTT) to annotate training
data for the models.

For pre-processing, we also tried to automatically
[discover color families](color-family-discovery/README.md) to eliminate human
assumptions in how the various color finishes map to color families. However,
we didn't manage to get the clustering to work well due to noise in the images.

## Operations and deployment ##

All the models are hosted on virtual machines behind a http service with the
same interface so that the models can be accessed in a uniform manner at
runtime. We tried hosting the models and services on
[Azure Container Instances](https://azure.microsoft.com/en-us/services/container-instances/)
but this added 2000ms to 4000ms extra latency.

The http services for Inception and MobileNet are based on the Python library
[hug](http://www.hug.rest) which enables developers to add http interfaces to
functions with simple decorators. The http service for Custom Vision AI is
based on [sanic](https://sanic.readthedocs.io/en/latest/), a very fast async
flask-like web framework for Python.

## Future work ##

- Operationalize the guitar object detection models and hook them up as a
  pre-processing step for the image classification.

- Run the color family discovery on a data-set that is pre-processed with the
  guitar object detection models.

