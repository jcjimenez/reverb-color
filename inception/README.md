# Inception
This directory contains the implementation proposal based on TensorFlow and Inception v3.

# Running Prediction
In order to run prediction against an image, you can do this:

```
docker-compose up
```

Then make HTTP request to the service with something like:

```
curl 'http://localhost:8000/v1/finish?image_url=rpxkfvq8wnvui2bxvfif.jpg'
```

# Retrain the model
To retrain the model, ensure you have a list of images like the ones in https://reverb904af5d223d2.blob.core.windows.net/training/reverb-data-320-clean.zip
then run something like the following:

```
/opt/tensorflow/examples/image_retraining/retrain --image_dir /container/reverb-color/reverb-data-320-clean
```

The model files will be output to /tmp/output_graph.pb and /tmp/output_labels.txt

