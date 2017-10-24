# Inception
This directory contains the implementation proposal based on TensorFlow and Inception v3.

# Setup
To get started with this, run the following:

```
docker build .
```

Copy the has after at the end of the line that reads like `Successfully built 28bb1679335c` and replace `28bb1679335c` with what you see below:

```
docker run --rm -v `pwd`:/container/reverb-color -w /container/reverb-color -it -p 8888:8888 28bb1679335c bash
```

# Running Prediction
In order to run prediction against an image, you can do this:

```
/opt/tensorflow/examples/label_image/label_image --graph=/opt/model/output_graph.pb --labels=/opt/model/output_labels.txt --output_layer=final_result:0 --input_width=320 --input_height=320 --image=/container/reverb-color/jazzmaster_seafoam_hero.jpg
```

# Retrain the model
To retrain the model, ensure you have a list of images like the ones in https://reverb904af5d223d2.blob.core.windows.net/training/reverb-data-320-clean.zip
then run something like the following:

```
/opt/tensorflow/examples/image_retraining/retrain --image_dir /container/reverb-color/reverb-data-320-clean
```

The model files will be output to /tmp/output_graph.pb and /tmp/output_labels.txt

