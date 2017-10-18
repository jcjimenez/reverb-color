# Inception
This directory contains the implementation proposal based on TensorFlow and Inception v3.

# Setup
To get started with this, run the following:

```
docker build .
```

Copy the has after at the end of the line that reads like `Successfully built bf2b5ebb6186` and replace `bf2b5ebb6186` with what you see below:

```
docker run --rm -v `pwd`:/container/reverb-color -w /container/reverb-color -it -p 8888:8888 bf2b5ebb6186 bash
```

# Running Prediction
In order to run prediction against an image, you can do this:

```
/opt/tensorflow/examples/label_image/label_image --graph=/opt/model/output_graph.pb --labels=/opt/model/output_labels.txt --output_layer=final_result:0 --image=/container/reverb-color/jazzmaster_seafoam_hero.jpg
```

# Download an updated model
In order to run prediction, you'll need a model, which you can get the latest via:
```
curl -O https://reverb904af5d223d2.blob.core.windows.net/training/model-2017-10-17.tgz
tar xfz model-2017-10-17.tgz
```

# Setup Inception From Source
To build inception from source, you can run the following commands:

```
git clone git@github.com:tensorflow/tensorflow.git
curl -O https://reverb904af5d223d2.blob.core.windows.net/training/reverb-data-320-clean.zip
unzip reverb-data-320-clean.zip
find reverb-data-320-clean -name .DS_Store | xargs rm

cd tensorflow/
./configure
bazel build --local_resources 2048,.5,1.0 tensorflow/examples/image_retraining:retrain
cp -rp bazel-bin/tensorflow/examples/image_retraining ../bin/label_image
```

