# frcnn
This REST service is built on top of keras-frcnn.

# Setup
First you'll need to download and unpack the pre-trained model (or train your
own by following the instructions in the section below):

```
curl -O https://reverb904af5d223d2.blob.core.windows.net/training/model-frcnn.tgz
tar tfz model-frcnn.tgz
```

This will output two files:

```
config.pickle
model_frcnn.hdf5
```

# Running
Make sure you run the server:

```
docker-compose up
```

Then make a GET request with the URL of an image:

```
curl 'http://localhost:8000/v1/guitar_bodies?image_url=https://images.reverb.com/image/upload/s--J3gg8O0u--/a_exif%2Cc_limit%2Ce_unsharp_mask:80%2Cf_auto%2Cfl_progressive%2Cg_south%2Ch_1600%2Cq_80%2Cw_1600/v1415395952/u0ftxwzelezloz7ygmfw.jpg'
```

# Training your own model
The easiest way to train your model is to create a csv file that looks something
like the following:

```
/path/to/image.jpg,x,y,width,height,class_name
```

Then run this command:

```
cd keras-frcnn
python3 train_frcnn.py --path /path/to/file.csv --parser simple
```

this will output the two files you'll need to place alongside `predict.py`:

```
config.pickle
model_frcnn.hdf5
```

# TODO
Implement based on https://github.com/delftrobotics/keras-retinanet

