# Inception
This directory contains a  implementation proposal based on TensorFlow and Inception v3.

# Running Prediction
In order to run prediction against an image, you can do this:

```
docker-compose up
```

Then make HTTP request to the service with something like (make sure commas are URL escaped):

```
curl 'http://localhost:8000/v1/finish?image_url=http://res.cloudinary.com/reverb/image/upload/s---YUQ-2CS--/c_thumb%2Ch_320%2Cw_320/rpxkfvq8wnvui2bxvfif.jpg'
```

# Retrain the model
Make sure you have a directory that contains directories representing image
classes that looks something like this:

```
color-families
├── green
│   ├── qweixldfjuwendksirns.jpg
│   └── rpxkfvq8wnvui2bxvfif.jpg
└── blue
    ├── asdfwerusdfjwessdfwe.jpg
    └── qwerasdzzsdfrtwersdf.jpg
```

Then run something like the following for mobilenet:

```
python3 retrain.py --architecture mobilenet_0.25_128_quantized --image_dir color-families-128 --output_graph model/color-families/graph.pb --output_labels model/color-families/labels.txt
```

The model files should be saved in a way that follows this directory structure:

```
model
├── color-families
│   ├── graph.pb
│   └── labels.txt
└── green
    ├── graph.pb
    └── labels.txt
```

