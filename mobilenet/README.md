# Mobilenet
This directory contains a  implementation proposal based on TensorFlow and Mobilenet.

# Running Prediction
In order to run prediction against an image, you can do this:

```
docker run -p 8000:80 cwolff/reverbmobilenet
```

Then make HTTP request to the service with something like (make sure commas are URL escaped):

```
curl 'http://localhost:8000/v1/finish?image_url=http://res.cloudinary.com/reverb/image/upload/s---YUQ-2CS--/c_thumb%2Ch_320%2Cw_320/rpxkfvq8wnvui2bxvfif.jpg'
```

# Retrain multiple models

Make sure you have a directory that contains directories representing 128x128
image classes that looks something like this:

```
color-families
├── green
│   ├── olive
|   │   ├── qweixldfjuwendksirns.jpg
|   │   └── rpxkfvq8wnvui2bxvfif.jpg
│   ├── seafoam
|   │   ├── qweixldfjuwendksirns.jpg
|   │   └── rpxkfvq8wnvui2bxvfif.jpg
└── blue
    ├── daphne-blue
    │   ├── qweixldfjuwendksirns.jpg
    │   └── rpxkfvq8wnvui2bxvfif.jpg
    └── electric-blue
        ├── qweixldfjuwendksirns.jpg
        └── rpxkfvq8wnvui2bxvfif.jpg
```

Then run something like the following for mobilenet:

```
python3 generate_models --architecture mobilenet_1.0_128 --image_dir=color-families --output_dir=model-mobilenet
```

# Retrain a single model
Make sure you have a directory that contains directories representing 128x128
image classes that looks something like this:

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
python3 retrain.py --architecture mobilenet_1.0_128 --image_dir color-families-128 --output_graph model/color-families/graph.pb --output_labels model/color-families/labels.txt
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

