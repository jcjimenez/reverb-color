# frcnn
This REST service is built on top of keras-frcnn.

# Running
Make sure you run the server:

```
hug -f predict.py
```

Then make a GET request with the URL of an image:

```
curl 'http://localhost:8000/v1/guitar_bodies?image_url=https://images.reverb.com/image/upload/s--J3gg8O0u--/a_exif%2Cc_limit%2Ce_unsharp_mask:80%2Cf_auto%2Cfl_progressive%2Cg_south%2Ch_1600%2Cq_80%2Cw_1600/v1415395952/u0ftxwzelezloz7ygmfw.jpg'
```

# TODO
Implement based on https://github.com/delftrobotics/keras-retinanet

