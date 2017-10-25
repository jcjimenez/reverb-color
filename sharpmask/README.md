# Reverb Sharpmask

Run Facebook sharpmask on Reverb guitar images

## Requirements

CUDA drivers for your system must be installed. Only really seems to work on linux. `nvidia-docker` must also be installed.

## Creating data

Run `scripts/tsv-to-annotations-json /path/to/labelled/guitars/positive data` to
generate annotations json and properly organized images directories. Script
requires ruby.

## Building container

```
sudo ./dev/build
```

## Running container

```
sudo ./dev/run
```

## Training model

Assuming you have a set of images plus bboxes.tsv files created by [VoTT](https://github.com/CatalystCode/VoTT), you can run `./scripts/tsv-to-annotations-json /path/to/images/and/tsvs ./data` to generate the training data.

Then, run `sudo ./dev/run` to enter the docker container. Then:

```
th train.lua -datadir ~/data -rundir ~/data/exps # kill this after desired amount of time
th train.lua -datadir ~/data -rundir ~/data/exps -dm ~/data/exps/deepmask/exp/bestmodel.t7 # ditto about killing task
```

## Running model

```
th creatMask.lua ~/data/exps/sharpmask/exp/bestmodel.t7 -img /path/to/img -output ~/data/out.jpg
```
