# custom-vision-ai

This directory contains the implementation proposal based on the CustomVision.ai service.

## Setup

To get started with this, first get your training key from the [customvision.ai settings pane](https://www.customvision.ai/projects#/settings).
Then run the following:

```sh
# set up python
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# download training data
curl -O https://reverb904af5d223d2.blob.core.windows.net/training/reverb-data-320-clean.zip
unzip reverb-data-320-clean.zip
```

You're now all set to build a model.

## Usage

The following snippet will upload tagged images to the custom vision service and kick of the training process for a
model. You can check the training progress and performance of the model via the custom vision service [dashboard](https://www.customvision.ai/projects).

```sh
venv/bin/python train_custom_vision.py \
  --project_name="reverb-data-320-clean-all" \
  --training_key="INSERT_YOUR_TRAINING_KEY_HERE" \
  --data_dir="./reverb-data-320-clean"
```

If you want to train a model on just a few labels, you can use the following:

```sh
venv/bin/python train_custom_vision.py \
  --project_name="reverb-data-320-clean-red-labels" \
  --training_key="INSERT_YOUR_TRAINING_KEY_HERE" \
  --data_dir="./reverb-data-320-clean" \
  --labels="aged-cherry-burst,aged-cherry-sunburst"
```

If you want to train a model using fewer training data, you can use the following:

```sh
venv/bin/python train_custom_vision.py \
  --project_name="reverb-data-320-clean-red-labels" \
  --training_key="INSERT_YOUR_TRAINING_KEY_HERE" \
  --data_dir="./reverb-data-320-clean" \
  --max_images=50
```

## Development

If you want to add a feature to the `custom_vision_client` library used in the scripts in this folder, open a pull
request against its repository: [py_custom_vision_client](https://github.com/CatalystCode/py_custom_vision_client).
