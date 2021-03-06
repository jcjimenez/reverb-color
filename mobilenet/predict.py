import argparse
import datetime
import os
import sys
from glob import glob
from shutil import unpack_archive

import hug
import numpy as np
import requests
import tensorflow as tf
from hug_middleware_cors import CORSMiddleware

api = hug.API(__name__)
api.http.add_middleware(CORSMiddleware(api))

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph


def download_file(url, to_path='', chunk_size=1024):
    local_filename = to_path or url.split('/')[-1]

    response = requests.get(url, stream=True)
    with open(local_filename, 'wb') as fobj:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fobj.write(chunk)


if not glob(os.path.join('model', '*')):
    download_file(os.getenv('TF_MODEL_URL'), 'model.tgz')
    unpack_archive('model.tgz')


MODELS = {}
for model_name in os.listdir("model"):
    print("Loading model %s" % model_name)
    MODELS[model_name] = load_graph(os.path.join("model", model_name, "graph.pb"))

def read_tensor_from_image_file(image_url, input_height=128, input_width=128, input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image_data = requests.get(image_url).content
  image = tf.image.decode_jpeg(image_data, channels = 3, name='jpeg_reader')

  float_caster = tf.cast(image, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  with tf.Session() as session:
    return session.run(normalized)

def load_labels(label_file):
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  return [l.rstrip().replace(" ", "-") for l in proto_as_ascii_lines]

def select_family(families):
    return families[0] if len(families) else None

def run_graph_v2(graph, image_tensor, label_path, input_layer="input", output_layer="final_result"):
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: image_tensor})
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_path)
    return [{
            "name": labels[i], 
            "score": float(results[i])
            } for i in top_k]


@hug.get('/finish', versions=1, output=hug.output_format.json)
def predict_finish(image_url: hug.types.text):
    input_layer = "input"
    output_layer = "final_result"
    image_tensor = read_tensor_from_image_file(image_url,
                                input_height=128,
                                input_width=128,
                                input_mean=128,
                                input_std=128)
    family_label_path = "model/color-families/labels.txt"
    families = run_graph_v2(MODELS['color-families'], image_tensor, family_label_path)
    selected_family = select_family(families)
    if not selected_family:
        return {
            "color_families": [],
            "finishes": []
        }
    finishes = []
    if selected_family:
        finish_model_base = os.path.join("model", selected_family["name"])
        finish_label_path = os.path.join(finish_model_base, "labels.txt")
        finishes = run_graph_v2(MODELS[selected_family['name']], image_tensor, finish_label_path)
    return {
        "color_families": families,
        "finishes": finishes
    }

@hug.get('/finish', versions=2, output=hug.output_format.json)
def predict_finish(image_url: hug.types.text):
    if not "finishes" in MODELS:
        return {
            "color_families": [],
            "finishes": []
        }
    finishes_graph = MODELS["finishes"]
    input_layer = "input"
    output_layer = "final_result"
    image_tensor = read_tensor_from_image_file(image_url,
                                input_height=128,
                                input_width=128,
                                input_mean=128,
                                input_std=128)
    family_label_path = "model/finishes/labels.txt"
    finishes = run_graph_v2(finishes_graph, image_tensor, family_label_path)
    return {
        "color_families": [],
        "finishes": finishes
    }

@hug.get('/hot-dog-not-hot-dog', versions=1, output=hug.output_format.json)
def predict_finish(image_url: hug.types.text):
    if not "hot-dog-not-hot-dog" in MODELS:
        return {
            "color_families": [],
            "finishes": []
        }
    finishes_graph = MODELS["hot-dog-not-hot-dog"]
    input_layer = "input"
    output_layer = "final_result"
    image_tensor = read_tensor_from_image_file(image_url,
                                input_height=128,
                                input_width=128,
                                input_mean=128,
                                input_std=128)
    family_label_path = "model/hot-dog-not-hot-dog/labels.txt"
    finishes = run_graph_v2(finishes_graph, image_tensor, family_label_path)
    return {
        "color_families": [],
        "finishes": finishes
    }
