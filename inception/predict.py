from glob import glob
from os import getenv
from os.path import join
from shutil import unpack_archive

import hug
import requests
import tensorflow as tf


def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def download_file(url, to_path='', chunk_size=1024):
    local_filename = to_path or url.split('/')[-1]

    response = requests.get(url, stream=True)
    with open(local_filename, 'wb') as fobj:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fobj.write(chunk)


if not glob(join('model', '*')):
    download_file(getenv('TF_MODEL_URL'), 'model.tgz')
    unpack_archive('model.tgz')


MODELS = {
    'color-families': tf.Graph(),
    'green': tf.Graph()
}

for k, graph in MODELS.items():
    with graph.as_default():
        load_graph(join("model", k, "graph.pb"))


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(image_data, labels, input_layer_name, output_layer_name, num_top_predictions):
    """
        Returns a list of predicted classes that looks like this:
        [
            {name:"red-family", score:0.8},
            {name:"pink-family", score:0.4},
        ]
    """
    with tf.Session() as sess:
        """
        Feed the image_data as input to the graph.
           predictions will contain a two-dimensional array, where one
           dimension represents the input image count, and the other has
           predictions per class
        """
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        response = list()
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            response.append({
                "name": human_string,
                "score": float(score)
            })
        return response


def select_family(families):
    if len(families):
        return families[0]
    else:
        return None


@hug.get('/finish', versions=1, output=hug.output_format.json)
def predict_finish(image_url: hug.types.text):
    with MODELS['color-families'].as_default():
        family_labels = load_labels("model/color-families/labels.txt")
        image_data = requests.get(image_url).content
        families = run_graph(image_data, family_labels, "DecodeJpeg/contents:0", "final_result:0", 5)
        selected_family = select_family(families)

    if not selected_family:
        return {
            "color_families": [],
            "finishes": []
        }

    finishes = []

    with MODELS[selected_family['name']].as_default():
        if select_family:
            finish_model_base = join("model", selected_family["name"])
            finish_labels = load_labels(join(finish_model_base, "labels.txt"))
            finishes = run_graph(image_data, finish_labels, "DecodeJpeg/contents:0", "final_result:0", 5)

    return {
        "color_families": families,
        "finishes": finishes
    }
