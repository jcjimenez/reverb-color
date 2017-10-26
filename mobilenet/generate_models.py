#!/usr/bin/env python

import sys
import os
import argparse
import glob
import tempfile
import shutil
import subprocess

def generate_model_paths(jpg_path):
    results = []
    components = jpg_path.split(os.sep)
    while len(components) > 2:
        model_index = len(components) - 3
        model = os.sep.join(components[:model_index + 1])
        results.append(model)
        del components[model_index]
    return results

def identify_models(dirs_list):
    models = set()
    for root, dirs, files in os.walk(dirs_list):
        for file in files:
            if file.endswith(".jpg"):
                for model_path in generate_model_paths(os.path.join(root, file)):
                    models.add(model_path)
    return models

def generate_models(model_paths, architecture, output_dir):
    temporary_dir = tempfile.gettempdir()
    training_data_basename = "tf_training_data_%s" % str(uuid.uuid4())
    training_data_dir = os.path.join(temporary_dir, training_data_basename)
    os.makedirs(training_data_dir, exist_ok=False)
    os.makedirs(output_dir, exist_ok=False)

    for model_path in model_paths:
        print ("Generating model for: %s" % model_path)
        model_name = os.path.basename(model_path)
        model_temp = os.path.join(training_data_dir, model_name)
        os.makedirs(model_temp, exist_ok=True)
        for object_class_name in os.listdir(model_path):
            object_class_path = os.path.join(model_path, object_class_name)
            if not os.path.isdir(object_class_path):
                continue
            print("Object class name: %s" % object_class_name)
            object_class_temp = os.path.join(model_temp, object_class_name)
            os.makedirs(object_class_temp, exist_ok=True)
            for file_source in glob.iglob("%s/**/*.jpg" % object_class_path, recursive=True):
                shutil.copy(file_source, object_class_temp)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        model_output_graph = os.path.join(model_output_dir, "graph.pb")
        model_output_labels = os.path.join(model_output_dir, "labels.txt")
        retrain_command = ["python3", "retrain.py", "--architecture", architecture, "--image_dir", model_temp, "--output_graph", model_output_graph, "--output_labels", model_output_labels]
        retrain_command_str = " ".join(retrain_command)
        print("Invoking retrain.py for %s: %s" % (model_path, retrain_command_str))
        subprocess.call(retrain_command)

parser = argparse.ArgumentParser(description= 'Generate Inception or Mobilenet models')
parser.add_argument("--image-dir", dest='image_directory', required=True)
parser.add_argument("--architecture", dest='architecture', type=str, default="mobilenet_0.25_128_quantized")
parser.add_argument("--output_dir", dest="output_dir", type=str, default="model")
args = parser.parse_args()

image_dir = args.image_directory
identified_models = identify_models(image_dir)
print(identified_models)

generate_models(identified_models, args.architecture, args.output_dir)
