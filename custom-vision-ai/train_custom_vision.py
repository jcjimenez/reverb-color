#!/usr/bin/env python3

from argparse import ArgumentTypeError
from glob import glob
from itertools import groupby
from math import floor
from os import listdir
from os.path import join, split
from random import sample
from requests.exceptions import HTTPError

from custom_vision_client import TrainingClient
from custom_vision_client import TrainingConfig


class AtMost(object):
    def __init__(self, max_value):
        self._max_value = max_value

    def __call__(self, value):
        try:
            number = int(value)
        except ValueError:
            raise ArgumentTypeError('Not a number: {}'.format(value))

        if number > self._max_value:
            raise ArgumentTypeError('Value should be less or equal to {}'
                                    .format(self._max_value))

        return number


def group_by_labels(image_path):
    return split(image_path)[0]


class AllTrainingData(object):
    def __init__(self, data_root, labels=None):
        self._data_root = data_root
        self._labels = set(labels or [])

    def get_images(self):
        for label in self._get_labels():
            images = glob(join(self._data_root, label, '**', '*.jpg'), recursive=True)
            grouped = groupby(images,group_by_labels)
            for group, images in grouped:
                labels = group.split('/')[1:]
                yield labels, [i for i in images]

    def _get_labels(self):
        labels = listdir(self._data_root)
        if self._labels:
            labels = [label for label in labels if label in self._labels]
        return labels


class SampledTrainingData(object):
    def __init__(self, data_root, max_labels, max_images, labels=None):
        self._data_root = data_root
        self._max_labels = max_labels
        self._max_images = max_images
        self._labels = labels or []

    def get_images(self):
        top_labels = sorted(
            AllTrainingData(self._data_root, self._labels).get_images(),
            key=lambda label_images: len(label_images[1]),
            reverse=True)[:self._max_labels]

        total_images = sum(len(images) for _, images in top_labels)

        for labels, images in top_labels:
            sample_importance = len(images) / total_images
            sample_size = int(floor(sample_importance * self._max_images))
            if sample_size < len(images):
                sampled_images = sample(images, sample_size)
            else:
                sampled_images = list(images)
            yield labels, sampled_images


def train(azure_region, project_name, training_key, data_dir,
          max_labels, max_images, labels):

    training_data = SampledTrainingData(data_dir, max_labels, max_images, labels)
    client = TrainingClient(TrainingConfig(azure_region, training_key))
    client._training_batch_size = 50
    project_id = client.create_project(project_name).Id

    for labels, images in training_data.get_images():
        print('Got labels {} with {} images'.format(labels, len(images)))
        for label in labels:
            try:
                client.create_tag(project_id, label)
            except HTTPError:
                continue # hack - can't create label more than once
        client.add_training_images(project_id, images, *labels)

    model_id = client.trigger_training(project_id).Id
    yield project_id, model_id


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--project_name', required=True)
    parser.add_argument('--training_key', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--labels', type=str, default='')
    parser.add_argument('--max_labels', type=AtMost(100), default=100)
    parser.add_argument('--max_images', type=AtMost(10000), default=1000)
    parser.add_argument('--azure_region', type=str, default='southcentralus')
    args = parser.parse_args()

    training_info = train(
        args.azure_region, args.project_name, args.training_key,
        args.data_dir, args.max_labels, args.max_images,
        args.labels and args.labels.split(','))

    for project_id, model_id in training_info:
        print('Training project {} model {}'.format(project_id, model_id))


if __name__ == '__main__':
    _main()
