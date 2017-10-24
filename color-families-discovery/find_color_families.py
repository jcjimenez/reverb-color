from glob import iglob
from os.path import join

import skimage.color
from numpy import ascontiguousarray
from skimage.io import imread
from skimage.util import view_as_blocks


def find_images(data_dir):
    return iglob(join(data_dir, '**', '*.jpg'), recursive=True)


def compute_mean_colors(image_path, chunk_size, converter):
    image = converter(imread(image_path))
    height, width, channels = image.shape

    means = tuple([] for _ in range(channels))
    chunk_shape = tuple(chunk_size for _ in range(channels - 1))

    try:
        for i in range(channels):
            channel = ascontiguousarray(image[..., i])
            for image_chunk in view_as_blocks(channel, chunk_shape):
                means[i].append(image_chunk.mean())
    except ValueError:
        return []

    return zip(*means)


def find_mean_colors(data_dir, chunk_size, converter):
    for image_path in find_images(data_dir):
        for color in compute_mean_colors(image_path, chunk_size, converter):
            yield color


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--color_converter', choices=['rgb2lab', 'rgb2hsv'], default='rgb2lab')
    args = parser.parse_args()

    print(','.join(args.color_converter[-3:]))
    for color in find_mean_colors(args.data_dir, args.chunk_size, getattr(skimage.color, args.color_converter)):
        print(','.join(str(num) for num in color))


if __name__ == '__main__':
    _main()
