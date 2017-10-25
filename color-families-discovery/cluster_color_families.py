from collections import OrderedDict
from glob import iglob
from os.path import join

from numpy import concatenate
from skimage.color import lab2rgb
from skimage.color import rgb2lab
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans


def find_images(data_dir):
    return iglob(join(data_dir, '**', '*.jpg'), recursive=True)


def train_kmeans(kmeans, buffer):
    training_data = concatenate(buffer, axis=0)
    kmeans.partial_fit(training_data)
    buffer.clear()


def cluster_colors(data_dir, num_clusters, batch_size):
    kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    buffer = []

    for i, image_path in enumerate(find_images(data_dir), start=1):
        image = rgb2lab(imread(image_path))
        height, width, channels = image.shape
        pixels = image.reshape((height * width, channels))
        buffer.append(pixels)
        if i % batch_size == 0:
            train_kmeans(kmeans, buffer)
    train_kmeans(kmeans, buffer)

    return kmeans


def print_clusters(kmeans):
    clusters = OrderedDict()
    for lab_cluster in kmeans.cluster_centers_:
        rgb_cluster = lab2rgb(lab_cluster.reshape((1, 1, 3)))
        rgb_cluster = tuple((rgb_cluster * 255).round().astype(int).reshape(3).tolist())
        clusters[rgb_cluster] = True

    print('r,g,b')
    for cluster in clusters:
        print(','.join(map(str, cluster)))


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    kmeans = cluster_colors(args.data_dir, args.num_clusters, args.batch_size)
    print_clusters(kmeans)


if __name__ == '__main__':
    _main()
