from matplotlib import use
use('Agg')

from matplotlib.pyplot import imshow
from matplotlib.pyplot import savefig
from pandas import read_csv


def plot(csv_path, save_to):
    df = read_csv(csv_path)
    pixels = (df.as_matrix() / 255.).reshape((1, len(df), 3))

    imshow(pixels)

    savefig(save_to)


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--save_to', type=str, required=True)
    args = parser.parse_args()

    plot(args.csv_path, args.save_to)


if __name__ == '__main__':
    _main()
