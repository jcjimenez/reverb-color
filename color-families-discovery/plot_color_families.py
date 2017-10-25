from matplotlib import use
use('Agg')

from matplotlib.pyplot import savefig
from pandas import read_csv
from seaborn import jointplot


def plot(csv_path, col1, col2, save_to):
    df = read_csv(csv_path)
    x = df[col1]
    y = df[col2]

    jointplot(x, y, stat_func=None,
              xlim=(x.min(), x.max()), ylim=(y.min(), y.max()),
              s=[1 for _ in range(len(df))], marker='.')

    savefig(save_to)


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--col1', type=str, required=True)
    parser.add_argument('--col2', type=str, required=True)
    parser.add_argument('--save_to', type=str, required=True)
    args = parser.parse_args()

    plot(args.csv_path, args.col1, args.col2, args.save_to)


if __name__ == '__main__':
    _main()
