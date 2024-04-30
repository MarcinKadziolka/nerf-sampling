import pickle
import matplotlib.pyplot as plt
import argparse


def main(args):
    filename = args.filename

    figx = pickle.load(open(f"{filename}.fig.pickle", "rb"))
    figx.show()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    main(args)
