from utils.get_model import get_model
from utils.make_dataset import install_dataset, get_data_loader
from utils.opts import parse_opts
from train import train


if __name__ == '__main__':
    opts = parse_opts()

    if opts.make_dataset:
        install_dataset()
    elif opts.train:
        train(opts)
