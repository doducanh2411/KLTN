from utils.opts import parse_opts
from utils.download_dataset import install_dataset
from train import train

if __name__ == '__main__':
    opts = parse_opts()

    if opts.make_dataset:
        install_dataset()
    elif opts.train:
        train(opts)
