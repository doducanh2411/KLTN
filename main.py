from utils.get_model import get_model
from utils.make_dataset import install_dataset, get_data_loader
from utils.opts import parse_opts
from train import train
import torch
import os
import torch.multiprocessing as mp

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.cuda.empty_cache()
    opts = parse_opts()

    world_size = torch.cuda.device_count()

    if opts.make_dataset:
        install_dataset()
    elif opts.train:
        mp.spawn(train, args=(world_size, opts),
                 nprocs=world_size, join=True)
