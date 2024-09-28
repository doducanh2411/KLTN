from utils.get_model import get_model
from utils.make_dataset import install_dataset, get_data_loader
from utils.opts import parse_opts
from train import train_model
import torch

if __name__ == '__main__':
    opts = parse_opts()

    if opts.make_dataset:
        install_dataset()
    elif opts.train:
        train_loader, val_loader = get_data_loader(
            opts.num_frames, opts.target_size, opts.num_classes, opts.batch_size)

        model = get_model(opts.model, opts.num_classes, opts.num_frames)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trained_model = train_model(model, train_loader, val_loader, criterion,
                                    optimizer, device, num_epochs=opts.epochs)
