import time
from tqdm import tqdm
import torch
from utils.color import colorstr
from utils.make_dataset import get_data_loader
from utils.get_model import get_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, opts):
    setup(rank, world_size)

    train_loader, val_loader = get_data_loader(
        opts.num_frames, opts.target_size, opts.num_classes, opts.batch_size)

    train_sampler = DistributedSampler(
        train_loader.dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(
        val_loader.dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=opts.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=opts.batch_size, sampler=val_sampler)

    model = get_model(opts.model, opts.num_classes,
                      opts.num_frames, opts.target_size)

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, train_loader, val_loader, criterion,
                                optimizer, rank, num_epochs=opts.epochs)

    if rank == 0:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        model_class_name = trained_model.__class__.__name__
        torch.save(trained_model.state_dict(),
                   f'{output_dir}/{model_class_name}_model.pth')
        torch.save(optimizer.state_dict(),
                   f'{output_dir}/{model_class_name}_optimizer.pth')

    cleanup()


def train_model(model, train_loader, val_loader, criterion, optimizer, rank, num_epochs=10):
    print(colorstr("white", "bold",
          f"Training {model.__class__.__name__} model !"))
    print(colorstr("red", "bold",
          f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."))

    since = time.time()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(colorstr(f"Epoch {epoch}/{num_epochs - 1}:"))

        for phase in ["train", "val"]:
            if phase == "train":
                print(
                    colorstr("yellow", "bold", "\n%20s" + "%15s" * 3)
                    % ("Training: ", "gpu_mem", "loss", "acc")
                )
                model.train()
            else:
                print(
                    colorstr("green", "bold", "\n%20s" + "%15s" * 3)
                    % ("Eval: ", "gpu_mem", "loss", "acc")
                )
                model.eval()

            running_items = 0.0
            running_loss = 0.0
            running_corrects = 0

            data_loader = train_loader if phase == "train" else val_loader

            _phase = tqdm(
                data_loader,
                total=len(data_loader),
                bar_format="{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}",
                unit="batch",
            )

            for videos, labels in _phase:
                videos = videos.to(rank)
                labels = labels.to(rank)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(videos)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_items += videos.size(0)
                running_loss += loss.item() * videos.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects.double() / running_items

                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3f}GB"

                desc = ("%35s" + "%15.6g" * 2) % (mem, epoch_loss, epoch_acc)
                _phase.set_description(desc)

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    history["best_epoch"] = epoch

                print(f"Best val Acc: {best_val_acc:.4f}")

    time_elapsed = time.time() - since
    history["INFO"] = (
        "Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs. Best val Acc: {:.4f}".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            time_elapsed % 60,
            num_epochs,
            best_val_acc,
        )
    )

    return model
