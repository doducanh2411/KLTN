import time
import torch
import os
import json
import wandb
from tqdm import tqdm
from dataset import VideoDataset
from utils.get_norm import get_norm
from utils.get_model import get_model
from utils.color import colorstr
from torch.utils.data import DataLoader


def get_data_loader(model_name, num_frames, num_classes, batch_size, multimodal):
    cpus = os.cpu_count()

    # dataset_path = os.path.join(os.getcwd(), "tikharm-dataset")
    dataset_path = os.path.join(
        "/kaggle/input/tikharm-dataset")  # For kaggle only

    train_path = os.path.join(dataset_path, 'Dataset', 'train')
    val_path = os.path.join(dataset_path, 'Dataset', 'val')

    transform = get_norm(model_name)

    train_captions = None
    val_captions = None

    print('multimodal', multimodal)

    if multimodal:
        train_captions = os.path.join(
            os.getcwd(), 'captions', 'train_caption.json')
        val_captions = os.path.join(
            os.getcwd(), 'captions', 'val_caption.json')

    train_dataset = VideoDataset(
        train_path, num_frames=num_frames, transform=transform, num_classes=num_classes, captions=train_captions)
    val_dataset = VideoDataset(
        val_path, num_frames=num_frames, transform=transform, num_classes=num_classes, captions=val_captions)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus)

    return train_loader, val_loader


def train(opts):
    wandb.init(project="KLTN")

    train_loader, val_loader = get_data_loader(
        opts.model, opts.num_frames, opts.num_classes, opts.batch_size, opts.multimodal)

    model = get_model(opts.model, opts.num_classes,
                      opts.num_frames)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opts.resume:
        if opts.path_to_model and os.path.exists(opts.path_to_model):
            model.load_state_dict(torch.load(
                opts.path_to_model, weights_only=True, map_location="cuda:0"))
            print(f"Loaded model state from {opts.path_to_model}")

        if opts.path_to_optimizer and os.path.exists(opts.path_to_optimizer):
            optimizer.load_state_dict(torch.load(
                opts.path_to_optimizer, weights_only=True, map_location="cuda:0"))
            print(f"Loaded optimizer state from {opts.path_to_optimizer}")

            # Move each optimizer state to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    model = model.to(device)

    trained_model, history = train_model(model, train_loader, val_loader, criterion,
                                         optimizer, device, num_epochs=opts.epochs, multimodal=opts.multimodal)

    # Save model, optimizer, and history
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    model_class_name = trained_model.__class__.__name__

    torch.save(trained_model.state_dict(),
               f'{output_dir}/{model_class_name}_model.pth')
    torch.save(optimizer.state_dict(),
               f'{output_dir}/{model_class_name}_optimizer.pth')

    history_file = f'{output_dir}/{model_class_name}_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f)

    wandb.finish()


def train_model(model, train_loader, val_loader, criterion, optimizer, device="cuda", num_epochs=10, multimodal=False):
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

            for batch in _phase:
                if multimodal:
                    videos, audio, labels, captions = batch
                    audio = audio.to(device)
                    captions = [caption[0] for caption in captions]
                else:
                    videos, _, labels, _ = batch

                videos = videos.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if multimodal:
                        outputs = model(videos, captions, audio)
                    else:
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

                wandb.log({
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc.item(),
                    "epoch": epoch,
                })

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

    return model, history
