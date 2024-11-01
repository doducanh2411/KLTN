import time
import torch
import os
import json
import wandb
from tqdm import tqdm
import torch_xla.core.xla_model as xm
from dataset import VideoDataset
from utils.get_norm import get_norm
from utils.get_model import get_model
from utils.color import colorstr
from torch.utils.data import DataLoader


def get_data_loader(model_name, num_frames, num_classes, batch_size, multimodal):
    cpus = os.cpu_count()
    dataset_path = os.path.join("/kaggle/input/tikharm-dataset")

    train_path = os.path.join(dataset_path, 'Dataset', 'train')
    val_path = os.path.join(dataset_path, 'Dataset', 'val')
    transform = get_norm(model_name)

    train_captions = None
    val_captions = None

    if multimodal:
        train_captions = os.path.join(
            os.getcwd(), 'captions', 'train_caption.json')
        val_captions = os.path.join(
            os.getcwd(), 'captions', 'val_caption.json')

    train_dataset = VideoDataset(train_path, num_frames=num_frames,
                                 transform=transform, num_classes=num_classes, captions=train_captions)
    val_dataset = VideoDataset(val_path, num_frames=num_frames,
                               transform=transform, num_classes=num_classes, captions=val_captions)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus)

    return train_loader, val_loader


def train(opts):
    wandb.init(project="KLTN")
    train_loader, val_loader = get_data_loader(
        opts.model, opts.num_frames, opts.num_classes, opts.batch_size, opts.multimodal)

    model = get_model(opts.model, opts.num_classes, opts.num_frames)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    device = xm.xla_device()  # Set TPU device

    trained_model, history = train_model_tpu(
        model, train_loader, val_loader, criterion, optimizer, device=device, num_epochs=opts.epochs, multimodal=opts.multimodal)

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    model_class_name = trained_model.__class__.__name__

    xm.save(trained_model.state_dict(),
            f'{output_dir}/{model_class_name}_model.pth')
    xm.save(optimizer.state_dict(),
            f'{output_dir}/{model_class_name}_optimizer.pth')

    history_file = f'{output_dir}/{model_class_name}_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f)

    wandb.finish()


def train_model_tpu(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, multimodal=False):
    print(colorstr("white", "bold",
          f"Training {model.__class__.__name__} model on TPU!"))
    # Wrap the model for TPU parallelism
    model.to(device)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    def train_loop_fn(loader, phase):
        running_loss, running_corrects, running_items = 0.0, 0, 0
        loader = tqdm(
            loader, desc=f"{phase.capitalize()} Progress", unit="batch")

        for batch in loader:
            # Set the mode here based on phase
            if phase == "train":
                model.train()
            else:
                model.eval()

            if multimodal:
                videos, labels, captions = batch
                captions = [caption for caption in captions]
            else:
                videos, labels, _ = batch

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                if multimodal:
                    outputs = model(videos, captions)
                else:
                    outputs = model(videos)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    xm.optimizer_step(optimizer)  # Sync TPU cores

            running_loss += loss.item() * videos.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_items += videos.size(0)

            epoch_loss = running_loss / running_items
            epoch_acc = running_corrects.double() / running_items
            loader.set_postfix(loss=epoch_loss, acc=epoch_acc.item())

        return running_loss / running_items, running_corrects.double() / running_items

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            # Remove global model.train() and model.eval()
            loader = train_loader if phase == "train" else val_loader
            epoch_loss, epoch_acc = train_loop_fn(loader, phase)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                history["best_epoch"] = epoch

            wandb.log({f"{phase}_loss": epoch_loss,
                      f"{phase}_acc": epoch_acc.item(), "epoch": epoch})

            print(
                f"{phase.capitalize()} - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
    return model, history
