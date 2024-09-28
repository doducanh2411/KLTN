import time
from tqdm import tqdm
import torch
from utils.color import colorstr


def train_model(model, train_loader, val_loader, criterion, optimizer, device="cuda", num_epochs=10):
    """
    Trains the given model using the provided data loaders, criterion, and optimizer.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to be used during training.
        optimizer (torch.optim.Optimizer): Optimizer to be used for updating model parameters.
        device (str, optional): Device to run the training on, either "cuda" or "cpu". Default is "cuda".
        num_epochs (int, optional): Number of epochs to train the model. Default is 10.
    """
    print(colorstr("cyan", "bold",
          f"Training {model.__class__.__name__} model !"))
    print(colorstr("cyan", "bold",
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

    model.to(device)

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
                videos = videos.to(device)
                labels = labels.to(device)

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

                # Cập nhật best validation accuracy
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
