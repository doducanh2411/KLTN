import subprocess
import zipfile
import os
from dataset import VideoDataset
from torch.utils.data import DataLoader


def install_dataset():
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "anhoangvo/tikharm-dataset"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)

        dataset_zip = "tikharm-dataset.zip"
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall("tikharm-dataset")
        print(f"Dataset extracted to 'tikharm-dataset' directory.")

        os.remove(dataset_zip)
        print(f"Removed the zip file: {dataset_zip}")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
    except zipfile.BadZipFile as e:
        print(f"Error occurred while unzipping: {e}")


def get_data_loader(num_frames, target_size, num_classes, batch_size):
    cpus = os.cpu_count()

    dataset_path = os.path.join(os.getcwd(), "tikharm-dataset")
    train_path = os.path.join(dataset_path, 'Dataset', 'train')
    val_path = os.path.join(dataset_path, 'Dataset', 'val')

    train_dataset = VideoDataset(
        train_path, num_frames=num_frames, target_size=target_size, num_classes=num_classes)
    val_dataset = VideoDataset(
        val_path, num_frames=num_frames, target_size=num_frames, num_classes=num_classes)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpus)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpus)

    return train_loader, val_loader
