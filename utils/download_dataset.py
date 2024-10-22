import subprocess
import zipfile
import os


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
