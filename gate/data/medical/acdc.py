import os
import pathlib
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import requests
import torch
from torch.utils.data import Dataset, random_split
from tqdm import tqdm


def download_and_extract_file(extract_to: str) -> pathlib.Path:
    """
    Downloads and extracts the ACDC dataset to the specified directory.

    :param extract_to: Path to the directory where the dataset will be extracted.
    :return: The path to the extracted dataset.
    """
    local_filename = pathlib.Path(extract_to) / "ACDC.tar.gz"
    extract_to = pathlib.Path(extract_to) / "ACDC"

    # Download the dataset only if it hasn't been downloaded yet
    if not local_filename.exists():
        url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192

        # Save the downloaded dataset to a local file
        with open(local_filename, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=str(local_filename),
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    # Extract the dataset
    with zipfile.ZipFile(local_filename, "r") as zip_ref:
        zip_ref.extractall(path=extract_to)

    return pathlib.Path(extract_to)


class ACDCDataset(Dataset):
    def __init__(self, root_dir: str, mode: str = "train"):
        self.root_dir = root_dir
        self.mode = mode

        # Download and extract the dataset
        self.root_dir = download_and_extract_file(self.root_dir)

        # Set the data directory based on the mode
        self.data_dir = pathlib.Path(self.root_dir) / "database" / f"{mode}ing"

        # Get the list of patients
        self.patients = sorted(
            [
                d
                for d in self.data_dir.iterdir()
                if d.is_dir() and d.name.startswith("patient")
            ]
        )

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        patient_dir = self.patients[idx]

        # Get the image and label file paths
        img_files = sorted(
            [
                f
                for f in patient_dir.glob("*_frame*.nii.gz")
                if not f.name.endswith("_gt.nii.gz")
            ]
        )
        label_files = sorted(patient_dir.glob("*_frame*_gt.nii.gz"))
        img_label_pairs = list(zip(img_files, label_files))

        four_d_img_file = patient_dir / f"{patient_dir.name}_4d.nii.gz"
        four_d_img = nib.load(four_d_img_file).get_fdata()
        four_d_img_tensor = torch.from_numpy(four_d_img).float()

        frame_data = []
        for img_file, label_file in img_label_pairs:
            img = nib.load(img_file).get_fdata()
            label = nib.load(label_file).get_fdata()

            img_tensor = torch.from_numpy(img).float()
            label_tensor = torch.from_numpy(label).long()

            frame_data.append({"img": img_tensor, "label": label_tensor})

        return {"four_d_img": four_d_img_tensor, "frame_data": frame_data}


def build_acdc_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> Dataset:
    """
    Build an ACDC dataset.

    Args:
        set_name: The name of the dataset split to return ("train", "val", or "test").
        data_dir: The directory where the dataset cache is stored.

    Returns:
        A Dataset object containing the specified dataset split.
    """
    # Create a generator with the specified seed
    rng = torch.Generator().manual_seed(42)
    train_data = ACDCDataset(root_dir=data_dir, mode="train")

    dataset_length = len(train_data)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    # Split the dataset into train and validation sets using the generator
    train_data, val_data = random_split(
        train_data, [train_length, val_length], generator=rng
    )

    test_data = ACDCDataset(root_dir=data_dir, mode="test")

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


def main():
    dataset_path = os.environ.get("PYTEST_DIR")
    dataset_path = pathlib.Path(dataset_path)
    dataset = ACDCDataset(root_dir=dataset_path)

    for i, sample in enumerate(dataset):
        four_d_img = sample["four_d_img"]
        frame_data = sample["frame_data"]

        print(f"Patient {i + 1}: 4D Image shape: {four_d_img.shape}")

        for j, frame in enumerate(frame_data):
            img = frame["img"]
            label = frame["label"]
            print(
                f"  Frame {j + 1}: Image shape: {img.shape}, Label shape: {label.shape}"
            )


if __name__ == "__main__":
    main()
