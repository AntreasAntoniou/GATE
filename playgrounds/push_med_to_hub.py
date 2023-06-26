from dataclasses import dataclass
import os
import multiprocessing as mp
import datasets
import numpy as np
import torch
import torchvision.transforms as T
from PIL.Image import LANCZOS
from rich import print as rprint
from tqdm.auto import tqdm
from monai.apps import DecathlonDataset
import monai.transforms as mT

dataset_root = os.environ["PYTEST_DIR"]


# split_names_list = ["train", "validation", "test"]
def report_summary_statistics(x):
    tensor = torch.tensor(x)
    mean = tensor.mean()
    std = tensor.std()
    max = tensor.max()
    min = tensor.min()
    rprint(f"mean: {mean}, std: {std}, max: {max}, min: {min}")
    return x


@dataclass
class TaskOptions:
    BrainTumour: str = "Task01_BrainTumour"
    Heart: str = "Task02_Heart"
    Liver: str = "Task03_Liver"
    Hippocampus: str = "Task04_Hippocampus"
    Prostate: str = "Task05_Prostate"
    Lung: str = "Task06_Lung"
    Pancreas: str = "Task07_Pancreas"
    HepaticVessel: str = "Task08_HepaticVessel"
    Spleen: str = "Task09_Spleen"
    Colon: str = "Task10_Colon"


task_list = vars(TaskOptions()).values()

transform = T.Compose(
    [
        mT.LoadImaged(keys=["image", "label"]),
        mT.EnsureChannelFirstd(keys=["image", "label"]),
        mT.ScaleIntensityd(keys="image"),
        mT.ToTensord(keys=["image", "label"]),
    ]
)

dataset_dict = {
    "medical_decathlon": lambda set_name, task_name: DecathlonDataset(
        dataset_root,
        task=task_name,
        section=set_name,
        transform=transform,
        download=True,
        seed=42,
        val_frac=0.0,
        num_workers=mp.cpu_count(),
        progress=True,
        copy_cache=True,
        as_contiguous=True,
        runtime_cache=False,
    ),
}

set_name_list = ["training", "test"]
if __name__ == "__main__":
    with tqdm(total=len(dataset_dict)) as pbar_dataset:
        for key, value in dataset_dict.items():
            hf_dataset_dict = dict()
            with tqdm(total=len(set_name_list)) as pbar_set_name:
                pbar_dataset.set_description(f"Processing {key}")
                print("Processing", key)
                for set_name in set_name_list:
                    pbar_set_name.set_description(f"Processing {set_name}")

                    def dataset_generator():
                        with tqdm(total=len(task_list)) as pbar_task:
                            for task_name in task_list:
                                print("Processing", task_name)
                                dataset = value(
                                    set_name=set_name, task_name=task_name
                                )
                                with tqdm(total=len(dataset)) as pbar_data:
                                    for idx, item in enumerate(dataset):
                                        pbar_data.update(1)
                                        print(item)
                                        yield item | {"task_name": task_name}

                                pbar_task.update(1)

                    hf_dataset = datasets.Dataset.from_generator(
                        dataset_generator,
                        cache_dir=dataset_root,
                        keep_in_memory=False,
                        num_proc=mp.cpu_count(),
                        writer_batch_size=50,
                    )
                    hf_dataset_dict[set_name] = hf_dataset
                    pbar_set_name.update(1)
                    pbar_set_name.set_description(f"Processing {set_name}")
            hf_dataset_dict_full = datasets.DatasetDict(hf_dataset_dict)
            completed = False
            while not completed:
                try:
                    hf_dataset_dict_full.push_to_hub(
                        repo_id=f"GATE-engine/{key}", private=False
                    )
                    completed = True
                except Exception as e:
                    print(e)
            pbar_dataset.update(1)

# dataset_dict = {
#     "omniglot": lambda set_name: l2l.vision.datasets.FullOmniglot(
#         root=dataset_root,
#         download=True,
#         transform=T.Compose(
#             [
#                 T.Resize(28, interpolation=LANCZOS),
#             ]
#         ),
#     ),
# }

# if __name__ == "__main__":
#     with tqdm(total=len(dataset_dict)) as pbar_dataset:
#         for key, value in dataset_dict.items():
#             hf_dataset_dict = dict()
#             with tqdm(total=len(["full"])) as pbar_set_name:
#                 pbar_dataset.set_description(f"Processing {key}")
#                 for set_name in ["full"]:
#                     pbar_set_name.set_description(f"Processing {set_name}")
#                     dataset = value(set_name=set_name)
#                     data_dict = {"image": [], "label": []}
#                     with tqdm(total=len(dataset)) as pbar_data:
#                         for idx, item in enumerate(dataset):
#                             data_dict["image"].append(item[0])
#                             data_dict["label"].append(item[1])
#                             pbar_data.update(1)

#                     hf_dataset = datasets.Dataset.from_dict(data_dict)
#                     hf_dataset_dict[set_name] = hf_dataset
#                     pbar_set_name.update(1)
#             hf_dataset_dict_full = datasets.DatasetDict(hf_dataset_dict)
#             completed = False
#             while not completed:
#                 try:
#                     hf_dataset_dict_full.push_to_hub(
#                         repo_id=f"Antreas/{key}", private=False
#                     )
#                     completed = True
#                 except Exception as e:
#                     print(e)
#             pbar_dataset.update(1)
