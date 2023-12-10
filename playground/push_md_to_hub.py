import multiprocessing as mp
import os
from dataclasses import dataclass

import datasets
import dotenv
import huggingface_hub
import monai
import torch
from rich import print as rprint
from tqdm.auto import tqdm

dotenv.load_dotenv(
    dotenv_path="/disk/scratch_fast1/aantoni2/GATE/secrets/setup_variables.env"
)


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


def get_dataset(task_name: str, set_name: str):
    gate_dataset = monai.apps.DecathlonDataset(
        root_dir=os.environ.get("DATASET_DIR"),
        task=task_name,
        section=set_name,
        download=True,
        seed=0,
        val_frac=0.0,
        num_workers=64,
        progress=True,
        cache_num=0,
        cache_rate=0.0,
        copy_cache=False,
        as_contiguous=True,
        runtime_cache=False,
    )

    for idx, item in enumerate(gate_dataset):
        assert item["image"] is not None, "Image should not be None"
        assert item["label"] is not None, "Label should not be None"
        item["image"] = item["image"].permute([2, 3, 0, 1])
        item["label"] = item["label"].permute([2, 0, 1])

        if idx < 5:
            yield item
        else:
            return


if __name__ == "__main__":
    set_name_list = ["training"]
    task_list = vars(TaskOptions()).values()

    def dataset_generator(value, set_name, task_name):
        print("Processing", task_name)
        dataset = value(set_name=set_name, task_name=task_name)

        for item in tqdm(dataset):
            yield item

    for task_name in task_list:
        dataset_dict = {}
        for set_name in set_name_list:
            hf_dataset = datasets.Dataset.from_generator(
                generator=dataset_generator,
                cache_dir=os.getenv("DATASET_DIR"),
                gen_kwargs={
                    "set_name": set_name,
                    "task_name": task_name,
                    "value": get_dataset,
                },
                keep_in_memory=False,
                num_proc=mp.cpu_count(),
                writer_batch_size=15,
            )
            dataset_dict[set_name] = hf_dataset

        hf_dataset_dict_full = datasets.DatasetDict(dataset_dict)

        dataset_repo = f"GATE-engine/{task_name}-v2"
        huggingface_hub.create_repo(
            repo_id=dataset_repo,
            private=False,
            exist_ok=True,
            repo_type="dataset",
        )

        completed = False

        while not completed:
            try:
                hf_dataset_dict_full.push_to_hub(
                    repo_id=dataset_repo,
                    private=False,
                    max_shard_size="2GB",
                )
                completed = True
            except Exception as e:
                print(e)
