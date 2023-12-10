import multiprocessing as mp
import os

import datasets
import dotenv
import huggingface_hub
import torch
from rich import print as rprint
from tqdm.auto import tqdm

from gate.data.image.classification.happywhale import (
    build_dataset,
    get_label_dict,
)

dotenv.load_dotenv(dotenv_path="secrets/setup_variables.env")


def report_summary_statistics(x):
    tensor = torch.tensor(x)
    mean = tensor.mean()
    std = tensor.std()
    max = tensor.max()
    min = tensor.min()
    rprint(f"mean: {mean}, std: {std}, max: {max}, min: {min}")
    return x


def get_dataset(set_name: str):
    gate_dataset = build_dataset(data_dir=os.environ.get("DATASET_DIR"))[
        set_name
    ]
    label_dict = get_label_dict(gate_dataset.dataset)

    for idx, item in enumerate(gate_dataset):
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"

        yield {
            "image": item["image"],
            "species": item["labels"]["species"],
            "species_name": label_dict["species"][item["labels"]["species"]],
            "individual": item["labels"]["individual"],
            "individual_name": label_dict["individual"][
                item["labels"]["individual"]
            ],
        }


if __name__ == "__main__":
    set_name_list = ["train", "val", "test"]

    def dataset_generator(value, set_name):
        dataset = value(set_name=set_name)

        for item in tqdm(dataset):
            yield item

    dataset_dict = {}
    for set_name in set_name_list:
        hf_dataset = datasets.Dataset.from_generator(
            generator=dataset_generator,
            cache_dir=os.getenv("DATASET_DIR"),
            gen_kwargs={
                "set_name": set_name,
                "value": get_dataset,
            },
            keep_in_memory=False,
            num_proc=mp.cpu_count(),
            writer_batch_size=15,
        )
        dataset_dict[set_name] = hf_dataset

    hf_dataset_dict_full = datasets.DatasetDict(dataset_dict)

    dataset_repo = f"GATE-engine/happy-whale-dolphin-classification"
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
