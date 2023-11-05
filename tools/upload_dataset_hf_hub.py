import logging

import fire
from huggingface_hub import HfApi

logging.getLogger("huggingface_hub").setLevel(logging.NOTSET)
api = HfApi()


def upload_dataset(
    repo_id: str = "Antreas/GeoLifeCLEF2023",
    dataset_path: str = "/disk/scratch_fast1/data/GeoLife-CLEF-2023/",
):
    api.upload_folder(
        folder_path=dataset_path,
        repo_id=repo_id,
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )


if __name__ == "__main__":
    fire.Fire(upload_dataset)
