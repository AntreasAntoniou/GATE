import logging
import os
from typing import List

import fire
from huggingface_hub import HfApi
from rich.logging import RichHandler
from tqdm.auto import tqdm


def delete_models_with_string_in_name(string: str | List[str]):
    huggingface_username = os.environ.get("HF_USERNAME")
    huggingface_api_token = os.environ.get("HF_TOKEN")
    client = HfApi(token=huggingface_api_token)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RichHandler()
    logger.addHandler(handler)
    models = client.list_models(author=huggingface_username)
    is_string_list = isinstance(string, list)
    for model_repo in tqdm(models):
        if is_string_list:
            if any(
                string in model_repo.__dict__["modelId"] for string in string
            ):
                logger.info(f"Deleting {model_repo.__dict__['modelId']}")
                client.delete_repo(repo_id=model_repo.__dict__["modelId"])
        elif str(string) in model_repo.__dict__["modelId"]:
            logger.info(f"Deleting {model_repo.__dict__['modelId']}")
            client.delete_repo(repo_id=model_repo.__dict__["modelId"])


if __name__ == "__main__":
    fire.Fire(delete_models_with_string_in_name)
