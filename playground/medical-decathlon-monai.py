import logging
import os

import dotenv
import monai
import torch.nn.functional as F
import wandb

from gate.boilerplate.utils import enrichen_logger
from gate.boilerplate.wandb_utils import log_wandb_3d_volumes_and_masks

logger = logging.getLogger(__name__)
logger = enrichen_logger(logger)
# Load environment variables from .env file
dotenv.load_dotenv(
    dotenv_path="/disk/scratch_fast1/aantoni2/GATE/secrets/setup_variables.env"
)


def visualize_volume(
    item, prefix: str, image_key="image", labels_key="labels"
):
    input_volumes = item[image_key].unsqueeze(0)
    input_volumes = input_volumes.float()
    predicted_volumes = item[labels_key].float()
    label_volumes = item[labels_key].float()

    # predicted_volumes[predicted_volumes == -1] = 10
    # label_volumes[label_volumes == -1] = 10

    logger.info(
        f"Input volumes shape: {input_volumes.shape}, dtype: {input_volumes.dtype}, min: {input_volumes.min()}, max: {input_volumes.max()}, mean: {input_volumes.mean()}, std: {input_volumes.std()}"
    )
    logger.info(
        f"Predicted volumes shape: {predicted_volumes.shape}, dtype: {predicted_volumes.dtype}, min: {predicted_volumes.min()}, max: {predicted_volumes.max()}, mean: {predicted_volumes.mean()}, std: {predicted_volumes.std()}"
    )
    logger.info(
        f"Label volumes shape: {label_volumes.shape}, dtype: {label_volumes.dtype}, min: {label_volumes.min()}, max: {label_volumes.max()}, mean: {label_volumes.mean()}, std: {label_volumes.std()}"
    )

    # Start a Weights & Biases run

    target_size = 384
    # Visualize the data
    return log_wandb_3d_volumes_and_masks(
        F.interpolate(
            input_volumes.view(
                -1,
                input_volumes.shape[-3],
                input_volumes.shape[-2],
                input_volumes.shape[-1],
            ),
            size=(target_size, target_size),
            mode="bicubic",
        ).view(*input_volumes.shape[:-2] + (target_size, target_size)),
        F.interpolate(
            predicted_volumes.view(
                -1,
                predicted_volumes.shape[-3],
                predicted_volumes.shape[-2],
                predicted_volumes.shape[-1],
            ),
            size=(target_size, target_size),
            mode="nearest-exact",
        )
        .view(*predicted_volumes.shape[:-2] + (target_size, target_size))
        .long(),
        F.interpolate(
            label_volumes.view(
                -1,
                predicted_volumes.shape[-3],
                predicted_volumes.shape[-2],
                predicted_volumes.shape[-1],
            ),
            size=(target_size, target_size),
            mode="nearest-exact",
        )
        .view(*predicted_volumes.shape[:-2] + (target_size, target_size))
        .long(),
        prefix=prefix,
    )


def test_build_gate_visualize_dataset():
    wandb.init(project="gate_visualization_pytest")
    # for task_option in TaskOptions:
    task_name = "Task01_BrainTumour"
    gate_dataset = monai.apps.DecathlonDataset(
        root_dir=os.environ.get("DATASET_DIR"),
        task=task_name,
        section="training",
        download=True,
        seed=0,
        val_frac=0.2,
        num_workers=64,
        progress=True,
        cache_num=0,
        cache_rate=0.0,
        copy_cache=False,
        as_contiguous=True,
        runtime_cache=False,
    )

    for item in gate_dataset:
        logger.info(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["label"] is not None, "Label should not be None"
        item["image"] = item["image"].permute([2, 3, 0, 1])
        item["label"] = item["label"].permute([2, 0, 1])
        wandb.log(
            visualize_volume(
                item, prefix=f"{task_name}/train", labels_key="label"
            ),
        )
        break


if __name__ == "__main__":
    test_build_gate_visualize_dataset()
