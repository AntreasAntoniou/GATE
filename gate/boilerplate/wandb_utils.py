import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import wandb

logger = logging.getLogger(name=__name__)


def visualize_video_with_labels(video, logits, labels, name):
    logger.info(
        f"name: {name}, mean: {video_data.mean()}, std: {video_data.std()}, min: {video_data.min()}, max: {video_data.max()}, dtype: {video_data.dtype}"
    )
    video_data = video.cpu() - video.min()
    video_data = video_data / video_data.max()
    video_data = video_data * 255

    if isinstance(logits, torch.Tensor):
        logits = logits.cpu()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()

    logger.info(
        f"name: {name}, mean: {video_data.mean()}, std: {video_data.std()}, min: {video_data.min()}, max: {video_data.max()}, dtype: {video_data.dtype}"
    )

    # Log the video and labels to wandb
    wandb_video_data_dict = {}
    for idx, (video_clip, logit, label) in enumerate(
        zip(video_data, logits, labels)
    ):
        wandb_video_data_dict[f"{name}/video/{idx}"] = wandb.Video(
            video_clip,
            fps=1,
            format="gif",
            caption=f"Predicted: {logit}, True: {label}",
        )

    return wandb_video_data_dict


def log_wandb_3d_volumes_and_masks(
    volumes: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_idx_to_description: Optional[dict] = None,
    prefix: str = "general",
) -> None:
    """
    Function to visualize MRI volumes using Weights & Biases (wandb).

    Args:
        input_volumes (torch.Tensor): Input volumes with shape (b, s, c, h, w).
        predicted_volumes (torch.Tensor): Predicted volumes with shape (b, s, h, w).
        label_volumes (torch.Tensor): Label volumes with shape (b, s, h, w).
        label_idx_to_description (dict): Dictionary mapping label indices to descriptions.
        run_name (str): Name of the wandb run.

    Returns:
        None
    """

    # Convert PyTorch tensors to NumPy arrays
    input_volumes_np = normalize_image(volumes.float()).cpu()
    predicted_volumes_np = logits.long().cpu()
    label_volumes_np = labels.long().cpu()

    if len(input_volumes_np.shape) == 4:
        input_volumes_np = input_volumes_np.unsqueeze(0)
    if len(predicted_volumes_np.shape) == 3:
        predicted_volumes_np = predicted_volumes_np.unsqueeze(0)
    if len(label_volumes_np.shape) == 3:
        label_volumes_np = label_volumes_np.unsqueeze(0)

    for data, name in zip(
        [input_volumes_np, predicted_volumes_np, label_volumes_np],
        ["Input", "Predicted", "Label"],
    ):
        if data is input_volumes_np:
            assert (
                len(data.shape) == 5
            ), f"{name} volumes should be 5D in the shape of (b, s, c, h, w)"
        else:
            assert (
                len(data.shape) == 4
            ), f"{name} volumes should be 4D in the shape of (b, s, h, w)"

    # If no label description is provided, use a default mapping of indices to themselves
    if label_idx_to_description is None:
        unique_labels = np.unique(label_volumes_np)
        label_idx_to_description = {
            label: str(label) for label in unique_labels
        }

    # Define a helper function to create a wandb.Image with masks
    def wb_mask(bg_img, pred_mask, true_mask):
        return wandb.Image(
            bg_img,
            masks={
                "prediction": {
                    "mask_data": pred_mask,
                    "class_labels": label_idx_to_description,
                },
                "ground truth": {
                    "mask_data": true_mask,
                    "class_labels": label_idx_to_description,
                },
            },
        )

    image_mask_list = []
    # Log volumes to wandb
    for i in range(input_volumes_np.shape[0]):
        bg_image = (
            create_montage(input_volumes_np[i]).permute([2, 0, 1]).float()
        )
        prediction_mask = create_montage(predicted_volumes_np[i]).long()
        true_mask = create_montage(label_volumes_np[i]).long()

        bg_image = T.ToPILImage()(bg_image)
        prediction_mask = prediction_mask.numpy()
        true_mask = true_mask.numpy()
        image_mask_list.append(wb_mask(bg_image, prediction_mask, true_mask))

    return {f"{prefix}/medical_segmentation_episode": image_mask_list}


def log_wandb_masks(
    images: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_idx_to_description: Dict[int, str],
    prefix: str = "general",
):
    def wb_mask(bg_img, pred_mask, true_mask):
        return wandb.Image(
            bg_img,
            masks={
                "prediction": {
                    "mask_data": pred_mask,
                    "class_labels": label_idx_to_description,
                },
                "ground truth": {
                    "mask_data": true_mask,
                    "class_labels": label_idx_to_description,
                },
            },
        )

    mask_list = []
    for i in range(len(images)):
        bg_image = T.ToPILImage()(normalize_image(images[i]))
        prediction_mask = logits[i].detach().cpu().numpy().astype(np.uint8)
        true_mask = labels[i].detach().cpu().numpy().astype(np.uint8)

        mask_list.append(wb_mask(bg_image, prediction_mask, true_mask))

    return {f"{prefix}/segmentation_episode": mask_list}


def log_wandb_images(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    prefix: str = "general",
):
    episode_list = []
    for i in range(images.shape[0]):
        image = images[i]
        reconstruction = reconstructions[i]
        normalized_image = normalize_image(image)
        normalized_reconstruction = normalize_image(reconstruction)
        ae_episode = torch.cat(
            [normalized_image, normalized_reconstruction], dim=2
        )
        ae_episode = wandb.Image(ae_episode)
        episode_list.append(ae_episode)

    return {f"{prefix}/autoencoder_episode": episode_list}


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    min_val = torch.min(image)
    max_val = torch.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)

    return normalized_image


def create_montage(arr: np.ndarray) -> np.ndarray:
    """
    Create a 2D montage from a 3D or 4D numpy array.

    Args:
        arr (np.ndarray): Input array with shape (s, h, w) or (s, c, h, w).

    Returns:
        np.ndarray: 2D montage.
    """

    # Check the shape of the input array
    assert len(arr.shape) in [
        3,
        4,
    ], "Input array should be 3D in the shape of (s, h, w) or 4D in the shape of (s, c, h, w)"

    # Get the shape of the input array
    if len(arr.shape) == 3:
        s, h, w = arr.shape
        c = None
    else:
        s, c, h, w = arr.shape

    # Compute the new height and width
    h_new = w_new = math.ceil(math.sqrt(s))

    # Create an empty array to hold the montage
    montage = (
        np.empty((h_new * h, w_new * w, c))
        if c is not None
        else np.empty((h_new * h, w_new * w))
    )

    # Fill the montage with slices from the input array
    for i in range(h_new):
        for j in range(w_new):
            idx = i * w_new + j
            if idx < s:
                if c is not None:
                    if isinstance(arr, torch.Tensor):
                        montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = (
                            arr[idx].permute(1, 2, 0).numpy()
                        )
                    else:
                        montage[
                            i * h : (i + 1) * h, j * w : (j + 1) * w
                        ] = arr[idx].transpose(1, 2, 0)
                else:
                    montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = arr[
                        idx
                    ]
            else:
                # Fill any extra entries with empty arrays
                montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = (
                    np.zeros((h, w, c)) if c is not None else np.zeros((h, w))
                )

    return torch.tensor(montage)


def visualize_volume(item, name):
    input_volumes = item["image"]
    input_volumes = input_volumes.float().unsqueeze(0).unsqueeze(0)
    predicted_volumes = item["labels"].float().unsqueeze(0)
    label_volumes = item["labels"].float().unsqueeze(0)

    print(
        f"Input volumes shape: {input_volumes.shape}, dtype: {input_volumes.dtype}, min: {input_volumes.min()}, max: {input_volumes.max()}, mean: {input_volumes.mean()}, std: {input_volumes.std()}"
    )
    print(
        f"Predicted volumes shape: {predicted_volumes.shape}, dtype: {predicted_volumes.dtype}, min: {predicted_volumes.min()}, max: {predicted_volumes.max()}, mean: {predicted_volumes.mean()}, std: {predicted_volumes.std()}"
    )
    print(
        f"Label volumes shape: {label_volumes.shape}, dtype: {label_volumes.dtype}, min: {label_volumes.min()}, max: {label_volumes.max()}, mean: {label_volumes.mean()}, std: {label_volumes.std()}"
    )

    # Start a Weights & Biases run
    run = wandb.init(
        project="gate-visualization", job_type="visualize_dataset"
    )

    # Visualize the data
    wandb.log(
        log_wandb_3d_volumes_and_masks(
            F.interpolate(
                input_volumes.reshape(
                    -1,
                    input_volumes.shape[-3],
                    input_volumes.shape[-2],
                    input_volumes.shape[-1],
                ),
                size=(256, 256),
                mode="bicubic",
            ).reshape(*input_volumes.shape[:-2] + (256, 256)),
            predicted_volumes.long(),
            label_volumes.long(),
            prefix=name,
        )
    )

    # Finish the run
    run.finish()
