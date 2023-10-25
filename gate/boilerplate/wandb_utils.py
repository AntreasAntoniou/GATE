from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
import wandb

from gate.boilerplate.utils import create_montage, get_logger, normalize_image

logger = get_logger(__name__)


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
