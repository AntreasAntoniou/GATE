import logging
import math
from typing import Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from PIL import Image

logger = logging.getLogger(name=__name__)


def visualize_video_with_labels(video, logits, labels, name):
    video = video.cpu()
    video_data = video - video.min()
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


def visualize_volume(item, prefix: str):
    input_volumes = item["image"].float()
    predicted_volumes = item["labels"].float()
    label_volumes = item["labels"].float()

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
    input_volumes = normalize_image(volumes.float()).cpu()
    predicted_volumes = logits.long().cpu()
    label_volumes = labels.long().cpu()

    # If no label description is provided, use a default mapping of indices to themselves
    if label_idx_to_description is None:
        unique_labels = np.unique(label_volumes)
        label_idx_to_description = {
            label: str(label) for label in unique_labels
        }

    logger.debug(
        f"unique labels: {torch.unique(label_volumes)}, frequency: {torch.bincount(label_volumes.flatten())}"
    )

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

    # Log volumes to wandb

    predicted_volumes = predicted_volumes.squeeze()
    label_volumes = label_volumes.squeeze()

    bg_image = create_montage(input_volumes).permute([2, 0, 1]).float()
    prediction_mask = create_montage(predicted_volumes).long().squeeze()
    true_mask = create_montage(label_volumes).long().squeeze()

    bg_image = T.ToPILImage()(bg_image)
    prediction_mask = prediction_mask.cpu().numpy()
    true_mask = true_mask.cpu().numpy()

    return {
        f"{prefix}/medical_segmentation_episode": [
            wb_mask(bg_image, prediction_mask, true_mask)
        ]
    }


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


def log_scatter_wandb(
    x_values: list, y_values: list, filename: str = "scatter_plot.png"
) -> wandb.Image:
    """
    Creates a scatter plot with a logarithmic y-axis using given data, saves it as a PNG,
    and returns a wandb.Image object.

    Args:
        x_values (list): List of x-values.
        y_values (list): List of y-values.
        filename (str): Filename and path where the image is to be saved.

    Returns:
        wandb.Image: A wandb.Image object.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="markers"))

    fig.update_layout(yaxis=dict(type="log", autorange=True))

    pio.write_image(fig, filename)

    return wandb.Image(filename)


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


def log_wandb_image_classification(
    images: torch.Tensor,
    labels: Union[List[int], torch.Tensor, Dict[str, int]],
    logits: Union[torch.Tensor, Dict[str, torch.Tensor]],
    prefix: str = "general",
):
    """
    Log image classification information to Weights and Biases (WandB).

    Args:
        images (torch.Tensor): Input images for classification.
        labels (Union[List[int], torch.Tensor, Dict[str, int]]): Ground truth labels.
        logits (Union[torch.Tensor, Dict[str, torch.Tensor]]): Model output logits.
        prefix (str, optional): Prefix to differentiate between different logs. Default is "general".
    """

    # Determine column names based on whether labels and logits are dictionaries
    columns = ["image"]
    if isinstance(labels, dict):
        columns.extend([f"{k}_label" for k in labels.keys()])
    else:
        columns.append("label")

    if isinstance(logits, dict):
        columns.extend([f"{k}_logits" for k in logits.keys()])
    else:
        columns.append("logit")

    predictions_table = wandb.Table(columns=columns)

    for i in range(len(images)):
        # Convert images to tensor if they are PIL Images
        image = (
            T.ToTensor()(images[i]).cpu()
            if isinstance(images[i], Image.Image)
            else images[i].cpu()
        )

        # Prepare the label and prediction data
        row_data = [wandb.Image(image)]

        if isinstance(labels, dict):
            row_data.extend(labels[k][i] for k in labels.keys())
        else:
            row_data.append(labels[i])

        if isinstance(logits, dict):
            row_data.extend(
                logits[k][i].detach().cpu().argmax() for k in logits.keys()
            )
        else:
            row_data.append(logits[i].detach().cpu().argmax())

        # Add the row to the table
        predictions_table.add_data(*row_data)

    # Log the predictions table
    return {f"{prefix}_predictions": predictions_table}


# Ensure to call this within a context where wandb.run is active.


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
        5,
    ], "Input array should be 3D in the shape of (s, h, w) or 4D in the shape of (s, c, h, w)"

    # Get the shape of the input array
    if len(arr.shape) == 3:
        s, h, w = arr.shape
        c = None
    elif len(arr.shape) == 4:
        s, c, h, w = arr.shape
    else:
        b, s, c, h, w = arr.shape
        arr = arr.reshape(-1, c, h, w)

    # Compute the new height and width
    h_new = w_new = math.ceil(math.sqrt(s))

    # Create an empty array to hold the montage
    montage = (
        torch.empty((h_new * h, w_new * w, c))
        if c is not None
        else torch.empty((h_new * h, w_new * w))
    )

    # Fill the montage with slices from the input array
    for i in range(h_new):
        for j in range(w_new):
            idx = i * w_new + j
            if idx < s:
                if c is not None:
                    if isinstance(arr, torch.Tensor):
                        montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = (
                            arr[idx].permute(1, 2, 0)
                        )
                    else:
                        montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = (
                            arr[idx].transpose(1, 2, 0)
                        )
                else:
                    montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = arr[
                        idx
                    ]
            else:
                # Fill any extra entries with empty arrays
                montage[i * h : (i + 1) * h, j * w : (j + 1) * w] = (
                    torch.zeros((h, w, c))
                    if c is not None
                    else torch.zeros((h, w))
                )

    return torch.tensor(montage)


def visualize_volume(item, prefix: str, target_size: int = 384):
    input_volumes = item["image"].float()
    predicted_volumes = item["labels"].float()
    label_volumes = item["labels"].float()

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
