import logging
import math
import os
import os.path
import pathlib
import shutil
import signal
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Union

import accelerate
import huggingface_hub
import numpy as np
import orjson as json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from rich.logging import RichHandler
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.traceback import install
from rich.tree import Tree

import wandb
from gate.config.variables import HF_OFFLINE_MODE

int_or_str = Union[int, str]


def get_logger(
    name=__name__,
    logging_level: Optional[int_or_str] = None,
    set_rich: bool = False,
) -> logging.Logger:
    """Initializes a python command line logger with nice defaults."""

    logger = logging.getLogger(name)

    logging_level = logging_level or logging.INFO

    logger.setLevel(logging_level)

    if set_rich:
        ch = RichHandler()

        # create formatter
        formatter = logging.Formatter("%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    install()

    return logger


def get_hydra_config(logger_level: str = "INFO"):
    return dict(
        job_logging=dict(
            version=1,
            formatters=dict(
                simple=dict(
                    level=logger_level,
                    format="%(message)s",
                    datefmt="[%X]",
                )
            ),
            handlers=dict(
                rich={
                    "class": "rich.logging.RichHandler",
                    # "formatter": "simple",
                }
            ),
            root={"handlers": ["rich"], "level": logger_level},
            disable_existing_loggers=False,
        ),
        hydra_logging=dict(
            version=1,
            formatters=dict(
                simple=dict(
                    level=logging.CRITICAL,
                    format="%(message)s",
                    datefmt="[%X]",
                )
            ),
            handlers={
                "rich": {
                    "class": "rich.logging.RichHandler",
                    # "formatter": "simple",
                }
            },
            root={"handlers": ["rich"], "level": logging.CRITICAL},
            disable_existing_loggers=False,
        ),
        run={
            "dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"
        },
        sweep={
            "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
            "subdir": "${hydra.job.num}",
        },
    )


def timeout(timeout_secs: int):
    def wrapper(func):
        @wraps(func)
        def time_limited(*args, **kwargs):
            # Register an handler for the timeout
            def handler(signum, frame):
                raise Exception(f"Timeout for function '{func.__name__}'")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)

            # Define a timeout for your function
            signal.alarm(timeout_secs)

            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                logging.error(f"Exploded due to time out on {args, kwargs}")
                raise exc
            finally:
                # disable the signal alarm
                signal.alarm(0)

            return result

        return time_limited

    return wrapper


def demo_logger():
    logger = get_logger(__name__)

    logger.info("Hello World")
    # logger.debug("Debugging")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")
    logger.exception("Exception")


def set_seed(seed: int):
    accelerate.utils.set_seed(seed)


def pretty_config(
    config: DictConfig,
    resolve: bool = True,
):
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        # theme list ['default', 'emacs', 'friendly', 'friendly_grayscale',
        # 'colorful', 'autumn', 'murphy', 'manni', 'material', 'monokai',
        # 'perldoc', 'pastie', 'borland', 'trac', 'native', 'fruity', 'bw',
        # 'vim', 'vs', 'tango', 'rrt', 'xcode', 'igor', 'paraiso-light',
        # 'paraiso-dark', 'lovelace', 'algol', 'algol_nu', 'arduino',
        # 'rainbow_dash', 'abap', 'solarized-dark', 'solarized-light',
        # 'sas', 'staroffice', 'stata', 'stata-light', 'stata-dark',
        # 'inkpot', 'zenburn', 'gruvbox-dark', 'gruvbox-light',
        # 'dracula', 'one-dark', 'lilypond', 'nord', 'nord-darker',
        # 'github-dark']
        branch.add(Syntax(branch_content, "yaml", theme="one-dark"))

    return tree


def pretty_print_dictionary(dictionary: Dict[str, Any]) -> None:
    rprint(Pretty(dictionary))


def save_json(
    filepath: Union[str, pathlib.Path], dict_to_store: Dict, overwrite=True
):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param dict_to_store: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    if overwrite and filepath.exists():
        filepath.unlink(missing_ok=True)

    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as json_file:
        json_file.write(json.dumps(dict_to_store))
    return filepath


def load_json(filepath: Union[str, pathlib.Path]):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    with open(filepath, "rb") as json_file:
        dict_to_load = json.loads(json_file.read())

    return dict_to_load


logger = get_logger(name=__name__)


def download_model_with_name(
    hf_repo_path: str,
    hf_cache_dir: str,
    model_name: str,
    download_only_if_finished: bool = False,
    local_checkpoint_store_dir: Optional[str] = None,
) -> Dict[str, pathlib.Path]:
    """
    Download model checkpoint files given the model name from Hugging Face hub.

    :param hf_repo_path: The Hugging Face repository path
    :param hf_cache_dir: The cache directory to store downloaded files
    :param model_name: The model name to download
    :param download_only_if_finished: Download only if the model training is finished (optional)
    :return: A dictionary with the filepaths of the downloaded files
    """

    if local_checkpoint_store_dir is not None:
        ckpt_dir = pathlib.Path(local_checkpoint_store_dir) / model_name
        validated_ckpt_dir = True
        if ckpt_dir.exists():
            path_dict = {
                "trainer_state_filepath": ckpt_dir / "trainer_state.pt",
                "optimizer_filepath": ckpt_dir / "optimizer.bin",
                "model_filepath": ckpt_dir / "pytorch_model.bin",
                "random_states_filepath": ckpt_dir / "random_states_0.pkl",
                "root_filepath": ckpt_dir,
                "validation_passed": True,
            }
            for key, value in path_dict.items():
                if isinstance(value, pathlib.Path):
                    print(f"Checking {key} exists: {value.exists()}")
                    if not value.exists():
                        validated_ckpt_dir = False
                        break
            if validated_ckpt_dir:
                return path_dict

    if not HF_OFFLINE_MODE:

        def download_and_copy(
            filename: str,
            target_path: pathlib.Path,
            subfolder: str = f"checkpoints/{model_name}",
        ) -> None:
            file_path = huggingface_hub.hf_hub_download(
                repo_id=hf_repo_path,
                cache_dir=pathlib.Path(hf_cache_dir),
                resume_download=True,
                subfolder=subfolder,
                filename=filename,
                repo_type="model",
            )
            shutil.copy(pathlib.Path(file_path), target_path)

        checkpoint_dir = (
            pathlib.Path(hf_cache_dir) / "checkpoints" / model_name
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        file_mapping = {
            "trainer_state.pt": "trainer_state_filepath",
            "optimizer.bin": "optimizer_filepath",
            "pytorch_model.bin": "model_filepath",
            "random_states_0.pkl": "random_states_filepath",
            "scaler.pt": "scaler_filepath",
        }

        downloaded_files = {}
        invalid_download = False
        for filename, key in file_mapping.items():
            try:
                target_path = checkpoint_dir / filename
                download_and_copy(filename, target_path)
                downloaded_files[key] = target_path
            except Exception as e:
                if filename != "scaler.pt":
                    invalid_download = True
                logger.info(f"Error downloading {filename}: {e}")
        # Handle config.yaml separately
        config_target_path = pathlib.Path(hf_cache_dir) / "config.yaml"
        download_and_copy("config.yaml", config_target_path, subfolder="")
        downloaded_files["config_filepath"] = config_target_path

        if download_only_if_finished:
            state_dict = torch.load(
                downloaded_files["trainer_state_filepath"]
            )["state_dict"]["eval"][0]["auc-macro"]
            if len(state_dict.keys()) < 40:
                return {}

        downloaded_files["root_filepath"] = checkpoint_dir
        downloaded_files["validation_passed"] = not invalid_download

        return downloaded_files

    return None


def create_hf_model_repo(hf_repo_path: str) -> str:
    huggingface_hub.login(
        token=os.environ["HF_TOKEN"], add_to_git_credential=True
    )
    print(f"Creating repo {hf_repo_path}")
    return huggingface_hub.create_repo(
        hf_repo_path, repo_type="model", exist_ok=True, private=True
    )


def create_directories(hf_cache_dir: str) -> None:
    pathlib.Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(hf_cache_dir) / "checkpoints").mkdir(
        parents=True, exist_ok=True
    )
    return hf_cache_dir


def store_config_yaml(
    cfg: Any,
    hf_cache_dir: str,
):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_yaml_path = pathlib.Path(hf_cache_dir) / "config.yaml"

    with open(config_yaml_path, "w") as file:
        yaml.dump(config_dict, file)

    return config_yaml_path


def upload_config_files(hf_repo_path: str, config_yaml_path) -> None:
    hf_api = huggingface_hub.HfApi(token=os.environ["HF_TOKEN"])

    hf_api.upload_file(
        repo_id=hf_repo_path,
        path_or_fileobj=config_yaml_path.as_posix(),
        path_in_repo="config.yaml",
    )


def get_checkpoint_dict(files: Dict) -> Dict[int, str]:
    return {
        int(file.split("/")[-2].split("_")[-1]): "/".join(file.split("/")[:-1])
        for file in files
        if "checkpoints/ckpt" in file
    }


def download_checkpoint(
    hf_repo_path: str,
    ckpt_identifier: str,
    hf_cache_dir: str,
    local_checkpoint_store_dir: Optional[str] = None,
) -> Tuple[pathlib.Path, str]:
    logger.info(
        f"Downloading checkpoint {hf_repo_path}/{ckpt_identifier} from Hugging Face hub 👨🏻‍💻"
    )
    if isinstance(ckpt_identifier, int):
        ckpt_identifier = f"ckpt_{ckpt_identifier}"

    path_dict = download_model_with_name(
        hf_repo_path,
        hf_cache_dir,
        ckpt_identifier,
        local_checkpoint_store_dir=local_checkpoint_store_dir,
    )
    logger.info(f"Downloaded checkpoint to {hf_cache_dir}")
    return path_dict


def create_hf_model_repo_and_download_maybe(
    cfg: Any,
    hf_cache_dir: str,
    hf_repo_path: str,
    resume_from_checkpoint: str,
    resume: bool,
) -> Tuple[Optional[pathlib.Path], str]:
    if not HF_OFFLINE_MODE:
        create_hf_model_repo(hf_repo_path)

    hf_cache_dir = create_directories(hf_cache_dir)
    config_yaml_path = store_config_yaml(cfg, hf_cache_dir)

    if not HF_OFFLINE_MODE:
        upload_config_files(
            hf_repo_path=hf_repo_path, config_yaml_path=config_yaml_path
        )

    checkpoint_store_dir = (
        pathlib.Path(cfg.current_experiment_dir) / cfg.exp_name / "checkpoints"
    )

    if not HF_OFFLINE_MODE:
        hf_api = huggingface_hub.HfApi(token=os.environ["HF_TOKEN"])
        remote_files = hf_api.list_repo_files(repo_id=hf_repo_path)
        remote_ckpt_dict = get_checkpoint_dict(remote_files)
        remote_ckpt_names = sorted(list(remote_ckpt_dict.keys()), reverse=True)
        logger.info(
            f"remote ckpt dict: {remote_ckpt_dict}, remote ckpt names: {remote_ckpt_names}"
        )
    else:
        remote_ckpt_names = []

    local_files = [str(file) for file in checkpoint_store_dir.glob("*")]
    local_ckpt_dict = {
        int(file.split("checkpoints/ckpt_")[1]): file for file in local_files
    }
    local_ckpt_names = sorted(list(local_ckpt_dict.keys()), reverse=True)

    logger.info(
        f"local ckpt dict: {local_ckpt_dict}, local ckpt names: {local_ckpt_names}"
    )

    mixed_ckpt_list = remote_ckpt_names + local_ckpt_names
    mixed_ckpt_list = sorted(list(set(mixed_ckpt_list)), reverse=True)

    if len(mixed_ckpt_list) == 0:
        return None

    if resume_from_checkpoint:
        return download_checkpoint(
            hf_cache_dir=hf_cache_dir,
            hf_repo_path=hf_repo_path,
            ckpt_identifier=resume_from_checkpoint,
            local_checkpoint_store_dir=checkpoint_store_dir,
        )
    elif resume:
        valid_model_downloaded = False
        idx = 0

        while not valid_model_downloaded:
            if len(mixed_ckpt_list) < idx + 1:
                logger.info("No valid checkpoint found. starting from scratch")
                return None

            download_dict = download_checkpoint(
                hf_cache_dir=hf_cache_dir,
                hf_repo_path=hf_repo_path,
                ckpt_identifier=mixed_ckpt_list[idx],
                local_checkpoint_store_dir=checkpoint_store_dir,
            )
            valid_model_downloaded = download_dict["validation_passed"]
            idx += 1
        return download_dict
    else:
        print(f"Created repo {hf_repo_path}, {hf_cache_dir}")
        return None


def download_model_checkpoint_from_hub(
    hf_repo_path: str,
    hf_cache_dir: str,
    checkpoint_identifier: Optional[str] = None,
    get_latest: bool = False,
    local_checkpoint_store_dir: Optional[str] = None,
) -> Tuple[Optional[pathlib.Path], str]:
    if get_latest and checkpoint_identifier is not None:
        raise ValueError(
            "Only one of `get_latest` and `checkpoint_identifier` can be set to True"
        )

    hf_api = huggingface_hub.HfApi(token=os.environ["HF_TOKEN"])
    files = hf_api.list_repo_files(repo_id=hf_repo_path)
    ckpt_dict = get_checkpoint_dict(files)

    if len(ckpt_dict) == 0:
        return None

    if get_latest:
        latest_ckpt = ckpt_dict[max(ckpt_dict.keys())].split("/")[-1]
        return download_checkpoint(
            hf_cache_dir=hf_cache_dir,
            hf_repo_path=hf_repo_path,
            ckpt_identifier=latest_ckpt,
            local_checkpoint_store_dir=local_checkpoint_store_dir,
        )

    if checkpoint_identifier is not None:
        return download_checkpoint(
            hf_cache_dir=hf_cache_dir,
            hf_repo_path=hf_repo_path,
            ckpt_identifier=checkpoint_identifier,
            local_checkpoint_store_dir=local_checkpoint_store_dir,
        )

    return None


def count_files_recursive(directory: str) -> int:
    file_count = 0

    for root, _, files in os.walk(directory):
        file_count += len(files)

    return file_count


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    min_val = torch.min(image)
    max_val = torch.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    # print(
    #     f"min: {torch.min(normalized_image)}, max: {torch.max(normalized_image)}, mean: {torch.mean(normalized_image)}, std: {torch.std(normalized_image)}"
    # )
    # normalized_image = normalized_image * 255.0
    return normalized_image


def log_wandb_images(
    experiment_tracker: Any,
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    global_step: int = 0,
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

    experiment_tracker.log(
        {f"{prefix}/autoencoder_episode": episode_list},
        step=global_step,
    )


def log_wandb_masks(
    experiment_tracker: Any,
    images: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_idx_to_description: Dict[int, str],
    global_step: int = 0,
    prefix: str = "general",
):
    if experiment_tracker is None:
        wandb.init()
        experiment_tracker = wandb

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

    experiment_tracker.log(
        {f"{prefix}/segmentation_episode": mask_list}, step=global_step
    )


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


def log_wandb_3d_volumes_and_masks(
    volumes: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_idx_to_description: Optional[dict] = None,
    global_step: int = 0,
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

    wandb.log(
        {f"{prefix}/medical_segmentation_episode": image_mask_list},
        step=global_step,
    )


def visualize_volume(item, name):
    input_volumes = item["image"]
    input_volumes = input_volumes.float().unsqueeze(0).unsqueeze(0)
    predicted_volumes = item["labels"].float().unsqueeze(0)
    label_volumes = item["labels"].float().unsqueeze(0)

    # predicted_volumes[predicted_volumes == -1] = 10
    # label_volumes[label_volumes == -1] = 10

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
    log_wandb_3d_volumes_and_masks(
        F.interpolate(
            input_volumes.reshape(-1, input_volumes.shape[-3], 512, 512),
            size=(256, 256),
            mode="bicubic",
        ).reshape(*input_volumes.shape[:-2] + (256, 256)),
        predicted_volumes.long(),
        label_volumes.long(),
        prefix=name,
    )

    # Finish the run
    run.finish()


def visualize_video_with_labels(video, logits, labels, name):
    video_data = video.cpu() * 255.0

    if isinstance(logits, torch.Tensor):
        logits = logits.cpu()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()

    print(
        f"name: {name}, mean: {video_data.mean()}, std: {video_data.std()}, min: {video_data.min()}, max: {video_data.max()}, dtype: {video_data.dtype}"
    )

    # Log the video and labels to wandb
    for idx, (video_clip, logit, label) in enumerate(
        zip(video_data, logits, labels)
    ):
        wandb.log(
            {
                f"{name}/video/{idx}": wandb.Video(
                    video_clip,
                    fps=1,
                    format="gif",
                    caption=f"Predicted: {logit}, True: {label}",
                ),
            }
        )
