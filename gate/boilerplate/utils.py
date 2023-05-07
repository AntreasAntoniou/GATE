import logging
import shutil
import signal
from functools import wraps

import accelerate
import torch
from huggingface_hub import (
    create_repo,
    hf_hub_download,
    login,
    snapshot_download,
)
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich.traceback import install
from rich.tree import Tree

import os
import pathlib
import shutil
import yaml
from omegaconf import OmegaConf
from huggingface_hub import (
    login,
    create_repo,
    download_model_with_name,
    snapshot_download,
    HfApi,
)
from typing import Optional, Tuple, Dict

import os
import os.path
import pathlib
from typing import Any, Dict, Union

import orjson as json


def get_logger(
    name=__name__, logging_level: str = None, set_rich: bool = False
) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

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

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup

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
    logger.debug("Debugging")
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

        branch.add(Syntax(branch_content, "yaml"))

    return tree


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
    hf_repo_path, hf_cache_dir, model_name, download_only_if_finished=False
):
    if not pathlib.Path(
        pathlib.Path(hf_cache_dir) / "checkpoints" / f"{model_name}"
    ).exists():
        pathlib.Path(
            pathlib.Path(hf_cache_dir) / "checkpoints" / f"{model_name}"
        ).mkdir(parents=True, exist_ok=True)

    config_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        filename="config.yaml",
        repo_type="model",
    )

    config_path = pathlib.Path(hf_cache_dir) / "config.yaml"

    shutil.copy(
        pathlib.Path(config_filepath),
        config_path,
    )

    trainer_state_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="trainer_state.pt",
        repo_type="model",
    )

    trainer_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "trainer_state.pt"
    )

    shutil.copy(
        pathlib.Path(trainer_state_filepath),
        trainer_path,
    )
    logger.info(
        f"Trainer state copied to {trainer_path} from {trainer_state_filepath}."
    )

    if download_only_if_finished:
        state_dict = torch.load(trainer_path)["state_dict"]["eval"][0][
            "auc-macro"
        ]
        global_step_list = list(state_dict.keys())
        if len(global_step_list) < 40:
            return False

    optimizer_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="optimizer.bin",
        repo_type="model",
    )

    model_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="pytorch_model.bin",
        repo_type="model",
    )

    random_states_filepath = hf_hub_download(
        repo_id=hf_repo_path,
        cache_dir=pathlib.Path(hf_cache_dir),
        resume_download=True,
        subfolder=f"checkpoints/{model_name}",
        filename="random_states_0.pkl",
        repo_type="model",
    )

    try:
        scaler_state_filepath = hf_hub_download(
            repo_id=hf_repo_path,
            cache_dir=pathlib.Path(hf_cache_dir),
            resume_download=True,
            subfolder=f"checkpoints/{model_name}",
            filename="scaler.pt",
            repo_type="model",
        )
    except Exception as e:
        scaler_state_filepath = None

    target_optimizer_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "optimizer.bin"
    )

    shutil.copy(
        pathlib.Path(optimizer_filepath),
        target_optimizer_path,
    )

    target_model_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "pytorch_model.bin"
    )

    shutil.copy(
        pathlib.Path(model_filepath),
        target_model_path,
    )

    random_states_path = (
        pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}"
        / "random_states_0.pkl"
    )

    shutil.copy(
        pathlib.Path(random_states_filepath),
        random_states_path,
    )

    if scaler_state_filepath is not None:
        scaler_path = (
            pathlib.Path(hf_cache_dir)
            / "checkpoints"
            / f"{model_name}"
            / "scaler.pt"
        )

        shutil.copy(
            pathlib.Path(scaler_state_filepath),
            scaler_path,
        )

    return {
        "root_filepath": pathlib.Path(hf_cache_dir)
        / "checkpoints"
        / f"{model_name}",
        "optimizer_filepath": target_optimizer_path,
        "model_filepath": target_model_path,
        "random_states_filepath": random_states_path,
        "trainer_state_filepath": trainer_path,
        "config_filepath": config_path,
    }


def create_hf_model_repo(cfg: Any) -> str:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    print(f"Creating repo {cfg.hf_repo_path}")
    return create_repo(
        cfg.hf_repo_path, repo_type="model", exist_ok=True, private=True
    )


def create_directories(cfg: Any) -> None:
    pathlib.Path(cfg.hf_cache_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg.hf_cache_dir / "checkpoints").mkdir(
        parents=True, exist_ok=True
    )


def upload_config_files(cfg: Any, hf_repo_path: str) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_json_path = save_json(
        cfg.hf_cache_dir / "config.json", config_dict, overwrite=True
    )
    config_yaml_path = cfg.hf_cache_dir / "config.yaml"
    hf_api = HfApi(token=os.environ["HF_TOKEN"])

    with open(config_yaml_path, "w") as file:
        yaml.dump(config_dict, file)

    for filepath, path_in_repo in [
        (config_json_path, "config.json"),
        (config_yaml_path, "config.yaml"),
    ]:
        hf_api.upload_file(
            repo_id=hf_repo_path,
            path_or_fileobj=filepath.as_posix(),
            path_in_repo=path_in_repo,
        )


def get_checkpoint_dict(files: Dict) -> Dict[int, str]:
    return {
        int(file.split("/")[-2].split("_")[-1]): "/".join(file.split("/")[:-1])
        for file in files
        if "checkpoints/ckpt" in file
    }


def download_checkpoint(cfg: Any, model_name: str) -> Tuple[pathlib.Path, str]:
    logger.info(
        f"Downloading checkpoint '{model_name}' from Hugging Face hub 👨🏻‍💻"
    )
    path_dict = download_model_with_name(
        cfg.hf_repo_path, cfg.hf_cache_dir, model_name
    )
    logger.info(f"Downloaded checkpoint to {cfg.hf_cache_dir}")
    return path_dict["root_filepath"], cfg.hf_repo_path


def create_hf_model_repo_and_download_maybe(
    cfg: Any,
) -> Tuple[Optional[pathlib.Path], str]:
    repo_url = create_hf_model_repo(cfg)
    create_directories(cfg)
    upload_config_files(cfg, repo_url)

    hf_api = HfApi(token=os.environ["HF_TOKEN"])
    files = hf_api.list_repo_files(repo_id=cfg.hf_repo_path)
    ckpt_dict = get_checkpoint_dict(files)

    if cfg.resume_from_checkpoint:
        return download_checkpoint(cfg, cfg.resume_from_checkpoint)
    elif cfg.resume:
        latest_ckpt = ckpt_dict[max(ckpt_dict.keys())].split("/")[-1]
        return download_checkpoint(cfg, latest_ckpt)
    else:
        print(f"Created repo {cfg.hf_repo_path}, {cfg.hf_cache_dir}")
        return None, repo_url


def count_files_recursive(directory: str) -> int:
    file_count = 0

    for root, _, files in os.walk(directory):
        file_count += len(files)

    return file_count
