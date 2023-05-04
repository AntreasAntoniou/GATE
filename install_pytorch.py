import os
import platform
import subprocess
import sys
from typing import Tuple


def has_nvidia_gpu() -> bool:
    """
    Check if the system has an NVIDIA GPU available.

    Returns:
        bool: True if NVIDIA GPU is available, False otherwise.
    """
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_platform_and_gpu() -> Tuple[str, bool]:
    """
    Determine the platform (operating system) and whether an NVIDIA GPU is available.

    Returns:
        Tuple[str, bool]: A tuple containing the platform and a boolean indicating GPU availability.
    """
    system = platform.system()
    gpu = has_nvidia_gpu()
    return system, gpu


def install_pytorch():
    """
    Install PyTorch, torchvision, and torchaudio based on the current platform and GPU availability.
    """
    system, gpu = get_platform_and_gpu()

    if system == "Linux":
        index_url = (
            "https://download.pytorch.org/whl/cu118"
            if gpu
            else "https://download.pytorch.org/whl/cpu"
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                index_url,
            ]
        )
    elif system == "Darwin":
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )
    elif system == "Windows":
        index_url = "https://download.pytorch.org/whl/cu118" if gpu else None
        if index_url:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    index_url,
                ]
            )
        else:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                ]
            )
    else:
        raise ValueError(f"Unsupported platform: {system}")


if __name__ == "__main__":
    install_pytorch()
