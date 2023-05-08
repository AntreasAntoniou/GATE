#!/usr/bin/env python

from setuptools import find_packages, setup

main_requirements = [
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "accelerate",
    "datasets",
    "transformers",
    "orjson",
    "gh",
    "tabulate",
    "nvitop",
    "hydra-zen",
    "neptune",
    "pytorchvideo",
    "torchtyping",
    "h5py",
    "wandb",
    "rich",
    "opencv-python",
    "scipy",
    "segmentation-models-pytorch @ git+https://github.com/qubvel/segmentation_models.pytorch.git",
    "soundfile",
    "gulpio2 @ git+https://github.com/kiyoon/GulpIO2",
    "monai",
    "nibabel",
    "natsort",
]

dev_requirements = [
    "pytest",
    "isort",
    "jupyterlab",
    "black",
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx-autodoc-typehints",
    "sphinx-material",
    "matplotlib",
]

print(f"Installing {find_packages()}")
# TODO: Automate pip install for pytorch deps depending on platform and GPU availability --extra-index-url https://download.pytorch.org/whl/cu118
setup(
    name="gate",
    version="0.8.0",
    description="A minimal, stateless, machine learning research template for PyTorch",
    author="Antreas Antoniou",
    author_email="iam@antreas.io",
    packages=find_packages(),
    install_requires=main_requirements,
    extras_require={
        "dev": dev_requirements,
    },
)
