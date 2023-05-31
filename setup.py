#!/usr/bin/env python

from setuptools import find_packages, setup

# main_requirements = [
#     "torch",
#     "torchvision",
#     "torchaudio",
#     "torchgeometry",
#     "torchmetrics",
#     "accelerate",
#     "datasets",
#     "transformers",
#     "orjson",
#     "gh",
#     "tabulate",
#     "nvitop",
#     "hydra-zen",
#     "neptune",
#     "pytorchvideo",
#     "torchtyping",
#     "h5py",
#     "wandb",
#     "rich",
#     "opencv-python",
#     "scipy",
#     "scikit-learn",
#     "soundfile",
#     "monai",
#     "nibabel",
#     "decord",
#     "kaggle",
#     "natsort",
#     "gdown",
#     "gpath",
#     "cython",
#     "gulpio2 @ git+https://github.com/kiyoon/GulpIO2",
#     "timm @ git+https://github.com/huggingface/pytorch-image-models.git",
#     "tali @ git+https://github.com/AntreasAntoniou/TALI.git",
# ]

# # "learn2learn @ git+https://github.com/mlguild/learn2learn.git#egg=package[dev]",
# dev_requirements = [
#     "pytest",
#     "isort",
#     "jupyterlab",
#     "black",
#     "sphinx",
#     "sphinx_rtd_theme",
#     "sphinx-autodoc-typehints",
#     "sphinx-material",
#     "matplotlib",
#     "pytest-rich",
#     "pytest-sugar",
# ]

print(f"Installing {find_packages()}")
setup(
    name="gate",
    version="0.8.6",
    description="A minimal, stateless, machine learning research template for PyTorch",
    author="Antreas Antoniou",
    author_email="iam@antreas.io",
    packages=find_packages(),
    # install_requires=main_requirements,
    # extras_require={
    #     "dev": dev_requirements,
    # },
)
