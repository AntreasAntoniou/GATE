#!/usr/bin/env python

from setuptools import find_packages, setup


def read_requirements(file):
    """
    Read the contents of a file and return them as a list of strings.

    Args:
        file (str): The path to the file to be read.

    Returns:
        list: A list of strings representing the contents of the file.

    """
    with open(file, encoding="utf-8") as f:
        return f.read().splitlines()


print(f"Installing {find_packages()}")
setup(
    name="gate",
    version="0.9.5",
    description=(
        "GATE: Generalization After Transfer Evaluation Engine - A framework"
        " for efficient model encoder evaluation"
    ),
    long_description=(
        "GATE is a comprehensive benchmarking suite and software framework"
        " designed for evaluating the adaptability and performance of model"
        " encoders across various tasks, domains, and modalities. It offers"
        " three tiers of benchmarks optimized for different GPU budgets,"
        " automated dataset handling, and rich evaluation metrics."
    ),
    author="Antreas Antoniou",
    author_email="iam@antreas.io",
    url="https://github.com/AntreasAntoniou/GATE",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "gate = gate.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=(
        "machine learning, deep learning, model evaluation, benchmarking,"
        " transfer learning"
    ),
)
