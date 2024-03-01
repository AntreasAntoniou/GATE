#!/usr/bin/env python

from setuptools import find_packages, setup

print(f"Installing {find_packages()}")
setup(
    name="gate",
    version="0.9.3",
    description="A minimal, stateless, machine learning research template for PyTorch",
    author="Antreas Antoniou",
    author_email="iam@antreas.io",
    entry_points={
        "console_scripts": [
            "gate = gate.cli:main",
        ],
    },
    packages=find_packages(),
)
