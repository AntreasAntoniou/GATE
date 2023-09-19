#!/usr/bin/env python

from setuptools import find_packages, setup

print(f"Installing {find_packages()}")
setup(
    name="gate",
    version="0.8.10",
    description="A minimal, stateless, machine learning research template for PyTorch",
    author="Antreas Antoniou",
    author_email="iam@antreas.io",
    packages=find_packages(),
)
