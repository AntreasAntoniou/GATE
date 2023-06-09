#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda init fish
source ~/.bashrc

conda create -n main python=3.10 -y
conda activate main
conda install -c conda-forge mamba -y

mamba install -c conda-forge git gh -y
mamba install -c conda-forge git-lfs -y

git lfs install
git config --global credential.helper store

python3 install_pytorch.py
pip3 install -r requirements.txt

source ~/miniconda/envs/main/bin/activate