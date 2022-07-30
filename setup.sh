#!/bin/bash

pip install -r requirements.txt

apt-get install git-lfs
git-lfs install
git config --global credential.helper store

huggingface-cli login
wandb login