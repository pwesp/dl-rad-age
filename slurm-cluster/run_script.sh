#!/bin/bash

# Create python environment for custom packages while mainting site packages from docker container
python -m venv .venv --system-site-packages
source .venv/bin/activate

# Install custom packages (sepcified in setup.cfg and/or setup.py) into your python environment using development mode
python -m pip install -e .

# Run python script with config
python scripts/train_age_estimation_net.py $1