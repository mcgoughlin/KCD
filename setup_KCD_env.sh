#!/bin/bash

#SBATCH --account phwq4930-renal-canc
#SBATCH --qos epsrc
#SBATCH --time 0-24:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 1

set -x

module purge; module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3

CONDA_ENV_PATH="/bask/projects/p/phwq4930-renal-canc/conda_env/KCD_env"

# Create the environment. Only required once.
conda create --yes --prefix "${CONDA_ENV_PATH}"
conda init bash
source ~/.bashrc

# Activate the environment
conda activate "${CONDA_ENV_PATH}"
# Choose your version of Python
conda install --yes python=3.10

# Continue to install any further items as required.
# For example:
conda install --yes numpy pandas pip
conda install --yes -c conda-forge nibabel scikit-learn scikit-image
conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install --yes matplotlib
conda install --yes -c simpleitk simpleitk
pip install -e . 
