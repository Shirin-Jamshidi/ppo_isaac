#!/usr/bin/env bash

# -----------------------------
# Isaac Gym + PPO environment
# -----------------------------

# Initialize conda (important after reboot)
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the correct environment
conda activate isaacgym

# Isaac Gym installation path
export ISAAC_GYM_PATH=$HOME/isaacgym

# Make Isaac Gym importable
export PYTHONPATH=$ISAAC_GYM_PATH/python:$PYTHONPATH

# Fix Isaac Gym + conda Python shared library issue
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Optional: reduce GPU noise on shared machine
export CUDA_VISIBLE_DEVICES=0

echo "✅ Isaac Gym environment restored"
echo "Python: $(python --version)"
