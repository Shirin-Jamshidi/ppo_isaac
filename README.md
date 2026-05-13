# 1. Clone repo
git clone https://github.com/Shirin-Jamshidi/ppo_isaac.git \\
cd ~/ppo_isaac

conda create -n isaacgym python=3.8 -y
conda activate isaacgym
source ~/isaacgym_env.sh
python train.py
