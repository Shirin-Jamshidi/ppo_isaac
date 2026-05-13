# 1. Clone repo    
git clone https://github.com/Shirin-Jamshidi/ppo_isaac.git      
cd ~/ppo_isaac   

# 2. Create environment and set environment variables   
conda create -n isaacgym python=3.8 -y     
conda activate isaacgym    
source ~/isaacgym_env.sh

# 3. Run training
python train.py
