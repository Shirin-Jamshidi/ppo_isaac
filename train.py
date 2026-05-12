# from isaacgym import gymapi
# import torch
# import torch.optim as optim

# from model import ActorCritic
# from buffer import RolloutBuffer
# from ppo import ppo_update
# from env import make_env


# def main():
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu" # For testing without GPU
#     num_envs = 128
#     num_steps = 64
#     total_iterations = 50

#     env, obs_dim, act_dim = make_env(num_envs, device)

#     policy = ActorCritic(obs_dim, act_dim).to(device)
#     optimizer = optim.Adam(policy.parameters(), lr=3e-4)

#     buffer = RolloutBuffer(num_steps, num_envs, obs_dim, act_dim, device)

#     # obs = env.reset() # For the dummy env
#     next_obs, reward, done = env.step(action)

#     # reset finished envs
#     if done.any():
#         reset_obs = env.reset()
#         next_obs = torch.where(done.unsqueeze(-1) > 0, reset_obs, next_obs)

#     buffer.add(obs, action, logprob, reward, done, value)
#     obs = next_obs

#     for it in range(total_iterations):
#         buffer.reset()

#         for step in range(num_steps):
#             with torch.no_grad():
#                 action, logprob, value = policy.act(obs)

#             next_obs, reward, done = env.step(action)

#             buffer.add(obs, action, logprob, reward, done, value)
#             obs = next_obs

#         loss = ppo_update(policy, optimizer, buffer)
#         print(f"Iteration {it+1}, Loss: {loss:.4f}")

#     print("✅ PPO training loop finished successfully.")


# if __name__ == "__main__":
#     main()
from isaacgym import gymapi
import torch
import torch.optim as optim

from model import ActorCritic
from buffer import RolloutBuffer
from ppo import ppo_update

def main():
    device = "cpu"  # PPO on CPU (required for RTX 5090)

    num_envs = 128
    num_steps = 64
    total_iterations = 50

    env, obs_dim, act_dim = make_env(num_envs, device)

    policy = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, act_dim, device)

    # Initial reset
    obs = env.reset()
    obs = obs.to(device)

    for it in range(total_iterations):
        buffer.reset()

        for step in range(num_steps):
            # 1️⃣ Compute action FIRST
            with torch.no_grad():
                action, logprob, value = policy.act(obs)

            # 2️⃣ Step environment
            next_obs, reward, done = env.step(action)

            # 3️⃣ Ensure tensors are on CPU
            next_obs = next_obs.to(device)
            reward = reward.to(device)
            done = done.to(device)

            # 4️⃣ Store transition
            buffer.add(obs, action, logprob, reward, done, value)

            # 5️⃣ Reset finished envs
            if done.any():
                reset_obs = env.reset().to(device)
                next_obs = torch.where(
                    done.unsqueeze(-1) > 0, reset_obs, next_obs
                )

            obs = next_obs

        loss = ppo_update(policy, optimizer, buffer)
        print(f"Iteration {it + 1}, Loss: {loss:.4f}")

    print("✅ PPO training finished successfully.")

if __name__ == "__main__":
      main()