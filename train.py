import torch
import torch.optim as optim

from model import ActorCritic
from buffer import RolloutBuffer
from ppo import ppo_update
from env import make_env


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_envs = 128
    num_steps = 64
    total_iterations = 50

    env, obs_dim, act_dim = make_env(num_envs, device)

    policy = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, act_dim, device)

    obs = env.reset()

    for it in range(total_iterations):
        buffer.reset()

        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, value = policy.act(obs)

            next_obs, reward, done = env.step(action)

            buffer.add(obs, action, logprob, reward, done, value)
            obs = next_obs

        loss = ppo_update(policy, optimizer, buffer)
        print(f"Iteration {it+1}, Loss: {loss:.4f}")

    print("✅ PPO training loop finished successfully.")


if __name__ == "__main__":
    main()