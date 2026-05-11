import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def act(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return log_probs, entropy, value