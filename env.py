import torch


class DummyEnv:
    def __init__(self, num_envs, obs_dim, act_dim, device):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

    def reset(self):
        return torch.randn(self.num_envs, self.obs_dim, device=self.device)

    def step(self, actions):
        obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        reward = torch.rand(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, device=self.device)
        return obs, reward, done


def make_env(num_envs, device):
    obs_dim = 4
    act_dim = 1
    env = DummyEnv(num_envs, obs_dim, act_dim, device)
    return env, obs_dim, act_dim
``