# import torch


# class DummyEnv:
#     def __init__(self, num_envs, obs_dim, act_dim, device):
#         self.num_envs = num_envs
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.device = device

#     def reset(self):
#         return torch.randn(self.num_envs, self.obs_dim, device=self.device)

#     def step(self, actions):
#         obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
#         reward = torch.rand(self.num_envs, device=self.device)
#         done = torch.zeros(self.num_envs, device=self.device)
#         return obs, reward, done


# def make_env(num_envs, device):
#     obs_dim = 4
#     act_dim = 1
#     env = DummyEnv(num_envs, obs_dim, act_dim, device)
#     return env, obs_dim, act_dim


import torch
from isaacgym import gymapi, gymtorch
import os


class IsaacCartpoleEnv:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs

        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        sim_params.physx.use_gpu = True
        sim_params.physx.num_solver_iterations = 4
        sim_params.physx.num_position_iterations = 8

        compute_id = 0
        graphics_id = -1  # ❗ no rendering → shared-GPU safe

        self.sim = self.gym.create_sim(compute_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
        assert self.sim is not None, "Failed to create Isaac Gym sim"

        asset_root = os.path.join(os.environ["ISAAC_GYM_PATH"], "assets")
        asset_file = "urdf/cartpole.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        spacing = 2.0
        envs_per_row = int(num_envs ** 0.5)

        self.envs = []
        self.actors = []

        for i in range(num_envs):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-spacing, -spacing, 0.0),
                gymapi.Vec3(spacing, spacing, spacing),
                envs_per_row,
            )
            actor = self.gym.create_actor(env, cartpole_asset, gymapi.Transform(), "cartpole", i, 1)
            self.envs.append(env)
            self.actors.append(actor)

        self.gym.prepare_sim(self.sim)

        # Tensor access
        self.dof_state_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )

        self.obs_dim = 4
        self.act_dim = 1

    def reset(self):
        self.dof_state_tensor.zero_()
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        return self._get_obs()

    def step(self, actions):
        actions = torch.clamp(actions, -1.0, 1.0)

        forces = torch.zeros((self.num_envs, 2), device=self.device)
        forces[:, 1] = actions.squeeze(-1) * 100.0

        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(forces)
        )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._compute_done(obs)

        return obs, reward, done

    def _get_obs(self):
        # [cart_pos, cart_vel, pole_angle, pole_vel]
        return self.dof_state_tensor.view(self.num_envs, -1)

    def _compute_reward(self, obs):
        pole_angle = obs[:, 2]
        return 1.0 - pole_angle.abs()

    def _compute_done(self, obs):
        pole_angle = obs[:, 2]
        return (pole_angle.abs() > 0.4).float()


def make_env(num_envs, device):
    env = IsaacCartpoleEnv(num_envs, device)
    return env, env.obs_dim, env.act_dim