import torch


def ppo_update(
    policy,
    optimizer,
    buffer,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    rewards = buffer.rewards
    values = torch.cat([buffer.values, buffer.values[-1:].detach()])
    dones = buffer.dones

    # GAE
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + buffer.values

    # Flatten
    obs = buffer.obs.reshape(-1, buffer.obs.size(-1))
    actions = buffer.actions.reshape(-1, buffer.actions.size(-1))
    old_logprobs = buffer.logprobs.reshape(-1)
    returns = returns.reshape(-1)
    advantages = advantages.reshape(-1)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    logprobs, entropy, values = policy.evaluate(obs, actions)
    ratio = torch.exp(logprobs - old_logprobs)

    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = (returns - values).pow(2).mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()