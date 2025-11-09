from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def _sample(env, action, episode_length):

    next_obs, reward, cost, terminated, truncated, info = env.step(action)

    if episode_length + 1 >= 400:
        truncated = True

    return next_obs, reward, cost, terminated, truncated, info 

def evaluate(
    env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    episodes: int = 10,
) -> Dict[str, Any]:
    """Evaluate a policy on a Gymnasium-compatible safety environment."""

    returns, costs, lengths = [], [], []
    for _ in range(episodes):
        obs, info = env.reset()
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            # obs, reward, cost, terminated, truncated, info = _sample(env, action, ep_len)
            ep_ret += float(reward)
            ep_cost += float(cost)
            ep_len += 1

        returns.append(ep_ret)
        costs.append(ep_cost)
        lengths.append(ep_len)

    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/cost_mean": float(np.mean(costs)),
        "eval/ep_len_mean": float(np.mean(lengths)),
    }
