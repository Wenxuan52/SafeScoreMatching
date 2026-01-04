import argparse

import numpy as np

from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="ReplayBuffer cost field sanity check")
    parser.add_argument("--env_name", type=str, default="SafetyCarButton1-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    env = make_env(args.env_name, seed=args.seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    replay_buffer = ReplayBuffer(obs_shape, act_shape, capacity=args.steps)
    replay_buffer.seed(args.seed)

    obs, info = env.reset(seed=args.seed)

    for step in range(args.steps):
        action = env.action_space.sample()
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        replay_buffer.insert(obs, action, reward, cost, next_obs, terminated, truncated)
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()

    assert len(replay_buffer) == args.steps, f"replay buffer size mismatch: {len(replay_buffer)}"

    batch = replay_buffer.sample(args.batch_size)
    assert "costs" in batch, "Replay buffer batch missing 'costs' key"
    assert batch["costs"].dtype == np.float32, batch["costs"].dtype
    assert batch["costs"].shape == (args.batch_size,), batch["costs"].shape
    assert np.isfinite(batch["costs"]).all(), "Non-finite costs in sampled batch"

    print(
        "cost stats:",
        float(batch["costs"].mean()),
        float(batch["costs"].min()),
        float(batch["costs"].max()),
    )
    print(
        "reward stats:",
        float(batch["rewards"].mean()),
        float(batch["rewards"].min()),
        float(batch["rewards"].max()),
    )


if __name__ == "__main__":
    main()
