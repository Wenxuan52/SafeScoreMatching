"""Visualize policy rollouts on QuadrotorTracking2D-v0.

This script loads a trained checkpoint (TD3 or SafeScoreMatchingLearner),
rolls out one or more episodes, and plots x-z trajectories alongside the
reference circle and altitude constraints. It optionally saves a CSV with
per-step data for further analysis.
"""
import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from jaxrl5.agents import SafeScoreMatchingLearner, TD3Learner
from jaxrl5.envs import make_env


def _resolve_checkpoint_path(base_dir: str, step: Optional[int]) -> str:
    """Locate a checkpoint file under base_dir or its checkpoints/ subdir."""
    if os.path.isfile(base_dir):
        return base_dir

    ckpt_dir = os.path.join(base_dir, "checkpoints") if os.path.isdir(base_dir) else base_dir
    if step is not None:
        candidate = os.path.join(ckpt_dir, f"ckpt_{step}.msgpack")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    # Find latest checkpoint by filename ordering.
    candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_*.msgpack"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return sorted(candidates)[-1]


def _load_agent(agent_name: str, checkpoint_path: str, env) -> object:
    """Load an agent from a checkpoint path based on its type."""
    if agent_name == "ssm":
        return SafeScoreMatchingLearner.load(checkpoint_path)
    if agent_name == "td3":
        # TD3 checkpoints are expected to be serialized via flax.serialization.to_bytes(agent).
        with open(checkpoint_path, "rb") as f:
            data = f.read()
        if hasattr(serialization, "msgpack_restore"):
            return serialization.msgpack_restore(data)
        return serialization.from_bytes(TD3Learner, data)
    raise ValueError(f"Unsupported agent type: {agent_name}")


def _select_action(agent, obs: np.ndarray, deterministic: bool, agent_name: str, action_space) -> Tuple[np.ndarray, object]:
    obs_np = np.asarray(obs, dtype=np.float32)
    if agent_name == "ssm":
        if deterministic:
            action, agent = agent.eval_actions(obs_np)
        else:
            action, agent = agent.sample_actions(obs_np)
        action = np.asarray(action, dtype=np.float32)
    else:  # TD3 expects batch dimension
        obs_b = obs_np[None]
        if deterministic:
            action_b, agent = agent.eval_actions(obs_b)
        else:
            action_b, agent = agent.sample_actions(obs_b)
        action = np.asarray(action_b[0], dtype=np.float32)
    action = np.clip(action, action_space.low, action_space.high)
    return action, agent


def _reference_circle(num_points: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(angles)
    z = 1.0 + np.sin(angles)
    return x, z


def rollout_episodes(env, agent, episodes: int, deterministic: bool, agent_name: str, seed: int) -> List[List[Dict[str, float]]]:
    trajectories: List[List[Dict[str, float]]] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_data: List[Dict[str, float]] = []
        t = 0
        while True:
            action, agent = _select_action(agent, obs, deterministic, agent_name, env.action_space)
            step_out = env.step(action)
            if len(step_out) == 6:
                next_obs, reward, cost, terminated, truncated, info = step_out
            else:
                next_obs, reward, terminated, truncated, info = step_out
                cost = float(info.get("cost", 0.0))
            x, z = float(obs[0]), float(obs[2])
            x_ref, z_ref = float(obs[6]), float(obs[8])
            idx = int(info.get("idx", t % 360))
            ep_data.append(
                {
                    "t": t,
                    "x": x,
                    "z": z,
                    "x_ref": x_ref,
                    "z_ref": z_ref,
                    "reward": float(reward),
                    "cost": float(cost),
                    "h": float(info.get("h", 0.0)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "idx": idx,
                }
            )
            obs = next_obs
            t += 1
            if terminated or truncated:
                break
        trajectories.append(ep_data)
    return trajectories


def plot_trajectories(trajectories: List[List[Dict[str, float]]], out_path: str, agent_name: str, checkpoint_label: str) -> None:
    plt.figure(figsize=(8, 8))
    # Altitude constraints.
    plt.axhline(0.5, color="black", linestyle="-")
    plt.axhline(1.5, color="black", linestyle="-")
    # Reference circle.
    circle_x, circle_z = _reference_circle()
    plt.plot(circle_x, circle_z, "k--", label="reference circle")

    colors = plt.cm.tab10.colors
    for i, ep_data in enumerate(trajectories):
        xs = [d["x"] for d in ep_data]
        zs = [d["z"] for d in ep_data]
        status = "term" if ep_data[-1]["terminated"] else ("trunc" if ep_data[-1]["truncated"] else "done")
        plt.plot(xs, zs, color=colors[i % len(colors)], label=f"episode {i} ({status})")

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"{agent_name.upper()} trajectory @ {checkpoint_label}")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_csv(trajectories: List[List[Dict[str, float]]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "episode",
        "t",
        "x",
        "z",
        "x_ref",
        "z_ref",
        "reward",
        "cost",
        "h",
        "terminated",
        "truncated",
        "idx",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep_idx, ep_data in enumerate(trajectories):
            for row in ep_data:
                writer.writerow({"episode": ep_idx, **row})


def main():
    parser = argparse.ArgumentParser(description="Visualize Quadrotor policy trajectories")
    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (eval)")
    parser.add_argument("--agent", choices=["td3", "ssm"], required=True)
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory or file")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Specific checkpoint step")
    parser.add_argument("--out_dir", default="results/visualizations")
    parser.add_argument("--save_csv", action="store_true")
    args = parser.parse_args()

    env = make_env(args.env_name, seed=args.seed)
    ckpt_path = _resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_step)
    agent = _load_agent(args.agent, ckpt_path, env)

    trajectories = rollout_episodes(
        env,
        agent,
        episodes=args.episodes,
        deterministic=args.deterministic,
        agent_name=args.agent,
        seed=args.seed,
    )

    ckpt_label = os.path.basename(ckpt_path)
    out_png = os.path.join(args.out_dir, f"traj_{args.agent}_{ckpt_label}.png")
    plot_trajectories(trajectories, out_png, args.agent, ckpt_label)
    print(f"Saved trajectory plot to {out_png}")

    if args.save_csv:
        out_csv = os.path.join(args.out_dir, f"traj_{args.agent}_{ckpt_label}.csv")
        save_csv(trajectories, out_csv)
        print(f"Saved trajectory CSV to {out_csv}")

    env.close()


if __name__ == "__main__":
    main()

# Example usage (TD3):
# python examples/quadrotor/visualize_policy_trajectory.py \
#   --agent td3 \
#   --checkpoint_dir results/QuadrotorTracking2D-v0/jaxrl5_quad2d_td3_baseline/YYYY-MM-DD_seed0000 \
#   --checkpoint_step 50000 \
#   --episodes 1 \
#   --deterministic \
#   --out_dir results/visualizations/td3_step50000 \
#   --save_csv
# SSM variant is analogous with --agent ssm.
