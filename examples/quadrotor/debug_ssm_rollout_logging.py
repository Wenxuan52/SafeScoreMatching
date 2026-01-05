"""Roll out a SafeScoreMatching policy on QuadrotorTracking2D with detailed logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from jaxrl5.agents import SafeScoreMatchingLearner
from jaxrl5.envs import make_env


def _rescale_action_from_unit(action: np.ndarray, action_space) -> np.ndarray:
    """Map actions in [-1, 1] to the env's Box bounds."""

    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if np.allclose(low, -1.0) and np.allclose(high, 1.0):
        return np.asarray(action, dtype=np.float32)

    scale = (high - low) / 2.0
    bias = (high + low) / 2.0
    return scale * np.asarray(action, dtype=np.float32) + bias


def _to_scalar(x: Any) -> float:
    if isinstance(x, (np.ndarray, list)):
        arr = np.asarray(x, dtype=np.float32)
        return float(arr.reshape(-1)[0])
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x))


def rollout(args: argparse.Namespace) -> Dict[str, Any]:
    env = make_env(args.env_name, seed=args.seed)
    agent: SafeScoreMatchingLearner = SafeScoreMatchingLearner.load(args.checkpoint)

    reset_options: Dict[str, Any] = {
        "init_x": args.init_x,
        "init_z": args.init_z,
        "init_vx": 0.0,
        "init_vz": 0.0,
        "init_theta": 0.0,
        "init_omega": 0.0,
    }

    obs, info = env.reset(seed=args.seed, options=reset_options)
    logs = []

    for t in range(args.steps):
        raw_action, agent = agent.eval_actions(
            np.asarray(obs, dtype=np.float32), guidance_mode=args.guidance_mode
        )
        rescaled_action = _rescale_action_from_unit(raw_action, env.action_space)
        env_action = np.clip(
            rescaled_action, env.action_space.low, env.action_space.high
        ).astype(np.float32)

        lambda_val = _to_scalar(agent._lambda_values(np.asarray(obs)[None])[0])
        qh = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            np.asarray(obs, dtype=np.float32)[None],
            env_action[None],
            training=False,
        )
        vh = np.maximum(0.0, np.asarray(qh, dtype=np.float32))[0]
        safe_flag = bool(vh <= agent.safety_threshold)

        next_obs, reward, cost, terminated, truncated, step_info = env.step(env_action)

        log_entry = {
            "t": t,
            "obs": np.asarray(obs, dtype=np.float32).tolist(),
            "x": float(obs[0]),
            "z": float(obs[2]),
            "raw_action": np.asarray(raw_action, dtype=np.float32).tolist(),
            "rescaled_action": np.asarray(rescaled_action, dtype=np.float32).tolist(),
            "env_action": env_action.tolist(),
            "reward": _to_scalar(reward),
            "cost": _to_scalar(cost),
            "safe_flag": safe_flag,
            "hjb_v": _to_scalar(vh),
            "lambda": lambda_val,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": {k: _to_scalar(v) for k, v in step_info.items()},
        }

        print(
            f"t={t:02d} x={log_entry['x']:.2f} z={log_entry['z']:.2f} "
            f"raw_a={log_entry['raw_action']} env_a={log_entry['env_action']} "
            f"r={log_entry['reward']:.3f} c={log_entry['cost']} "
            f"Vh={log_entry['hjb_v']:.3f} lam={log_entry['lambda']:.3f} "
            f"safe={log_entry['safe_flag']} term={terminated} trunc={truncated}"
        )

        logs.append(log_entry)
        obs = next_obs
        if terminated or truncated:
            break

    output_path = Path(args.output)
    if output_path == Path("debug_ssm_rollout.json"):
        run_name = args.run_name or Path(args.checkpoint).stem
        output_path = (
            Path("results")
            / "debug_rollouts"
            / f"{run_name}_{args.guidance_mode}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "guidance_mode": args.guidance_mode,
                "run_name": args.run_name or Path(args.checkpoint).stem,
                "steps": logs[:20],
                "reset_info": info,
            },
            f,
            indent=2,
        )

    return {"output_path": str(output_path), "steps": logs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug SSM quadrotor rollout")
    parser.add_argument("--checkpoint", required=True, help="Path to SSM checkpoint")
    parser.add_argument("--output", default="debug_ssm_rollout.json")
    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--init_x", type=float, default=1.0)
    parser.add_argument("--init_z", type=float, default=1.0)
    parser.add_argument(
        "--guidance_mode",
        choices=["none", "reward_only", "safety_only", "both"],
        default="both",
    )
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    rollout(parse_args())
