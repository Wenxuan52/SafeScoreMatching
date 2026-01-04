#!/usr/bin/env python
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import TD3Learner
from jaxrl5.data import ReplayBuffer
from jaxrl5.envs import make_env
from jaxrl5.evaluation import evaluate
from jaxrl5.utils import append_history
from jaxrl5.wrappers import WANDBVideo

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_quad2d_td3_baseline", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string("env_name", "QuadrotorTracking2D-v0", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Evaluation episodes.")
flags.DEFINE_integer("log_interval", 400, "Logging interval (steps).")
flags.DEFINE_integer("eval_interval", 10000, "Evaluation interval (steps).")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 200_000, "Number of training steps.")
flags.DEFINE_integer("start_training", 10_000, "Number of steps before learning starts.")
flags.DEFINE_boolean("wandb", False, "Enable wandb logging.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("epoch_length", 400, "Number of environment steps per epoch statistic.")
flags.DEFINE_integer("eval_seed_offset", 12345, "Offset for evaluation environment seed.")
flags.DEFINE_boolean("save_video", False, "Upload videos during evaluation (wandb only).")
flags.DEFINE_integer("save_interval", 50_000, "Checkpoint save interval (steps).")
flags.DEFINE_string(
    "out_dir",
    "results/QuadrotorTracking2D-v0/jaxrl5_quad2d_td3_baseline",
    "Root directory for experiment outputs.",
)
config_flags.DEFINE_config_file(
    "config",
    "examples/quadrotor/configs/td3_quad2d_config.py",
    "Path to TD3 hyperparameter configuration.",
    lock_config=False,
)


def _maybe_init_wandb():
    if not FLAGS.wandb:
        return

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; disabling wandb logging.")
        FLAGS.wandb = False
        return

    run_name = FLAGS.run_name or None
    try:
        wandb.init(
            project=FLAGS.project_name,
            name=run_name,
            tags=[FLAGS.run_name] if FLAGS.run_name else None,
        )
    except Exception as exc:  # wandb can raise UsageError if not logged in
        print(f"wandb init failed ({exc}); disabling wandb logging.")
        FLAGS.wandb = False
        return

    wandb.config.update(FLAGS)


def _make_env(env_name: str, seed: int, allow_video: bool = True):
    env = make_env(env_name, seed=seed)
    if allow_video and FLAGS.wandb and FLAGS.save_video:
        env = WANDBVideo(env)
    return env


def _make_eval_policy(agent: TD3Learner) -> Callable[[np.ndarray], np.ndarray]:
    eval_agent = agent

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal eval_agent
        obs_batched = np.asarray(obs[None, :], dtype=np.float32)
        action_b, eval_agent = eval_agent.eval_actions(obs_batched)
        return np.asarray(action_b[0])

    return policy


def _evaluate_violation_rate(env, policy_fn: Callable[[np.ndarray], np.ndarray], episodes: int) -> Dict[str, float]:
    violation_rates = []
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        violations = 0
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            violations += int(cost > 0)
            steps += 1
        if steps > 0:
            violation_rates.append(violations / steps)
    if not violation_rates:
        return {"eval/violation_rate_mean": 0.0, "eval/violation_rate_std": 0.0}
    violation_array = np.asarray(violation_rates, dtype=np.float32)
    return {
        "eval/violation_rate_mean": float(np.mean(violation_array)),
        "eval/violation_rate_std": float(np.std(violation_array)),
    }


def main(_):
    _maybe_init_wandb()

    run_id = FLAGS.run_name or f"{datetime.date.today().isoformat()}_seed{FLAGS.seed:04d}"
    run_dir = Path(FLAGS.out_dir) / run_id
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Persist configuration for reproducibility.
    config_path = run_dir / "config.json"
    if not config_path.exists():
        def _serialize(val):
            if hasattr(val, "to_dict"):
                try:
                    return val.to_dict()
                except Exception:
                    pass
            if isinstance(val, Path):
                return str(val)
            return val

        config_snapshot = {
            "flags": {k: _serialize(FLAGS[k].value) for k in FLAGS},
            "td3_config": getattr(FLAGS.config, "to_dict", dict)(FLAGS.config),
        }
        config_path.write_text(json.dumps(config_snapshot, indent=2))

    train_env = _make_env(FLAGS.env_name, seed=FLAGS.seed, allow_video=True)
    eval_env = _make_env(FLAGS.env_name, seed=FLAGS.seed + FLAGS.eval_seed_offset, allow_video=True)

    obs_shape = train_env.observation_space.shape
    act_shape = train_env.action_space.shape

    replay_buffer = ReplayBuffer(obs_shape, act_shape, capacity=FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent: TD3Learner = globals()[model_cls].create(
        FLAGS.seed,
        train_env.observation_space,
        train_env.action_space,
        **kwargs,
    )

    observation, _ = train_env.reset(seed=FLAGS.seed)
    episode_return, episode_cost, episode_length = 0.0, 0.0, 0
    epoch_reward, epoch_cost = 0.0, 0.0
    experiment_name = FLAGS.run_name or FLAGS.project_name
    latest_cost_mean = None

    def _save_checkpoint(agent_to_save: TD3Learner, step: int, obs_sample: np.ndarray):
        step_dir = ckpt_root / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        agent_to_save.save(str(step_dir), step)

        # Lightweight load check for shape correctness.
        loaded = TD3Learner.load(str(step_dir), step)
        test_action_b, _ = loaded.eval_actions(np.asarray(obs_sample[None, :], dtype=np.float32))
        test_action = np.asarray(test_action_b[0])
        assert test_action.shape == train_env.action_space.shape, "Loaded action has wrong shape"
        assert np.all(np.isfinite(test_action)), "Loaded action contains non-finite values"
        assert np.all(test_action >= -1.1) and np.all(test_action <= 1.1), "Loaded action out of expected range"

    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        if step < FLAGS.start_training:
            action = np.asarray(train_env.action_space.sample(), dtype=np.float32)
        else:
            obs_batched = np.asarray(observation[None, :], dtype=np.float32)
            action_b, agent = agent.sample_actions(obs_batched)
            action = np.asarray(action_b[0], dtype=np.float32)
            action = np.clip(action, train_env.action_space.low, train_env.action_space.high)

        next_obs, reward, cost, terminated, truncated, info = train_env.step(action)
        done = bool(terminated or truncated)

        replay_buffer.insert(
            observation,
            action,
            float(reward),
            float(cost),
            next_obs,
            terminated,
            truncated,
        )

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1
        epoch_reward += float(reward)
        epoch_cost += float(cost)
        observation = next_obs

        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, utd_ratio=FLAGS.utd_ratio)

            if FLAGS.wandb and step % FLAGS.log_interval == 0:
                import wandb

                wandb.log({f"training/{k}": float(v) for k, v in update_info.items()}, step=step)

        if done:
            if FLAGS.wandb:
                import wandb

                wandb.log(
                    {
                        "training/return": episode_return,
                        "training/cost": episode_cost,
                        "training/length": episode_length,
                    },
                    step=step,
                )

            observation, _ = train_env.reset()
            episode_return, episode_cost, episode_length = 0.0, 0.0, 0

        if step % FLAGS.epoch_length == 0:
            epoch_index = step // FLAGS.epoch_length
            latest_cost_mean = epoch_cost / FLAGS.epoch_length
            if FLAGS.wandb:
                import wandb

                wandb.log(
                    {
                        "epoch/index": epoch_index,
                        "epoch/reward_sum": epoch_reward,
                        "epoch/reward_mean": epoch_reward / FLAGS.epoch_length,
                        "epoch/cost_sum": epoch_cost,
                        "epoch/cost_mean": latest_cost_mean,
                        "training/cost_mean": latest_cost_mean,
                    },
                    step=step,
                )
            epoch_reward, epoch_cost = 0.0, 0.0

        if step % FLAGS.eval_interval == 0:
            eval_agent = agent
            eval_policy = _make_eval_policy(eval_agent)
            eval_metrics = evaluate(eval_env, eval_policy, episodes=FLAGS.eval_episodes)
            violation_metrics = _evaluate_violation_rate(
                eval_env, eval_policy, episodes=FLAGS.eval_episodes
            )
            metrics_all = {**eval_metrics, **violation_metrics}

            if FLAGS.wandb:
                import wandb

                wandb.log(metrics_all, step=step)
            else:
                print(
                    f"[step {step}] return={metrics_all['eval/return_mean']:.2f} "
                    f"cost={metrics_all['eval/cost_mean']:.2f} "
                    f"viol_rate={metrics_all['eval/violation_rate_mean']:.3f} "
                    f"len={metrics_all['eval/ep_len_mean']:.1f}"
                )

            append_history(
                step,
                FLAGS.env_name,
                experiment_name,
                FLAGS.seed,
                {
                    "eval/return_mean": metrics_all["eval/return_mean"],
                    "eval/return_std": metrics_all["eval/return_std"],
                    "eval/cost_mean": metrics_all["eval/cost_mean"],
                    "eval/cost_std": metrics_all.get("eval/cost_std", float("nan")),
                    "eval/violation_rate_mean": metrics_all["eval/violation_rate_mean"],
                    "eval/violation_rate_std": metrics_all.get("eval/violation_rate_std", float("nan")),
                    "eval/ep_len_mean": metrics_all["eval/ep_len_mean"],
                },
            )

        if step % FLAGS.save_interval == 0:
            _save_checkpoint(agent, step, observation)

    _save_checkpoint(agent, FLAGS.max_steps, observation)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    app.run(main)
