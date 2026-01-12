#!/usr/bin/env python3
"""Visualize V_h(s) ≈ min_a Q_h(s,a) on (x,z) slices for QuadrotorTracking2D-v0."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxrl5.envs.quadrotor_tracking_2d import make_quadrotor_tracking_2d_env
from jaxrl5.networks import ddpm_sampler
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.tools.load_sac_lag import load_sac_lag
from jaxrl5.tools.load_ssm import load_ssm
from jaxrl5.wrappers.action_rescale import SymmetricActionWrapper


def parse_float_list(text: str) -> List[float]:
    items = [t.strip() for t in text.split(",") if t.strip()]
    return [float(item) for item in items]


def create_env(env_name: str, seed: int) -> gym.Env:
    if env_name == "QuadrotorTracking2D-v0":
        env = make_quadrotor_tracking_2d_env()
    else:
        env = gym.make(env_name)

    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
        env.action_space.high, 1.0
    ):
        env = SymmetricActionWrapper(env)

    env.reset(seed=seed)
    return env


def get_ref_from_env(
    env: gym.Env, ref_mode: str, ref_waypoint_idx: int | None, seed: int
) -> np.ndarray:
    if ref_mode == "zero":
        return np.zeros(6, dtype=np.float32)

    if ref_mode == "waypoint":
        try:
            waypoints = env.unwrapped._waypoints  # noqa: SLF001
            if ref_waypoint_idx is None:
                ref_waypoint_idx = 0
            ref = np.asarray(waypoints[int(ref_waypoint_idx)], dtype=np.float32)
            return ref
        except Exception:
            ref_mode = "reset"

    obs, _ = env.reset(seed=seed)
    return np.asarray(obs[6:], dtype=np.float32)


def build_obs_batch(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    z_dot: float,
    ref: np.ndarray,
) -> np.ndarray:
    B = grid_x.size
    obs = np.zeros((B, 12), dtype=np.float32)
    obs[:, 0] = grid_x.reshape(-1)
    obs[:, 2] = grid_z.reshape(-1)
    obs[:, 3] = float(z_dot)
    obs[:, 6:] = ref[None, :]
    return obs


def sample_actions_gaussian(
    agent,
    obs_tiled: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    dist = agent.actor.apply_fn({"params": agent.actor.params}, obs_tiled)
    actions = dist.sample(seed=rng)
    return actions


def sample_actions_ddpm(
    agent,
    obs_tiled: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    actions, _ = ddpm_sampler(
        agent.score_model.apply_fn,
        agent.score_model.params,
        agent.T,
        rng,
        agent.act_dim,
        obs_tiled,
        agent.alphas,
        agent.alpha_hats,
        agent.betas,
        agent.ddpm_temperature,
        agent.clip_sampler,
    )
    return actions


def eval_qh_batch(
    algo: str,
    agent,
    obs_tiled: jnp.ndarray,
    actions_flat: jnp.ndarray,
    batch_chunk: int,
) -> jnp.ndarray:
    total = obs_tiled.shape[0]
    if batch_chunk <= 0 or batch_chunk >= total:
        return eval_qh_batch_chunk(algo, agent, obs_tiled, actions_flat)

    outputs = []
    for start in range(0, total, batch_chunk):
        end = min(start + batch_chunk, total)
        outputs.append(
            eval_qh_batch_chunk(algo, agent, obs_tiled[start:end], actions_flat[start:end])
        )
    return jnp.concatenate(outputs, axis=0)


def eval_qh_batch_chunk(
    algo: str,
    agent,
    obs_tiled: jnp.ndarray,
    actions_flat: jnp.ndarray,
) -> jnp.ndarray:
    if algo in {"rac", "ssm"}:
        qh = agent.safety_critic.apply_fn(
            {"params": agent.safety_critic.params},
            obs_tiled,
            actions_flat,
            training=False,
        )
        return qh

    qcs = agent.cost_critic.apply_fn(
        {"params": agent.cost_critic.params},
        obs_tiled,
        actions_flat,
        training=False,
    )
    return qcs.min(axis=0)


def compute_vh_for_slice(
    algo: str,
    agent,
    obs_batch: np.ndarray,
    K: int,
    rng: jax.random.PRNGKey,
    batch_chunk: int,
) -> np.ndarray:
    B = obs_batch.shape[0]
    obs_tiled = np.repeat(obs_batch, K, axis=0)
    obs_tiled = jnp.asarray(obs_tiled)

    if algo == "ssm":
        actions_flat = sample_actions_ddpm(agent, obs_tiled, rng)
    else:
        actions_flat = sample_actions_gaussian(agent, obs_tiled, rng)

    qh_flat = eval_qh_batch(algo, agent, obs_tiled, actions_flat, batch_chunk)
    qh_flat = qh_flat.reshape(B, K)
    vh_flat = qh_flat.min(axis=1)
    return np.asarray(vh_flat)


def plot_vh_grid(
    X: np.ndarray,
    Z: np.ndarray,
    vh_list: List[np.ndarray],
    z_dot_list: List[float],
    threshold: float,
    out_path: str,
) -> None:
    cols = len(z_dot_list)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4), squeeze=False)

    all_vh = np.concatenate([v.reshape(-1) for v in vh_list])
    vmin = float(np.nanmin(all_vh))
    vmax = float(np.nanmax(all_vh))

    last_im = None
    for idx, (vh, z_dot) in enumerate(zip(vh_list, z_dot_list)):
        ax = axes[0, idx]
        last_im = ax.contourf(X, Z, vh, levels=50, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.contour(X, Z, vh, levels=[threshold], colors="white", linewidths=1.5)
        ax.set_title(f"z_dot={z_dot}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V_h(s) region on (x,z) grid.")
    parser.add_argument("--algo", choices=["rac", "ssm", "sac_lag"], required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_n", type=int, default=101)
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--z_dot_list", type=str, default="-1,0,1")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--out", type=str, default="vh_region.png")
    parser.add_argument("--save_npz", type=str, default="")
    parser.add_argument("--ref_mode", choices=["reset", "waypoint", "zero"], default="reset")
    parser.add_argument("--ref_waypoint_idx", type=int, default=None)
    parser.add_argument("--batch_chunk", type=int, default=0)
    args = parser.parse_args()

    env = create_env(args.env_name, args.seed)
    obs_space = env.observation_space
    act_space = env.action_space

    if args.algo == "rac":
        agent, _, meta = load_rac(
            args.ckpt_path,
            step=args.step,
            observation_space=obs_space,
            action_space=act_space,
            seed=args.seed,
            deterministic=False,
        )
        threshold = 0.0
    elif args.algo == "ssm":
        agent, _, meta = load_ssm(
            args.ckpt_path,
            step=args.step,
            observation_space=obs_space,
            action_space=act_space,
            seed=args.seed,
            deterministic=True,
        )
        threshold = 0.0
    else:
        agent, _, meta = load_sac_lag(
            args.ckpt_path,
            step=args.step,
            observation_space=obs_space,
            action_space=act_space,
            seed=args.seed,
            deterministic=False,
        )
        threshold = float(getattr(agent, "cost_limit", 0.0))

    ref = get_ref_from_env(env, args.ref_mode, args.ref_waypoint_idx, args.seed)

    z_dot_list = parse_float_list(args.z_dot_list)
    x = np.linspace(args.x_min, args.x_max, args.grid_n, dtype=np.float32)
    z = np.linspace(args.z_min, args.z_max, args.grid_n, dtype=np.float32)
    X, Z = np.meshgrid(x, z, indexing="xy")

    B = args.grid_n * args.grid_n
    print(f"Env obs_space={obs_space}, act_space={act_space}")
    print(f"ref_mode={args.ref_mode}, ref={ref}")
    print(f"B={B}, K={args.K}, batch_chunk={args.batch_chunk}")
    print(f"Output: {args.out}")

    vh_list = []
    for z_dot in z_dot_list:
        obs_batch = build_obs_batch(X, Z, z_dot, ref)
        rng = jax.random.PRNGKey(args.seed)
        vh_flat = compute_vh_for_slice(
            args.algo,
            agent,
            obs_batch,
            args.K,
            rng,
            args.batch_chunk,
        )
        vh = vh_flat.reshape(args.grid_n, args.grid_n)
        vh_list.append(vh)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_vh_grid(X, Z, vh_list, z_dot_list, threshold, args.out)

    if args.save_npz:
        np.savez(
            args.save_npz,
            X=X,
            Z=Z,
            z_dot_list=np.array(z_dot_list, dtype=np.float32),
            Vh_list=np.stack(vh_list, axis=0),
            threshold=float(threshold),
            algo=args.algo,
            ckpt_path=args.ckpt_path,
            step=args.step if args.step is not None else -1,
            K=args.K,
            ref_mode=args.ref_mode,
            ref=ref,
        )
        print(f"Saved npz to {args.save_npz}")


if __name__ == "__main__":
    main()
