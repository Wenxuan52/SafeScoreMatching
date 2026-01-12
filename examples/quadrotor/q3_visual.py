"""Q3 Figure: Multimodality of action distribution (baseline vs SSM).

Example:
python examples/quadrotor/q3_visual.py \
  --ckpt_ssm /ABS/PATH/TO/ssm_run_dir --step_ssm 1500000 \
  --ckpt_baseline /ABS/PATH/TO/sac_lag_run_dir --step_baseline 1500000 \
  --algo_baseline sac_lag \
  --num_samples 2000 --seed 0 \
  --z_star 0.52 --x_star 0.0 --vz_star 0.0 --waypoint_idx 0 \
  --cache_path examples/quadrotor/figures/q3_cache.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from examples.quadrotor.rollout_utils import make_quad2d_env, load_policy
except Exception:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from examples.quadrotor.rollout_utils import make_quad2d_env, load_policy


def _init_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "init_x": float(args.x_star),
        "init_vx": float(args.vx_star),
        "init_z": float(args.z_star),
        "init_vz": float(args.vz_star),
        "init_theta": float(args.theta_star),
        "init_omega": float(args.omega_star),
        "init_waypoint_idx": int(args.waypoint_idx),
    }


def _validate_action(action: np.ndarray, act_dim: int, *, context: str) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.ndim == 2 and action_np.shape[0] == 1:
        action_np = action_np[0]
    if action_np.shape != (act_dim,):
        raise ValueError(f"{context}: action shape {action_np.shape} != ({act_dim},)")
    if not np.all(np.isfinite(action_np)):
        preview = action_np[: min(5, action_np.size)]
        raise ValueError(f"{context}: non-finite action {preview}")
    return action_np


def _compute_h_from_obs(obs: np.ndarray) -> float:
    x = float(obs[0])
    z = float(obs[2])
    h_corridor = max(0.5 - z, z - 1.5, 0.0)
    h_oob_x = max(abs(x) - 2.0, 0.0)
    h_oob_z = max(abs(z) - 3.0, 0.0)
    return float(max(h_corridor, h_oob_x, h_oob_z))


def _sample_actions(
    policy_fn,
    obs_star: np.ndarray,
    *,
    num_samples: int,
    act_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    context_prefix: str,
) -> np.ndarray:
    actions = np.zeros((num_samples, act_dim), dtype=np.float32)
    for i in range(num_samples):
        action = policy_fn(obs_star)
        action = _validate_action(
            action,
            act_dim,
            context=f"{context_prefix} sample={i} obs={obs_star[:6]}",
        )
        action = np.clip(action, action_low, action_high)
        actions[i] = action
    return actions


def _one_step_h(
    env,
    *,
    init_opts: Dict[str, Any],
    action: np.ndarray,
    seed: int,
) -> float:
    obs, _ = env.reset(seed=seed, options=init_opts)
    step_out = env.step(action)
    if len(step_out) != 6:
        raise RuntimeError("cost wrapper missing: env.step must return 6 values")
    next_obs, _, _, _, _, _ = step_out
    return _compute_h_from_obs(next_obs)


def _compute_one_step_h_for_actions(
    env,
    actions: np.ndarray,
    *,
    init_opts: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    h_vals = np.zeros((actions.shape[0],), dtype=np.float32)
    for i, action in enumerate(actions):
        h_vals[i] = _one_step_h(env, init_opts=init_opts, action=action, seed=seed + i)
    return h_vals


def _resolve_save_paths(
    save_dir: str,
    *,
    num_samples: int,
    z_star: float,
    step_baseline: Optional[int],
    step_ssm: Optional[int],
) -> Tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    z_str = f"{z_star:.2f}".replace(".", "p")
    step_a = step_baseline if step_baseline is not None else "latest"
    step_b = step_ssm if step_ssm is not None else "latest"
    filename = f"q3_{timestamp}_N{num_samples}_z{z_str}_steps_{step_a}_{step_b}"
    return (
        os.path.join(save_dir, f"{filename}.png"),
        os.path.join(save_dir, f"{filename}.npz"),
    )


def _plot(
    *,
    actions_baseline: np.ndarray,
    actions_ssm: np.ndarray,
    h_baseline: np.ndarray,
    h_ssm: np.ndarray,
    title_baseline: str,
    title_ssm: str,
    num_samples: int,
    z_star: float,
    save_path: str,
    dpi: int,
    save_pdf: bool,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    all_h = np.concatenate([h_baseline, h_ssm])
    if all_h.size:
        vmin = float(np.nanpercentile(all_h, 5))
        vmax = float(np.nanpercentile(all_h, 95))
    else:
        vmin, vmax = 0.0, 1.0
    vmin = min(vmin, 0.0)
    vmax = max(vmax, 1e-6)

    hb1 = axes[0].hexbin(
        actions_baseline[:, 0],
        actions_baseline[:, 1],
        C=h_baseline,
        reduce_C_function=np.mean,
        gridsize=35,
        extent=[-1, 1, -1, 1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].scatter(
        actions_baseline[:, 0],
        actions_baseline[:, 1],
        s=4,
        alpha=0.05,
        color="black",
    )
    axes[0].set_title(title_baseline)
    axes[0].set_xlabel("a1")
    axes[0].set_ylabel("a2")
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].text(
        0.02,
        0.98,
        f"N={num_samples}, z*={z_star:.2f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    hb2 = axes[1].hexbin(
        actions_ssm[:, 0],
        actions_ssm[:, 1],
        C=h_ssm,
        reduce_C_function=np.mean,
        gridsize=35,
        extent=[-1, 1, -1, 1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].scatter(
        actions_ssm[:, 0],
        actions_ssm[:, 1],
        s=4,
        alpha=0.05,
        color="black",
    )
    axes[1].set_title(title_ssm)
    axes[1].set_xlabel("a1")
    axes[1].set_ylabel("a2")
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].text(
        0.02,
        0.98,
        f"N={num_samples}, z*={z_star:.2f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    cbar = fig.colorbar(hb2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("one-step h(s') (lower is safer; <=0 safe)")

    fig.savefig(save_path, dpi=dpi)
    if save_pdf:
        base, _ = os.path.splitext(save_path)
        fig.savefig(f"{base}.pdf", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _save_cache(
    cache_path: str,
    *,
    obs_star: np.ndarray,
    init_options: Dict[str, Any],
    actions_base: np.ndarray,
    h_next_base: np.ndarray,
    actions_ssm: np.ndarray,
    h_next_ssm: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    np.savez(
        cache_path,
        obs_star=obs_star,
        init_options=json.dumps(init_options),
        actions_base=actions_base,
        h_next_base=h_next_base,
        actions_ssm=actions_ssm,
        h_next_ssm=h_next_ssm,
        meta=json.dumps(meta),
    )


def _load_cache(cache_path: str):
    data = np.load(cache_path, allow_pickle=True)
    obs_star = data["obs_star"]
    init_options = json.loads(str(data["init_options"]))
    actions_base = data["actions_base"]
    h_next_base = data["h_next_base"]
    actions_ssm = data["actions_ssm"]
    h_next_ssm = data["h_next_ssm"]
    meta = json.loads(str(data["meta"]))
    return obs_star, init_options, actions_base, h_next_base, actions_ssm, h_next_ssm, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3 visualization for action multimodality")
    parser.add_argument("--ckpt_ssm", required=True)
    parser.add_argument("--step_ssm", type=int, default=None)
    parser.add_argument("--ckpt_baseline", required=True)
    parser.add_argument("--step_baseline", type=int, default=None)
    parser.add_argument("--algo_baseline", default="sac_lag")
    parser.add_argument("--algo_ssm", default="ssm")

    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic_ssm", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--deterministic_baseline", type=lambda x: x.lower() == "true", default=False)

    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")

    parser.add_argument("--x_star", type=float, default=0.0)
    parser.add_argument("--vx_star", type=float, default=0.0)
    parser.add_argument("--z_star", type=float, default=0.52)
    parser.add_argument("--vz_star", type=float, default=0.0)
    parser.add_argument("--theta_star", type=float, default=0.0)
    parser.add_argument("--omega_star", type=float, default=0.0)
    parser.add_argument("--waypoint_idx", type=int, default=0)

    parser.add_argument("--save_dir", default="examples/quadrotor/figures")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--save_pdf", action="store_true", default=False)
    parser.add_argument("--show", action="store_true", default=False)

    parser.add_argument("--cache_path", default="")
    parser.add_argument("--no_cache", action="store_true", default=False)

    args = parser.parse_args()

    cache_path = args.cache_path.strip()
    if cache_path and os.path.exists(cache_path) and not args.no_cache:
        (
            obs_star,
            init_options,
            actions_base,
            h_next_base,
            actions_ssm,
            h_next_ssm,
            meta,
        ) = _load_cache(cache_path)
    else:
        env = make_quad2d_env(env_name=args.env_name, seed=args.seed)
        init_options = _init_options(args)
        obs_star, _ = env.reset(seed=args.seed, options=init_options)
        obs_star = np.asarray(obs_star, dtype=np.float32)
        act_dim = env.action_space.shape[-1]

        _, policy_baseline, meta_base = load_policy(
            algo=args.algo_baseline,
            ckpt_path=args.ckpt_baseline,
            step=args.step_baseline,
            env=env,
            seed=args.seed,
            deterministic=args.deterministic_baseline,
        )
        _, policy_ssm, meta_ssm = load_policy(
            algo=args.algo_ssm,
            ckpt_path=args.ckpt_ssm,
            step=args.step_ssm,
            env=env,
            seed=args.seed,
            deterministic=args.deterministic_ssm,
        )

        actions_base = _sample_actions(
            policy_baseline,
            obs_star,
            num_samples=args.num_samples,
            act_dim=act_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            context_prefix=f"baseline={args.algo_baseline} step={args.step_baseline}",
        )
        actions_ssm = _sample_actions(
            policy_ssm,
            obs_star,
            num_samples=args.num_samples,
            act_dim=act_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            context_prefix=f"ssm={args.algo_ssm} step={args.step_ssm}",
        )

        h_next_base = _compute_one_step_h_for_actions(
            env,
            actions_base,
            init_opts=init_options,
            seed=args.seed + 1000,
        )
        h_next_ssm = _compute_one_step_h_for_actions(
            env,
            actions_ssm,
            init_opts=init_options,
            seed=args.seed + 2000,
        )
        env.close()

        meta = {
            "num_samples": args.num_samples,
            "seed": args.seed,
            "algo_baseline": args.algo_baseline,
            "algo_ssm": args.algo_ssm,
            "step_baseline": meta_base.get("step", args.step_baseline),
            "step_ssm": meta_ssm.get("step", args.step_ssm),
            "deterministic_baseline": args.deterministic_baseline,
            "deterministic_ssm": args.deterministic_ssm,
            "init_options": init_options,
        }

        if cache_path and not args.no_cache:
            _save_cache(
                cache_path,
                obs_star=obs_star,
                init_options=init_options,
                actions_base=actions_base,
                h_next_base=h_next_base,
                actions_ssm=actions_ssm,
                h_next_ssm=h_next_ssm,
                meta=meta,
            )

    save_path, auto_cache_path = _resolve_save_paths(
        args.save_dir,
        num_samples=args.num_samples,
        z_star=args.z_star,
        step_baseline=args.step_baseline,
        step_ssm=args.step_ssm,
    )
    if cache_path and os.path.exists(cache_path):
        cache_used = cache_path
    else:
        cache_used = auto_cache_path
        if not args.no_cache and not os.path.exists(cache_used):
            _save_cache(
                cache_used,
                obs_star=obs_star,
                init_options=init_options,
                actions_base=actions_base,
                h_next_base=h_next_base,
                actions_ssm=actions_ssm,
                h_next_ssm=h_next_ssm,
                meta=meta,
            )

    title_baseline = f"Gaussian baseline ({args.algo_baseline})"
    if args.step_baseline is not None:
        title_baseline += f" @ {args.step_baseline}"
    title_ssm = "SSM (Diffusion policy)"
    if args.step_ssm is not None:
        title_ssm += f" @ {args.step_ssm}"

    _plot(
        actions_baseline=actions_base,
        actions_ssm=actions_ssm,
        h_baseline=h_next_base,
        h_ssm=h_next_ssm,
        title_baseline=title_baseline,
        title_ssm=title_ssm,
        num_samples=args.num_samples,
        z_star=args.z_star,
        save_path=save_path,
        dpi=args.dpi,
        save_pdf=args.save_pdf,
        show=args.show,
    )

    print(f"[OK] saved: {save_path}")
    print(f"[OK] cache: {cache_used}")


if __name__ == "__main__":
    main()
