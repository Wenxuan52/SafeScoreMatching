"""Q1 Figure: Execution violations + Recovery ability for QuadrotorTracking2D-v0.

Example:
python examples/quadrotor/q1_visual.py \
  --ckpt_ssm /ABS/PATH/TO/ssm_run_dir --step_ssm 1500000 \
  --ckpt_rac /ABS/PATH/TO/rac_run_dir --step_rac 1500000 \
  --ckpt_sac_lag /ABS/PATH/TO/sac_lag_run_dir --step_sac_lag 1500000 \
  --grid_n 41 --mc_short 10 --horizon_short 40 --mc_tts 1 --horizon_long 360 \
  --x_min -2 --x_max 2 --z_min 0 --z_max 2 \
  --cache_path examples/quadrotor/q1_cache.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from examples.quadrotor.rollout_utils import make_quad2d_env, load_policy
except Exception:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from examples.quadrotor.rollout_utils import make_quad2d_env, load_policy


METHODS = ["ssm", "rac", "sac_lag"]
METHOD_LABELS = {
    "ssm": "SSM",
    "rac": "RAC",
    "sac_lag": "SAC-Lag",
}


def _init_options(x: float, z: float) -> Dict[str, Any]:
    return {
        "init_x": float(x),
        "init_vx": 0.0,
        "init_z": float(z),
        "init_vz": 0.0,
        "init_theta": 0.0,
        "init_omega": 0.0,
        "init_waypoint_idx": 0,
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


def _rollout_short_safe(
    env,
    policy_fn,
    *,
    seed: int,
    horizon: int,
    init_opts: Dict[str, Any],
    context_prefix: str,
) -> bool:
    obs, _ = env.reset(seed=seed, options=init_opts)
    act_dim = env.action_space.shape[-1]
    for _ in range(horizon):
        action = policy_fn(obs)
        action = _validate_action(
            action,
            act_dim,
            context=f"{context_prefix} short_safe seed={seed} obs={np.asarray(obs)[:5]}",
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)
        step_out = env.step(action)
        if len(step_out) != 6:
            raise RuntimeError("cost wrapper missing: env.step must return 6 values")
        obs, _, cost, terminated, truncated, _ = step_out
        if terminated or truncated:
            return False
        if cost > 0.0:
            return False
    return True


def _rollout_tts(
    env,
    policy_fn,
    *,
    seed: int,
    horizon: int,
    stay_steps: int,
    cap_value: int,
    init_opts: Dict[str, Any],
    context_prefix: str,
) -> int:
    obs, _ = env.reset(seed=seed, options=init_opts)
    act_dim = env.action_space.shape[-1]
    consecutive_safe = 0
    for t in range(horizon):
        action = policy_fn(obs)
        action = _validate_action(
            action,
            act_dim,
            context=f"{context_prefix} tts seed={seed} obs={np.asarray(obs)[:5]}",
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)
        step_out = env.step(action)
        if len(step_out) != 6:
            raise RuntimeError("cost wrapper missing: env.step must return 6 values")
        obs, _, cost, terminated, truncated, _ = step_out
        if terminated or truncated:
            return cap_value
        if cost == 0.0:
            consecutive_safe += 1
        else:
            consecutive_safe = 0
        if consecutive_safe >= stay_steps:
            return t - stay_steps + 1
    return cap_value


def _compute_metrics_for_method(
    algo: str,
    ckpt_path: str,
    step: Optional[int],
    *,
    seed: int,
    deterministic: bool,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    mc_short: int,
    horizon_short: int,
    mc_tts: int,
    horizon_long: int,
    stay_steps: int,
    cap_value: int,
) -> Tuple[np.ndarray, np.ndarray]:
    env = make_quad2d_env(seed=seed)
    _, policy_fn, _ = load_policy(
        algo=algo,
        ckpt_path=ckpt_path,
        step=step,
        env=env,
        seed=seed,
        deterministic=deterministic,
    )

    grid_n = len(x_grid)
    p_safe = np.zeros((grid_n, grid_n), dtype=np.float32)
    tts = np.full((grid_n, grid_n), cap_value, dtype=np.float32)

    context_prefix = f"algo={algo} ckpt={ckpt_path} step={step}"
    for zi, z0 in enumerate(z_grid):
        for xi, x0 in enumerate(x_grid):
            init_opts = _init_options(x0, z0)
            safe_hits = 0
            try:
                for k in range(mc_short):
                    seed_k = seed + 100000 * (zi * grid_n + xi) + k
                    if _rollout_short_safe(
                        env,
                        policy_fn,
                        seed=seed_k,
                        horizon=horizon_short,
                        init_opts=init_opts,
                        context_prefix=context_prefix,
                    ):
                        safe_hits += 1
                p_safe[zi, xi] = safe_hits / float(mc_short)

                tts_vals = []
                for k in range(mc_tts):
                    seed_k = seed + 200000 * (zi * grid_n + xi) + k
                    tts_val = _rollout_tts(
                        env,
                        policy_fn,
                        seed=seed_k,
                        horizon=horizon_long,
                        stay_steps=stay_steps,
                        cap_value=cap_value,
                        init_opts=init_opts,
                        context_prefix=context_prefix,
                    )
                    tts_vals.append(tts_val)
                tts[zi, xi] = float(np.mean(tts_vals)) if tts_vals else float(cap_value)
            except Exception as exc:
                warnings.warn(
                    f"{algo} failed at x={x0:.2f}, z={z0:.2f}: {exc}",
                    RuntimeWarning,
                )
                p_safe[zi, xi] = 0.0
                tts[zi, xi] = float(cap_value)
    env.close()
    return p_safe, tts


def _plot_results(
    *,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    p_safe_all: np.ndarray,
    tts_all: np.ndarray,
    cap_value: int,
    horizon_short: int,
    mc_short: int,
    stay_steps: int,
    save_path: str,
    save_pdf: bool,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

    extent = [x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]]

    col1_title = f"Short-horizon safety probability (H={horizon_short}, MC={mc_short})"
    col2_title = f"Time-to-Enter-and-Stay (cap={cap_value}, stay={stay_steps})"

    col1_images = []
    col2_images = []

    for row, algo in enumerate(METHODS):
        label = METHOD_LABELS[algo]
        ax1 = axes[row, 0]
        ax2 = axes[row, 1]

        img1 = ax1.imshow(
            p_safe_all[row],
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
        )
        img2 = ax2.imshow(
            tts_all[row],
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=float(cap_value),
            cmap="viridis_r",
            aspect="auto",
        )
        col1_images.append(img1)
        col2_images.append(img2)

        ax1.set_ylabel(f"{label}\nz")
        ax2.set_ylabel(f"{label}\nz")
        for ax in (ax1, ax2):
            ax.set_xlabel("x")
            ax.axhline(0.5, color="black", linewidth=1.0)
            ax.axhline(1.5, color="black", linewidth=1.0)
            ax.axvline(-2.0, color="black", linewidth=1.0, alpha=0.6)
            ax.axvline(2.0, color="black", linewidth=1.0, alpha=0.6)

    axes[0, 0].set_title(col1_title)
    axes[0, 1].set_title(col2_title)

    cbar1 = fig.colorbar(col1_images[0], ax=axes[:, 0], fraction=0.02, pad=0.02)
    cbar1.set_label("p_safe")
    cbar2 = fig.colorbar(col2_images[0], ax=axes[:, 1], fraction=0.02, pad=0.02)
    cbar2.set_label(f"TTS (capped at {cap_value})")

    fig.savefig(save_path, dpi=dpi)
    if save_pdf:
        base, _ = os.path.splitext(save_path)
        fig.savefig(f"{base}.pdf", dpi=dpi)
    plt.close(fig)


def _resolve_save_path(
    save_dir: str,
    *,
    grid_n: int,
    horizon_short: int,
    mc_short: int,
    cap_value: int,
    stay_steps: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"q1_{timestamp}_grid{grid_n}_H{horizon_short}_MC{mc_short}"
        f"_TTS{cap_value}_stay{stay_steps}.png"
    )
    return os.path.join(save_dir, filename)


def _save_cache(
    cache_path: str,
    *,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    p_safe_all: np.ndarray,
    tts_all: np.ndarray,
    method_names: List[str],
    params: Dict[str, Any],
) -> None:
    np.savez(
        cache_path,
        x_grid=x_grid,
        z_grid=z_grid,
        p_safe=p_safe_all,
        tts=tts_all,
        method_names=np.asarray(method_names),
        params=params,
    )


def _load_cache(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    data = np.load(cache_path, allow_pickle=True)
    x_grid = data["x_grid"]
    z_grid = data["z_grid"]
    p_safe_all = data["p_safe"]
    tts_all = data["tts"]
    method_names = list(data["method_names"].tolist())
    params = dict(data["params"].item())
    return x_grid, z_grid, p_safe_all, tts_all, method_names, params


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 visualization for QuadrotorTracking2D-v0")
    parser.add_argument("--ckpt_ssm", required=True)
    parser.add_argument("--ckpt_rac", required=True)
    parser.add_argument("--ckpt_sac_lag", required=True)
    parser.add_argument("--step_ssm", type=int, default=None)
    parser.add_argument("--step_rac", type=int, default=None)
    parser.add_argument("--step_sac_lag", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--mc_short", type=int, default=10)
    parser.add_argument("--horizon_short", type=int, default=40)
    parser.add_argument("--mc_tts", type=int, default=1)
    parser.add_argument("--horizon_long", type=int, default=360)
    parser.add_argument("--stay_steps", type=int, default=10)
    parser.add_argument("--grid_n", type=int, default=41)
    parser.add_argument("--x_min", type=float, default=-2.0)
    parser.add_argument("--x_max", type=float, default=2.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--cache_path", default="")
    parser.add_argument("--no_cache", action="store_true", default=False)
    parser.add_argument("--save_dir", default=os.path.dirname(__file__))
    parser.add_argument("--save_pdf", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    cap_value = args.horizon_long
    x_grid = np.linspace(args.x_min, args.x_max, args.grid_n)
    z_grid = np.linspace(args.z_min, args.z_max, args.grid_n)

    cache_path = args.cache_path.strip()
    if cache_path and os.path.exists(cache_path) and not args.no_cache:
        x_grid, z_grid, p_safe_all, tts_all, method_names, params = _load_cache(cache_path)
    else:
        p_safe_all = np.zeros((len(METHODS), args.grid_n, args.grid_n), dtype=np.float32)
        tts_all = np.zeros((len(METHODS), args.grid_n, args.grid_n), dtype=np.float32)
        method_names = METHODS
        params = {
            "mc_short": args.mc_short,
            "horizon_short": args.horizon_short,
            "mc_tts": args.mc_tts,
            "horizon_long": args.horizon_long,
            "stay_steps": args.stay_steps,
            "grid_n": args.grid_n,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "seed": args.seed,
            "deterministic": args.deterministic,
        }

        ckpt_map = {
            "ssm": (args.ckpt_ssm, args.step_ssm),
            "rac": (args.ckpt_rac, args.step_rac),
            "sac_lag": (args.ckpt_sac_lag, args.step_sac_lag),
        }

        for idx, algo in enumerate(METHODS):
            ckpt_path, step = ckpt_map[algo]
            p_safe, tts = _compute_metrics_for_method(
                algo,
                ckpt_path,
                step,
                seed=args.seed,
                deterministic=args.deterministic,
                x_grid=x_grid,
                z_grid=z_grid,
                mc_short=args.mc_short,
                horizon_short=args.horizon_short,
                mc_tts=args.mc_tts,
                horizon_long=args.horizon_long,
                stay_steps=args.stay_steps,
                cap_value=cap_value,
            )
            p_safe_all[idx] = p_safe
            tts_all[idx] = tts

        if cache_path and not args.no_cache:
            _save_cache(
                cache_path,
                x_grid=x_grid,
                z_grid=z_grid,
                p_safe_all=p_safe_all,
                tts_all=tts_all,
                method_names=method_names,
                params=params,
            )

    save_path = _resolve_save_path(
        args.save_dir,
        grid_n=args.grid_n,
        horizon_short=args.horizon_short,
        mc_short=args.mc_short,
        cap_value=cap_value,
        stay_steps=args.stay_steps,
    )

    _plot_results(
        x_grid=x_grid,
        z_grid=z_grid,
        p_safe_all=p_safe_all,
        tts_all=tts_all,
        cap_value=cap_value,
        horizon_short=args.horizon_short,
        mc_short=args.mc_short,
        stay_steps=args.stay_steps,
        save_path=save_path,
        save_pdf=args.save_pdf,
        dpi=args.dpi,
    )

    if cache_path and os.path.exists(cache_path):
        cache_used = cache_path
    else:
        cache_used = "(no cache)"
    print(f"[OK] saved: {save_path}")
    print(f"[OK] cache: {cache_used}")


if __name__ == "__main__":
    main()
