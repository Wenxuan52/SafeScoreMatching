"""Q2 Figure: HJ maximal feasible set (K) vs closed-loop feasible set (K_pi).

Example:
python examples/quadrotor/q2_visual.py \
  --hj_path /ABS/PATH/TO/hj_grid_quad2d.npz \
  --ckpt_ssm /ABS/PATH/TO/ssm_run_dir --step_ssm 1500000 \
  --ckpt_rac /ABS/PATH/TO/rac_run_dir --step_rac 1500000 \
  --ckpt_sac_lag /ABS/PATH/TO/sac_lag_run_dir --step_sac_lag 1500000 \
  --vz_slices -1 0 1 \
  --grid_n 41 --mc 10 --horizon 80 \
  --x_min -2 --x_max 2 --z_min 0 --z_max 2 \
  --cache_path examples/quadrotor/figures/q2_cache.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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


def _default_init_options(x: float, z: float, vz: float) -> Dict[str, Any]:
    return {
        "init_x": float(x),
        "init_vx": 0.0,
        "init_z": float(z),
        "init_vz": float(vz),
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


def _compute_h_from_obs(obs: np.ndarray) -> float:
    x = float(obs[0])
    z = float(obs[2])
    h_corridor = max(0.5 - z, z - 1.5)
    h_oob_x = max(0.0, abs(x) - 2.0)
    h_oob_z = max(0.0, abs(z) - 3.0)
    return float(max(h_corridor, h_oob_x, h_oob_z))


def _resolve_hj_keys(data: Dict[str, Any], key: Optional[str], candidates: List[str]) -> str:
    if key:
        return key
    for cand in candidates:
        if cand in data:
            return cand
    raise KeyError(f"Unable to infer key from candidates {candidates}")


def _coerce_hj_grid(data: Dict[str, Any], *, v_key: str, x_key: str, z_key: str, vz_key: str):
    V = np.asarray(data[v_key])
    vz_slices = np.asarray(data[vz_key]).astype(np.float32)

    if x_key in data and z_key in data:
        x = np.asarray(data[x_key])
        z = np.asarray(data[z_key])
        if x.ndim == 2 and z.ndim == 2:
            X = x
            Z = z
            if V.ndim != 3:
                raise ValueError("HJ value V must be 3D when X/Z are meshgrids")
            nz, nx = X.shape
            Vh = _coerce_v_shape(V, len(vz_slices), nz, nx)
            return X, Z, vz_slices, Vh
        if x.ndim == 1 and z.ndim == 1:
            X, Z = np.meshgrid(x, z)
            Vh = _coerce_v_shape(V, len(vz_slices), len(z), len(x))
            return X, Z, vz_slices, Vh
    raise ValueError("Unsupported HJ grid format")


def _coerce_v_shape(V: np.ndarray, k: int, nz: int, nx: int) -> np.ndarray:
    if V.shape == (k, nz, nx):
        return V
    if V.shape == (k, nx, nz):
        return V.transpose(0, 2, 1)
    if V.shape == (nz, nx, k):
        return V.transpose(2, 0, 1)
    if V.shape == (nx, nz, k):
        return V.transpose(2, 1, 0)
    raise ValueError(f"Unexpected V shape {V.shape}, expected (K,Nz,Nx) variants")


def load_hj_data(
    hj_path: str,
    *,
    v_key: Optional[str],
    x_key: Optional[str],
    z_key: Optional[str],
    vz_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(hj_path, allow_pickle=True)
    data_dict = {k: data[k] for k in data.files}

    v_key_resolved = _resolve_hj_keys(data_dict, v_key, ["V", "Vh", "value", "values"])
    x_key_resolved = _resolve_hj_keys(data_dict, x_key, ["x_grid", "xs", "x", "X"])
    z_key_resolved = _resolve_hj_keys(data_dict, z_key, ["z_grid", "zs", "z", "Z"])
    vz_key_resolved = _resolve_hj_keys(data_dict, vz_key, ["vz_slices", "vzs", "vz"])

    X, Z, vz_slices, V = _coerce_hj_grid(
        data_dict,
        v_key=v_key_resolved,
        x_key=x_key_resolved,
        z_key=z_key_resolved,
        vz_key=vz_key_resolved,
    )

    meta = {
        "hj_path": hj_path,
        "v_key": v_key_resolved,
        "x_key": x_key_resolved,
        "z_key": z_key_resolved,
        "vz_key": vz_key_resolved,
    }
    return X, Z, vz_slices, V, meta


def _match_vz_indices(requested: np.ndarray, available: np.ndarray) -> List[int]:
    indices = []
    for vz in requested:
        idx = int(np.argmin(np.abs(available - vz)))
        if not np.isclose(available[idx], vz):
            warnings.warn(
                f"Requested vz={vz} not found; using closest {available[idx]}",
                RuntimeWarning,
            )
        indices.append(idx)
    return indices


def _compute_e_max_h(
    env,
    policy_fn,
    *,
    seed: int,
    horizon: int,
    mc: int,
    done_penalty: float,
    init_opts: Dict[str, Any],
    seed_base: int,
    context_prefix: str,
) -> float:
    act_dim = env.action_space.shape[-1]
    max_h_vals = []
    for k in range(mc):
        seed_k = seed_base + k
        obs, _ = env.reset(seed=seed_k, options=init_opts)
        max_h = 0.0
        try:
            for _ in range(horizon):
                action = policy_fn(obs)
                action = _validate_action(
                    action,
                    act_dim,
                    context=f"{context_prefix} seed={seed_k} obs={np.asarray(obs)[:5]}",
                )
                action = np.clip(action, env.action_space.low, env.action_space.high)
                step_out = env.step(action)
                if len(step_out) != 6:
                    raise RuntimeError("cost wrapper missing: env.step must return 6 values")
                next_obs, _, _, terminated, truncated, _ = step_out
                max_h = max(max_h, _compute_h_from_obs(next_obs))
                if terminated or truncated:
                    max_h = max(max_h, done_penalty)
                    break
                obs = next_obs
        except Exception as exc:
            warnings.warn(f"{context_prefix} rollout error: {exc}", RuntimeWarning)
            max_h = max(max_h, done_penalty)
        max_h_vals.append(max_h)
    return float(np.mean(max_h_vals)) if max_h_vals else float(done_penalty)


def _compute_metrics(
    algo: str,
    ckpt_path: str,
    step: Optional[int],
    *,
    env_name: str,
    seed: int,
    deterministic: bool,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    vz_slices: np.ndarray,
    mc: int,
    horizon: int,
    done_penalty: float,
    seed_stride: int,
) -> np.ndarray:
    env = make_quad2d_env(env_name=env_name, seed=seed)
    _, policy_fn, _ = load_policy(
        algo=algo,
        ckpt_path=ckpt_path,
        step=step,
        env=env,
        seed=seed,
        deterministic=deterministic,
    )

    nz = len(z_grid)
    nx = len(x_grid)
    k = len(vz_slices)
    e_max_h = np.full((k, nz, nx), done_penalty, dtype=np.float32)

    for vz_idx, vz in enumerate(vz_slices):
        for zi, z0 in enumerate(z_grid):
            for xi, x0 in enumerate(x_grid):
                init_opts = _default_init_options(x0, z0, vz)
                seed_base = seed_stride * (vz_idx * nz * nx + zi * nx + xi) + seed
                context_prefix = (
                    f"algo={algo} vz={vz} x={x0:.2f} z={z0:.2f} step={step}"
                )
                try:
                    e_max_h[vz_idx, zi, xi] = _compute_e_max_h(
                        env,
                        policy_fn,
                        seed=seed,
                        horizon=horizon,
                        mc=mc,
                        done_penalty=done_penalty,
                        init_opts=init_opts,
                        seed_base=seed_base,
                        context_prefix=context_prefix,
                    )
                except Exception as exc:
                    warnings.warn(f"{context_prefix} failed: {exc}", RuntimeWarning)
                    e_max_h[vz_idx, zi, xi] = float(done_penalty)

    env.close()
    return e_max_h


def _resolve_save_path(save_dir: str, *, suffix: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(save_dir, f"q2_{timestamp}_{suffix}.png")


def _plot_results(
    *,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    vz_slices: np.ndarray,
    method_names: List[str],
    e_max_h: np.ndarray,
    hj_X: np.ndarray,
    hj_Z: np.ndarray,
    hj_V: np.ndarray,
    hj_vz_indices: List[int],
    vmax: float,
    plot_x_bounds: bool,
    save_path: str,
    save_pdf: bool,
    dpi: int,
    show: bool,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    extent = [x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]]

    for row, algo in enumerate(method_names):
        for col, vz in enumerate(vz_slices):
            ax = axes[row, col]
            img = ax.imshow(
                e_max_h[row, col],
                origin="lower",
                extent=extent,
                vmin=0.0,
                vmax=vmax,
                cmap="viridis",
                aspect="auto",
            )

            hj_idx = hj_vz_indices[col]
            try:
                ax.contour(
                    hj_X,
                    hj_Z,
                    hj_V[hj_idx],
                    levels=[0.0],
                    colors="red",
                    linestyles="--",
                    linewidths=1.5,
                )
            except Exception as exc:
                warnings.warn(f"HJ contour failed for vz={vz}: {exc}", RuntimeWarning)

            try:
                ax.contour(
                    x_grid,
                    z_grid,
                    e_max_h[row, col],
                    levels=[0.0],
                    colors="white",
                    linestyles="-",
                    linewidths=1.5,
                )
            except Exception as exc:
                warnings.warn(f"K_pi contour failed for {algo} vz={vz}: {exc}", RuntimeWarning)

            ax.axhline(0.5, color="black", linewidth=1.0)
            ax.axhline(1.5, color="black", linewidth=1.0)
            if plot_x_bounds:
                ax.axvline(-2.0, color="black", linewidth=1.0, alpha=0.6)
                ax.axvline(2.0, color="black", linewidth=1.0, alpha=0.6)

            if row == 0:
                ax.set_title(f"vz={vz}")
            if col == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(algo, algo)}\nz")
            ax.set_xlabel("x")

    cbar = fig.colorbar(img, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("E[max_{t<=H} h(s_t)]")

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
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    vz_slices: np.ndarray,
    method_names: List[str],
    e_max_h: np.ndarray,
    plot_meta: Dict[str, Any],
    hj_meta: Dict[str, Any],
) -> None:
    np.savez(
        cache_path,
        x_grid=x_grid,
        z_grid=z_grid,
        vz_slices=vz_slices,
        method_names=np.asarray(method_names),
        E_max_h=e_max_h,
        plot_meta=json.dumps(plot_meta),
        hj_meta=json.dumps(hj_meta),
    )


def _load_cache(cache_path: str):
    data = np.load(cache_path, allow_pickle=True)
    x_grid = data["x_grid"]
    z_grid = data["z_grid"]
    vz_slices = data["vz_slices"]
    method_names = list(data["method_names"].tolist())
    e_max_h = data["E_max_h"]
    plot_meta = json.loads(str(data["plot_meta"]))
    hj_meta = json.loads(str(data["hj_meta"]))
    return x_grid, z_grid, vz_slices, method_names, e_max_h, plot_meta, hj_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 visualization for QuadrotorTracking2D-v0")
    parser.add_argument("--hj_path", required=True)
    parser.add_argument("--hj_v_key", default=None)
    parser.add_argument("--hj_x_key", default=None)
    parser.add_argument("--hj_z_key", default=None)
    parser.add_argument("--hj_vz_key", default=None)

    parser.add_argument("--ckpt_ssm", required=True)
    parser.add_argument("--ckpt_rac", required=True)
    parser.add_argument("--ckpt_sac_lag", required=True)
    parser.add_argument("--step_ssm", type=int, default=None)
    parser.add_argument("--step_rac", type=int, default=None)
    parser.add_argument("--step_sac_lag", type=int, default=None)

    parser.add_argument("--env_name", default="QuadrotorTracking2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--mc", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--done_penalty", type=float, default=1.0)
    parser.add_argument("--stay_seed_stride", type=int, default=100000)

    parser.add_argument("--grid_n", type=int, default=41)
    parser.add_argument("--x_min", type=float, default=-2.0)
    parser.add_argument("--x_max", type=float, default=2.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--vz_slices", type=float, nargs="+", default=[-1.0, 0.0, 1.0])

    parser.add_argument("--save_dir", default="examples/quadrotor/figures")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--save_pdf", action="store_true", default=False)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--plot_x_bounds", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--cache_path", default="")
    parser.add_argument("--no_cache", action="store_true", default=False)

    args = parser.parse_args()

    vz_slices = np.asarray(args.vz_slices, dtype=np.float32)
    x_grid = np.linspace(args.x_min, args.x_max, args.grid_n)
    z_grid = np.linspace(args.z_min, args.z_max, args.grid_n)

    cache_path = args.cache_path.strip()
    if cache_path and os.path.exists(cache_path) and not args.no_cache:
        x_grid, z_grid, vz_slices, method_names, e_max_h, plot_meta, hj_meta = _load_cache(cache_path)
    else:
        method_names = METHODS
        e_max_h = np.zeros((len(METHODS), len(vz_slices), len(z_grid), len(x_grid)), dtype=np.float32)
        plot_meta = {
            "horizon": args.horizon,
            "mc": args.mc,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "done_penalty": args.done_penalty,
            "grid_n": args.grid_n,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "vz_slices": args.vz_slices,
            "seed_stride": args.stay_seed_stride,
        }
        hj_meta = {
            "hj_path": args.hj_path,
            "hj_v_key": args.hj_v_key,
            "hj_x_key": args.hj_x_key,
            "hj_z_key": args.hj_z_key,
            "hj_vz_key": args.hj_vz_key,
        }

        ckpt_map = {
            "ssm": (args.ckpt_ssm, args.step_ssm),
            "rac": (args.ckpt_rac, args.step_rac),
            "sac_lag": (args.ckpt_sac_lag, args.step_sac_lag),
        }

        for midx, algo in enumerate(METHODS):
            ckpt_path, step = ckpt_map[algo]
            e_max_h[midx] = _compute_metrics(
                algo,
                ckpt_path,
                step,
                env_name=args.env_name,
                seed=args.seed,
                deterministic=args.deterministic,
                x_grid=x_grid,
                z_grid=z_grid,
                vz_slices=vz_slices,
                mc=args.mc,
                horizon=args.horizon,
                done_penalty=args.done_penalty,
                seed_stride=args.stay_seed_stride,
            )

        if cache_path and not args.no_cache:
            _save_cache(
                cache_path,
                x_grid=x_grid,
                z_grid=z_grid,
                vz_slices=vz_slices,
                method_names=method_names,
                e_max_h=e_max_h,
                plot_meta=plot_meta,
                hj_meta=hj_meta,
            )

    hj_X, hj_Z, hj_vz_slices, hj_V, hj_meta_loaded = load_hj_data(
        args.hj_path,
        v_key=args.hj_v_key,
        x_key=args.hj_x_key,
        z_key=args.hj_z_key,
        vz_key=args.hj_vz_key,
    )

    hj_vz_indices = _match_vz_indices(vz_slices, hj_vz_slices)

    all_values = e_max_h[np.isfinite(e_max_h)]
    vmax = float(np.nanpercentile(all_values, 95)) if all_values.size else 1.0
    vmax = max(vmax, 1e-6)

    save_path = _resolve_save_path(
        args.save_dir,
        suffix=f"grid{len(x_grid)}_H{args.horizon}_MC{args.mc}",
    )

    _plot_results(
        x_grid=x_grid,
        z_grid=z_grid,
        vz_slices=vz_slices,
        method_names=method_names,
        e_max_h=e_max_h,
        hj_X=hj_X,
        hj_Z=hj_Z,
        hj_V=hj_V,
        hj_vz_indices=hj_vz_indices,
        vmax=vmax,
        plot_x_bounds=args.plot_x_bounds,
        save_path=save_path,
        save_pdf=args.save_pdf,
        dpi=args.dpi,
        show=args.show,
    )

    if cache_path and os.path.exists(cache_path):
        cache_used = cache_path
    else:
        cache_used = "(no cache)"
    print(f"[OK] saved: {save_path}")
    print(f"[OK] cache: {cache_used}")


if __name__ == "__main__":
    main()
