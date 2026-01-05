import inspect
import json

import jax
import jax.numpy as jnp
from flax.training import checkpoints

import importlib
import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jaxrl5.agents import SafeScoreMatchingLearner, TD3LagLearner, TD3Learner
from jaxrl5.envs import make_env


def _resolve_checkpoint_path(base_dir: str, step: Optional[int]) -> str:
    """Locate a checkpoint directory or file, supporting step_* layout."""

    if os.path.isfile(base_dir):
        return base_dir

    def _find_in_dir(dir_path: str) -> Optional[str]:
        candidates = glob.glob(os.path.join(dir_path, "ckpt_*"))
        if candidates:
            return sorted(candidates)[-1]
        return None

    ckpt_root = base_dir
    if os.path.isdir(base_dir) and os.path.isdir(os.path.join(base_dir, "checkpoints")):
        ckpt_root = os.path.join(base_dir, "checkpoints")

    if step is not None:
        step_dir = os.path.join(ckpt_root, f"step_{step}")
        if os.path.isdir(step_dir):
            candidate = _find_in_dir(step_dir)
            if candidate:
                return candidate
        # Fallback to direct file under ckpt_root
        direct = os.path.join(ckpt_root, f"ckpt_{step}")
        msgpack = direct + ".msgpack"
        for cand in (direct, msgpack):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"Checkpoint for step {step} not found under {ckpt_root}")

    # Latest checkpoint search
    step_dirs = sorted(glob.glob(os.path.join(ckpt_root, "step_*")))
    if step_dirs:
        latest = step_dirs[-1]
        candidate = _find_in_dir(latest)
        if candidate:
            return candidate

    candidate = _find_in_dir(ckpt_root)
    if candidate:
        return candidate

    raise FileNotFoundError(f"No checkpoints found under {ckpt_root}")


def _unwrap_agent(obj, expected_type):
    if isinstance(obj, expected_type):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, expected_type):
                return v
    raise TypeError(
        f"Loaded checkpoint is of type {type(obj)}; expected {expected_type.__name__}."
    )

def _load_json_config_near_ckpt(ckpt_path: str) -> Dict:
    """Try to load a config json near the run dir, if present."""
    # ckpt_path might be .../checkpoints/ckpt_xxx.msgpack
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))  # up to .../<run>/
    candidates = [
        os.path.join(run_dir, "config.json"),
        os.path.join(run_dir, "args.json"),
        os.path.join(run_dir, "flags.json"),
        os.path.join(run_dir, "params.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _create_ssm_learner_from_env(
    env,
    seed: int,
    cfg: Optional[Dict] = None,
    ckpt_state: Optional[Dict] = None,
):
    """
    Create a SafeScoreMatchingLearner with kwargs matched to training-time hyperparams.
    Priority:
      1) explicit cfg (config.json/args.json near run dir)
      2) inferred from ckpt_state (hidden dims / diffusion steps / fourier features)
      3) fall back to create() defaults
    """
    cfg = cfg or {}
    sig = inspect.signature(SafeScoreMatchingLearner.create)
    params = sig.parameters

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    kwargs = {}

    def set_if_exists(name: str, value):
        if name in params and name not in kwargs:
            kwargs[name] = value

    # ---- common arguments ----
    set_if_exists("env", env)
    set_if_exists("observation_space", env.observation_space)
    set_if_exists("action_space", env.action_space)

    set_if_exists("seed", seed)
    set_if_exists("rng", jax.random.PRNGKey(seed))

    for n in ("obs_dim", "observation_dim", "state_dim", "observation_size"):
        set_if_exists(n, obs_dim)
    for n in ("action_dim", "act_dim", "action_size"):
        set_if_exists(n, act_dim)

    # ---- fill from cfg if names match create(...) signature ----
    for k, v in cfg.items():
        if k in params and k not in kwargs:
            kwargs[k] = v

    # ---- infer from ckpt if still missing ----
    if ckpt_state is not None:
        # 1) hidden dims (fixes your current (142,512) vs (142,256) mismatch)
        inferred_hidden = _infer_ssm_hidden_dims_from_ckpt(ckpt_state)
        if inferred_hidden is not None:
            for name in (
                "hidden_dims",
                "mlp_dims",
                "score_model_hidden_dims",
                "reverse_hidden_dims",
                "policy_hidden_dims",
                "actor_hidden_dims",
            ):
                set_if_exists(name, inferred_hidden)

        # 2) diffusion steps (T)
        inferred_T = _infer_ssm_num_timesteps_from_ckpt(ckpt_state)
        if inferred_T is not None:
            for name in ("num_timesteps", "diffusion_steps", "n_timesteps", "T"):
                set_if_exists(name, inferred_T)

        # 3) fourier features count
        inferred_ff = _infer_ssm_fourier_features_from_ckpt(ckpt_state)
        if inferred_ff is not None:
            for name in ("fourier_features", "fourier_dim", "num_fourier_features"):
                set_if_exists(name, inferred_ff)

    # ---- sanity: detect missing required args early ----
    missing = []
    for p in params.values():
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is inspect._empty:
            if p.name not in kwargs:
                missing.append(p.name)
    if missing:
        raise TypeError(
            f"SafeScoreMatchingLearner.create missing required args: {missing}\n"
            f"Provided kwargs: {sorted(kwargs.keys())}\n"
            f"Tip: put these into config.json/args.json near the run dir, "
            f"or extend _create_ssm_learner_from_env to infer them from ckpt."
        )

    return SafeScoreMatchingLearner.create(**kwargs)


def _restore_trainstate(ts, d):
    if d is None or not isinstance(d, dict):
        return ts
    return ts.replace(
        step=d.get("step", ts.step),
        params=d.get("params", ts.params),
        opt_state=d.get("opt_state", ts.opt_state),
    )

def _extract_td3_cfg(cfg_json: Dict) -> Dict:
    """
    train_td3_quad2d_baseline.py 保存的 config.json 结构是：
      { "flags": {...}, "td3_config": {...} }
    这里把 td3_config 提取出来。
    """
    if not isinstance(cfg_json, dict):
        return {}
    if "td3_config" in cfg_json and isinstance(cfg_json["td3_config"], dict):
        return dict(cfg_json["td3_config"])
    # 兼容：直接把 cfg_json 当 td3_config
    return dict(cfg_json)


def _resolve_model_cls(model_cls: str):
    """
    model_cls 通常是 "TD3Learner"（也可能是别名），尽量解析到类对象。
    """
    if not model_cls:
        return TD3Learner

    # 1) 当前模块 globals
    if model_cls in globals() and isinstance(globals()[model_cls], type):
        return globals()[model_cls]

    # 2) jaxrl5.agents.td3.td3_learner
    try:
        mod = importlib.import_module("jaxrl5.agents.td3.td3_learner")
        if hasattr(mod, model_cls):
            return getattr(mod, model_cls)
    except Exception:
        pass

    # 3) jaxrl5.agents.td3.td3_lag_learner
    try:
        mod = importlib.import_module("jaxrl5.agents.td3.td3_lag_learner")
        if hasattr(mod, model_cls):
            return getattr(mod, model_cls)
    except Exception:
        pass

    # 4) fallback
    return TD3Learner


def _build_td3_from_state_dict(state: Dict, env, seed: int, cfg_json: Dict):
    """
    用 config.json 里的 td3_config 重建 TD3Learner，再把 state dict 回填。
    """
    td3_cfg = _extract_td3_cfg(cfg_json)

    kwargs = dict(td3_cfg)
    model_cls_name = kwargs.pop("model_cls", None)
    cls = _resolve_model_cls(model_cls_name)

    # 这里用和训练脚本一致的 create 方式：create(seed, obs_space, act_space, **kwargs)
    agent = cls.create(
        seed,
        env.observation_space,
        env.action_space,
        **kwargs,
    )

    # 回填 TrainState（actor / critic / target_*）
    if "actor" in state and hasattr(agent, "actor"):
        agent = agent.replace(actor=_restore_trainstate(agent.actor, state["actor"]))
    if "critic" in state and hasattr(agent, "critic"):
        agent = agent.replace(critic=_restore_trainstate(agent.critic, state["critic"]))
    if "target_actor" in state and hasattr(agent, "target_actor"):
        agent = agent.replace(target_actor=_restore_trainstate(agent.target_actor, state["target_actor"]))
    if "target_critic" in state and hasattr(agent, "target_critic"):
        agent = agent.replace(target_critic=_restore_trainstate(agent.target_critic, state["target_critic"]))

    # 回填标量/数组字段（best-effort）
    for k in [
        "rng",
        "tau",
        "discount",
        "exploration_noise",
        "target_policy_noise",
        "target_policy_noise_clip",
        "actor_delay",
    ]:
        if k in state and hasattr(agent, k):
            agent = agent.replace(**{k: jnp.asarray(state[k])})

    return agent


def _restore_ssm_from_state_dict(learner, state: Dict):
    # actor in your ckpt is None; keep None
    if "actor" in state and state["actor"] is None and hasattr(learner, "actor"):
        learner = learner.replace(actor=None)

    if "rng" in state and state["rng"] is not None and hasattr(learner, "rng"):
        learner = learner.replace(rng=jnp.asarray(state["rng"]))

    # trainstates
    for name in [
        "score_model", "critic_1", "critic_2",
        "target_critic_1", "target_critic_2",
        "safety_critic", "target_safety_critic",
    ]:
        if name in state and hasattr(learner, name):
            learner = learner.replace(**{name: _restore_trainstate(getattr(learner, name), state[name])})

    # scalars/arrays (best-effort)
    for k in [
        "discount", "tau", "safety_tau", "ddpm_temperature", "M_q", "cost_limit",
        "safety_discount", "safety_lambda", "alpha_coef", "safety_threshold",
        "safety_grad_scale", "safe_lagrange_coef",
        "betas", "alphas", "alpha_hats",
    ]:
        if k in state and state[k] is not None and hasattr(learner, k):
            learner = learner.replace(**{k: jnp.asarray(state[k])})

    return learner



def _build_ssm_from_state_dict(state: Dict, env, seed: int, cfg: Dict):
    learner = _create_ssm_learner_from_env(env, seed=seed, cfg=cfg)

    # actor is None in your ckpt; keep it None
    learner = learner.replace(actor=None)

    # rng / schedules / scalars
    if "rng" in state and state["rng"] is not None:
        learner = learner.replace(rng=jnp.asarray(state["rng"]))

    # Restore trainstates
    if "score_model" in state:
        learner = learner.replace(score_model=_restore_trainstate(learner.score_model, state["score_model"]))
    if "critic_1" in state:
        learner = learner.replace(critic_1=_restore_trainstate(learner.critic_1, state["critic_1"]))
    if "critic_2" in state:
        learner = learner.replace(critic_2=_restore_trainstate(learner.critic_2, state["critic_2"]))
    if "target_critic_1" in state:
        learner = learner.replace(target_critic_1=_restore_trainstate(learner.target_critic_1, state["target_critic_1"]))
    if "target_critic_2" in state:
        learner = learner.replace(target_critic_2=_restore_trainstate(learner.target_critic_2, state["target_critic_2"]))
    if "safety_critic" in state:
        learner = learner.replace(safety_critic=_restore_trainstate(learner.safety_critic, state["safety_critic"]))
    if "target_safety_critic" in state:
        learner = learner.replace(target_safety_critic=_restore_trainstate(learner.target_safety_critic, state["target_safety_critic"]))

    # Scalars / arrays (best-effort)
    scalar_keys = [
        "discount", "tau", "safety_tau", "ddpm_temperature", "M_q", "cost_limit",
        "safety_discount", "safety_lambda", "alpha_coef", "safety_threshold",
        "safety_grad_scale", "safe_lagrange_coef",
    ]
    array_keys = ["betas", "alphas", "alpha_hats"]

    repl = {}
    for k in scalar_keys:
        if k in state and state[k] is not None and hasattr(learner, k):
            repl[k] = jnp.asarray(state[k])
    for k in array_keys:
        if k in state and state[k] is not None and hasattr(learner, k):
            repl[k] = jnp.asarray(state[k])
    if repl:
        learner = learner.replace(**repl)

    return learner

def _infer_ssm_hidden_dims_from_ckpt(state: Dict) -> Optional[Tuple[int, ...]]:
    """Infer score-model hidden dims from checkpoint params (best-effort)."""
    try:
        k0 = state["score_model"]["params"]["MLP_1"]["Dense_0"]["kernel"]  # (in_dim, h0)
        h0 = int(k0.shape[1])
        # try get second layer width too
        k1 = state["score_model"]["params"]["MLP_1"]["Dense_1"]["kernel"]  # (h0, h1) usually (h0, h0)
        h1 = int(k1.shape[1])
        return (h0, h1)
    except Exception:
        return None


def _infer_ssm_num_timesteps_from_ckpt(state: Dict) -> Optional[int]:
    try:
        return int(np.asarray(state["betas"]).shape[0])
    except Exception:
        return None

def _infer_ssm_fourier_features_from_ckpt(state: Dict) -> Optional[int]:
    # FourierFeatures_0 kernel shape is typically (num_features, input_dim)
    try:
        ff = state["score_model"]["params"]["FourierFeatures_0"]["kernel"]
        return int(ff.shape[0])
    except Exception:
        return None



def _load_agent(agent_name: str, checkpoint_path: str, env, seed: int) -> object:
    if agent_name == "ssm":
        try:
            loaded = SafeScoreMatchingLearner.load(checkpoint_path)
            return _unwrap_agent(loaded, SafeScoreMatchingLearner)
        except TypeError:
            # dict-style checkpoint
            state = checkpoints.restore_checkpoint(checkpoint_path, target=None)

            # 可选：读取 run dir 下的 config.json/args.json（如果你有）
            cfg = _load_json_config_near_ckpt(checkpoint_path) if "_load_json_config_near_ckpt" in globals() else {}

            learner = _create_ssm_learner_from_env(env, seed=seed, cfg=cfg, ckpt_state=state)
            learner = _restore_ssm_from_state_dict(learner, state)
            return learner

    if agent_name in ("td3", "td3_lag"):
        if agent_name == "td3_lag":
            try:
                loaded = TD3LagLearner.load(checkpoint_path)
                return _unwrap_agent(loaded, TD3LagLearner)
            except Exception:
                pass

        try:
            loaded = TD3Learner.load(checkpoint_path)
            return _unwrap_agent(loaded, TD3Learner)
        except TypeError:
            # dict-style checkpoint fallback
            state = checkpoints.restore_checkpoint(checkpoint_path, target=None)
            cfg_json = _load_json_config_near_ckpt(checkpoint_path)
            return _build_td3_from_state_dict(state, env, seed=seed, cfg_json=cfg_json)


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
        obs, info = env.reset(seed=seed + ep, options={
            "init_x": 1.0, "init_z": 1.0,
            "init_vx": 0.0, "init_vz": 0.0,
            "init_theta": 0.0, "init_omega": 0.0,
            "init_waypoint_idx": 0,   # 可选
            })

        print("start x,z =", float(obs[0]), float(obs[2]))
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
    parser.add_argument("--agent", choices=["td3", "td3_lag", "ssm"], required=True)
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory or file")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Specific checkpoint step")
    parser.add_argument("--out_dir", default="results/visualizations")
    parser.add_argument("--save_csv", action="store_true")
    args = parser.parse_args()

    env = make_env(args.env_name, seed=args.seed)
    ckpt_path = _resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_step)
    agent = _load_agent(args.agent, ckpt_path, env, seed=args.seed)

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
