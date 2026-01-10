import inspect
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from flax import serialization
from flax.core import frozen_dict

from jaxrl5.agents.cal.cal_learner import CALAgent
from jaxrl5.tools.checkpoints import resolve_checkpoint


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _extract_step(path: str) -> Optional[int]:
    m = re.search(r"ckpt_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _find_config_path(run_dir: Path) -> Path:
    candidates = [
        "config.json",
        "variant.json",
        "flags.json",
        "args.json",
        "params.json",
    ]
    for name in candidates:
        cand = run_dir / name
        if cand.exists():
            return cand
    listing = sorted(os.listdir(run_dir))[:50]
    raise FileNotFoundError(
        f"No config file found in {run_dir}. Tried {candidates}. Contents (first 50): {listing}"
    )


def _load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_create_kwargs(cfg: Dict) -> Dict:
    sig = inspect.signature(CALAgent.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def _extract_config(cfg: Dict) -> Dict:
    if "config" in cfg and isinstance(cfg["config"], dict):
        return cfg["config"]
    return cfg


def _build_policy_fn(
    state_holder: Dict[str, CALAgent],
    *,
    deterministic: bool,
) -> PolicyFn:
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        single = obs_np.ndim == 1
        if single:
            obs_np = obs_np[None]

        if deterministic:
            actions, new_agent = state_holder["agent"].eval_actions(obs_np)
        else:
            actions, new_agent = state_holder["agent"].sample_actions(obs_np)

        state_holder["agent"] = new_agent
        if single:
            actions = actions[0]
        return np.asarray(actions, dtype=np.float32)

    return policy_fn


def load_cal(
    ckpt_path: str,
    step: Optional[int] = None,
    *,
    observation_space=None,
    action_space=None,
    seed: int = 0,
    config_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    deterministic: bool = True,
    **kwargs,
) -> Tuple[CALAgent, PolicyFn, Dict]:
    if observation_space is None or action_space is None:
        raise ValueError("load_cal requires observation_space and action_space to build a template agent.")

    resolved = resolve_checkpoint(ckpt_path, step)

    # Prefer direct class loader if available.
    direct_agent = None
    if hasattr(CALAgent, "load"):
        try:
            direct_agent = CALAgent.load(resolved)
        except Exception:
            direct_agent = None

    config_path_used: Optional[Path] = None
    if isinstance(direct_agent, CALAgent):
        agent = direct_agent
    else:
        data = Path(resolved).read_bytes()
        # Prefer from_bytes first (works for direct msgpack serialization).
        try:
            agent = serialization.from_bytes(CALAgent, data)
        except Exception:
            state = None
            if hasattr(serialization, "msgpack_restore"):
                state = serialization.msgpack_restore(data)

            if isinstance(state, CALAgent):
                agent = state
            elif isinstance(state, (dict, frozen_dict.FrozenDict)):
                resolved_run_dir = (
                    Path(run_dir) if run_dir is not None else Path(resolved).parent.parent
                )
                try:
                    config_path_used = (
                        Path(config_path)
                        if config_path is not None
                        else _find_config_path(resolved_run_dir)
                    )
                    cfg_dict = _extract_config(_load_config(config_path_used))
                except FileNotFoundError:
                    cfg_dict = {}

                filtered_cfg = _filter_create_kwargs(cfg_dict)
                # Allow explicit kwargs to override when config isn't available.
                filtered_cfg.update(_filter_create_kwargs(kwargs))
                template = CALAgent.create(
                    seed=seed,
                    observation_space=observation_space,
                    action_space=action_space,
                    **filtered_cfg,
                )
                agent = serialization.from_state_dict(template, state)
            else:
                raise TypeError(
                    f"Loaded checkpoint type {type(state)} is not CALAgent or a container of it"
                )

    state_holder = {"agent": agent}
    policy_fn = _build_policy_fn(state_holder, deterministic=deterministic)

    meta = {
        "algo": "cal",
        "ckpt_resolved_path": str(resolved),
        "step": _extract_step(resolved),
        "deterministic": deterministic,
    }
    if config_path_used is not None:
        meta["config_path"] = str(config_path_used)
    return state_holder["agent"], policy_fn, meta
