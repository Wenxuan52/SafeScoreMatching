import inspect
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from flax import serialization

from jaxrl5.agents.safe_matching.safe_matching_learner import SafeScoreMatchingLearner
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
    sig = inspect.signature(SafeScoreMatchingLearner.create)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in cfg.items() if k in allowed}


def load_ssm(
    ckpt_path: str,
    step: Optional[int] = None,
    *,
    observation_space=None,
    action_space=None,
    seed: int = 0,
    config_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    deterministic: bool = True,
    ddpm_temperature: Optional[float] = None,
    **kwargs,
) -> Tuple[SafeScoreMatchingLearner, PolicyFn, Dict]:
    if observation_space is None or action_space is None:
        raise ValueError("load_ssm requires observation_space and action_space to build a template agent.")

    resolved = resolve_checkpoint(ckpt_path, step)
    try:
        direct_agent = SafeScoreMatchingLearner.load(resolved)
    except Exception:
        direct_agent = None

    config_path_used: Optional[Path] = None
    if isinstance(direct_agent, SafeScoreMatchingLearner):
        agent = direct_agent
    else:
        data = Path(resolved).read_bytes()
        state = serialization.msgpack_restore(data)
        if not isinstance(state, dict):
            raise TypeError(
                f"Restored checkpoint is not a dict or SafeScoreMatchingLearner (got {type(state)})."
            )

        resolved_run_dir = Path(run_dir) if run_dir is not None else Path(resolved).parent.parent
        config_path_used = Path(config_path) if config_path is not None else _find_config_path(resolved_run_dir)
        cfg_dict = _load_config(config_path_used)
        filtered_cfg = _filter_create_kwargs(cfg_dict)
        try:
            template = SafeScoreMatchingLearner.create(
                seed=seed,
                observation_space=observation_space,
                action_space=action_space,
                **filtered_cfg,
            )
        except TypeError as exc:
            raise TypeError(
                f"Failed to create template SafeScoreMatchingLearner with config at {config_path_used}: {exc}"
            ) from exc

        agent = serialization.from_state_dict(template, state)

    if ddpm_temperature is not None and hasattr(agent, "ddpm_temperature"):
        agent = agent.replace(ddpm_temperature=ddpm_temperature)

    state_holder = {"agent": agent}

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        if deterministic:
            action, new_agent = state_holder["agent"].eval_actions(obs_np)
        else:
            action, new_agent = state_holder["agent"].sample_actions(obs_np)
        state_holder["agent"] = new_agent
        return np.asarray(action, dtype=np.float32)

    meta = {
        "algo": "ssm",
        "ckpt_resolved_path": str(resolved),
        "step": _extract_step(resolved),
        "deterministic": deterministic,
    }
    if ddpm_temperature is not None:
        meta["ddpm_temperature"] = ddpm_temperature
    if config_path_used is not None:
        meta["config_path"] = str(config_path_used)
    return state_holder["agent"], policy_fn, meta
