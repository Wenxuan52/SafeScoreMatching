from typing import Callable, Dict, Optional, Tuple

import numpy as np

from jaxrl5.agents.safe_matching.safe_matching_learner import SafeScoreMatchingLearner
from jaxrl5.tools.checkpoints import resolve_checkpoint


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _extract_step(path: str) -> Optional[int]:
    import os
    import re

    m = re.search(r"ckpt_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_ssm(
    ckpt_path: str,
    step: Optional[int] = None,
    *,
    deterministic: bool = True,
    ddpm_temperature: Optional[float] = None,
) -> Tuple[SafeScoreMatchingLearner, PolicyFn, Dict]:
    resolved = resolve_checkpoint(ckpt_path, step)
    agent = SafeScoreMatchingLearner.load(resolved)

    if ddpm_temperature is not None and hasattr(agent, "ddpm_temperature"):
        agent = agent.replace(ddpm_temperature=ddpm_temperature)

    state = {"agent": agent}

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)
        if deterministic:
            action, new_agent = state["agent"].eval_actions(obs_np)
        else:
            action, new_agent = state["agent"].sample_actions(obs_np)
        state["agent"] = new_agent
        return np.asarray(action, dtype=np.float32)

    meta = {
        "algo": "ssm",
        "ckpt_resolved_path": resolved,
        "step": _extract_step(resolved),
        "deterministic": deterministic,
    }
    if ddpm_temperature is not None:
        meta["ddpm_temperature"] = ddpm_temperature
    return state["agent"], policy_fn, meta
