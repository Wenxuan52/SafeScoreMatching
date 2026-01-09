from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register


_CUSTOM_ENV_ID = "QuadrotorTracking2D-v0"


def ensure_custom_envs_registered() -> None:
    """Register custom environments if they are not yet in Gymnasium's registry."""

    try:
        gym.spec(_CUSTOM_ENV_ID)
    except gym.error.Error:
        register(
            id=_CUSTOM_ENV_ID,
            entry_point="jaxrl5.envs.quadrotor_tracking_2d:make_quadrotor_tracking_2d_env",
        )
