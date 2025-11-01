from __future__ import annotations

import safety_gymnasium


def make_safety_env(
    env_name: str,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs,
):
    """Factory returning a Gymnasium-compatible Safety-Gymnasium environment."""

    env = safety_gymnasium.make(env_name, render_mode=render_mode, **kwargs)

    # Optionally perform an initial reset to apply the seed immediately.
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    return env
