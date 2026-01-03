from __future__ import annotations

import numpy as np

from jaxrl5.envs import make_env


if __name__ == "__main__":
    env = make_env("QuadrotorTracking2D-v0", seed=0)
    obs, info = env.reset()
    print("obs shape/dtype:", obs.shape, obs.dtype)

    assert obs.shape == (12,)

    for i in range(5):
        step_out = env.step(env.action_space.sample())
        print(
            len(step_out),
            "tuple",
            "reward=",
            step_out[1],
            "cost=",
            step_out[2],
            "terminated=",
            step_out[3],
            "truncated=",
            step_out[4],
            "keys=",
            list(step_out[5].keys())[:10],
        )
        if not np.isfinite(step_out[2]):
            raise RuntimeError("Cost should be finite")

    assert len(step_out) == 6
    cost_val = float(step_out[2])
    assert cost_val in (0.0, 1.0), "Cost should be binary"
    for required_key in ["h", "cost", "z", "idx"]:
        assert required_key in step_out[5], f"Missing key {required_key} in info"

    print("debug_make_quadrotor_tracking2d: completed checks successfully")
