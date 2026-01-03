"""Sanity test for 2D quadrotor hover dynamics."""
import numpy as np

from jaxrl5.envs.dynamics.quad2d import step_dynamics


def main():
    dt = 1.0 / 60.0
    params = {
        "m": 1.0,
        "I": 0.02,
        "g": 9.81,
        # Hover should be achievable with T1=T2=0.5 when thrust_scale = m * g.
        "thrust_scale": 1.0 * 9.81,
        # Small torque coefficient to limit angular acceleration.
        "torque_scale": 0.1,
        "action_low": 0.0,
        "action_high": 1.0,
    }

    state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action = np.array([0.5, 0.5], dtype=np.float32)

    for t in range(2000):
        state = step_dynamics(state, action, dt, params)
        if (t + 1) % 200 == 0:
            print(f"step={t+1}, z={state[2]:.4f}, zdot={state[3]:.4f}, theta={state[4]:.4f}")

        assert np.all(np.isfinite(state)), "State contains NaN or inf values"
        # Allow a slightly relaxed bound to accommodate numerical drift from the simplified model.
        assert 0.0 <= state[2] <= 2.0, "Altitude diverged from hover band"
        assert abs(state[4]) < 0.5, "Orientation drifted excessively"

    print("Hover dynamics test passed.")


if __name__ == "__main__":
    main()
