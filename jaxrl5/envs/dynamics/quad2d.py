"""Planar 2D quadrotor dynamics utilities."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def step_dynamics(state: np.ndarray, action: np.ndarray, dt: float, params: Dict) -> np.ndarray:
    """Propagate planar quadrotor dynamics using semi-implicit Euler.

    Args:
        state: Array of shape (6,) = [x, xdot, z, zdot, theta, thetadot].
        action: Array of shape (2,) = [T1, T2] in normalized units.
        dt: Integration timestep.
        params: Dictionary containing dynamics parameters. Expected keys:
            - m: Mass.
            - I: Moment of inertia.
            - g: Gravity constant.
            - thrust_scale: Scale mapping normalized thrust to force.
            - torque_scale: Scale for torque from thrust difference.
            - action_low: Lower bound for normalized thrust (default 0.0).
            - action_high: Upper bound for normalized thrust (default 1.0).

    Returns:
        Next state as a new numpy array with shape (6,).
    """

    action_low = params.get("action_low", 0.0)
    action_high = params.get("action_high", 1.0)
    clipped_action = np.clip(np.asarray(action, dtype=np.float32), action_low, action_high)

    x, xdot, z, zdot, theta, thetadot = np.asarray(state, dtype=np.float64).copy()

    m = params.get("m", 1.0)
    I = params.get("I", 0.02)
    g = params.get("g", 9.81)
    thrust_scale = params["thrust_scale"]
    torque_scale = params["torque_scale"]

    F1 = thrust_scale * clipped_action[0]
    F2 = thrust_scale * clipped_action[1]
    u1 = F1 + F2
    tau = torque_scale * (clipped_action[1] - clipped_action[0])

    xddot = -(u1 / m) * math.sin(theta)
    zddot = (u1 / m) * math.cos(theta) - g
    thetaddot = tau / I

    # Semi-implicit (symplectic) Euler integration: update velocities first, then positions.
    xdot += xddot * dt
    x += xdot * dt
    zdot += zddot * dt
    z += zdot * dt
    thetadot += thetaddot * dt
    theta += thetadot * dt

    next_state = np.array([x, xdot, z, zdot, theta, thetadot], dtype=np.float32)
    return next_state
