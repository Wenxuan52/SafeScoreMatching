import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class QuadrotorTracking2DEnv(gym.Env):
    """Planar quadrotor tracking a circular trajectory with altitude constraint.

    The implementation follows the task specification in the RCRL paper (Appendix D.1).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        g: float = 9.81,
        m: float = 1.0,
        I: float = 0.02,
        thrust_scale: float = None,
        torque_scale: float = 0.1,
        waypoint_center: Tuple[float, float] = (0.0, 1.0),
        waypoint_radius: float = 1.0,
        max_episode_steps: int = 360,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dt = dt
        self.g = g
        self.m = m
        self.I = I
        self.thrust_scale = thrust_scale if thrust_scale is not None else self.m * self.g
        self.torque_scale = torque_scale
        self.center = waypoint_center
        self.radius = waypoint_radius
        self.max_episode_steps = max_episode_steps

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.Q = np.diag([10.0, 1.0, 10.0, 1.0, 0.2, 0.2])
        self.R = np.diag([1e-4, 1e-4])
        # Hover reference action in normalized units.
        self.a_ref = np.array([0.5, 0.5], dtype=np.float32)

        self.state = np.zeros(6, dtype=np.float32)
        self._waypoints = self._create_waypoints()
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._waypoint_idx = 0

    def _create_waypoints(self) -> np.ndarray:
        angles = np.deg2rad(np.arange(0, 360))
        xs = self.center[0] + self.radius * np.cos(angles)
        zs = self.center[1] + self.radius * np.sin(angles)
        waypoints = np.stack([xs, np.zeros_like(xs), zs, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs)], axis=1)
        return waypoints.astype(np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        low = np.array([-1.5, -1.0, 0.25, -1.5, -0.2, -0.1], dtype=np.float32)
        high = np.array([1.5, 1.0, 1.75, 1.5, 0.2, 0.1], dtype=np.float32)
        self.state = self._rng.uniform(low, high).astype(np.float32)

        self._t = 0
        self._waypoint_idx = self._nearest_waypoint_idx(self.state[0], self.state[2])

        observation = self._get_observation()
        info: Dict[str, Any] = {"idx": int(self._waypoint_idx)}
        return observation, info

    def _nearest_waypoint_idx(self, x: float, z: float) -> int:
        dx = x - self.center[0]
        dz = z - self.center[1]
        dist_sq = dx * dx + dz * dz
        if dist_sq < 1e-6:
            return int(self._rng.integers(0, len(self._waypoints)))
        deltas = self._waypoints[:, [0, 2]] - np.array([x, z], dtype=np.float32)
        idx = int(np.argmin(np.sum(np.square(deltas), axis=1)))
        return idx

    def _get_observation(self) -> np.ndarray:
        ref = self._waypoints[self._waypoint_idx]
        obs = np.concatenate([self.state, ref]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        F1 = self.thrust_scale * action[0]
        F2 = self.thrust_scale * action[1]
        u1 = F1 + F2
        tau = self.torque_scale * (action[1] - action[0])

        x, xdot, z, zdot, theta, thetadot = self.state.astype(np.float64)

        xddot = -(u1 / self.m) * math.sin(theta)
        zddot = (u1 / self.m) * math.cos(theta) - self.g
        thetaddot = tau / self.I

        xdot += xddot * self.dt
        x += xdot * self.dt
        zdot += zddot * self.dt
        z += zdot * self.dt
        thetadot += thetaddot * self.dt
        theta += thetadot * self.dt

        self.state = np.array([x, xdot, z, zdot, theta, thetadot], dtype=np.float32)

        self._waypoint_idx = (self._waypoint_idx + 1) % len(self._waypoints)
        obs = self._get_observation()

        x_ref = self._waypoints[self._waypoint_idx]
        state_err = self.state - x_ref
        action_err = action - self.a_ref
        reward = -float(state_err @ self.Q @ state_err + action_err @ self.R @ action_err)

        h = max(0.5 - z, z - 1.5)
        cost = float(h > 0.0)

        terminated = bool(abs(x) > 2.0 or abs(z) > 3.0)
        self._t += 1
        truncated = bool(self._t >= self.max_episode_steps)

        info: Dict[str, Any] = {
            "h": float(h),
            "cost": float(cost),
            "z": float(z),
            "idx": int(self._waypoint_idx),
        }

        return obs, reward, terminated, truncated, info


def make_quadrotor_tracking_2d_env(**kwargs) -> QuadrotorTracking2DEnv:
    return QuadrotorTracking2DEnv(**kwargs)
