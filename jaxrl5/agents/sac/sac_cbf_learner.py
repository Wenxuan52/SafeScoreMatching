"""Soft Actor-Critic with a CBF action correction layer (SAC-CBF)."""

from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax.serialization as serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.envs.dynamics.quad2d import step_dynamics
from jaxrl5.networks import Ensemble, MLP, StateActionValue, subsample_ensemble


class SACCbfLearner(Agent):
    """SAC learner that applies a CBF-based action correction at execution time."""

    critic: TrainState
    target_critic: TrainState
    temp: TrainState

    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)

    cbf_enabled: bool = struct.field(pytree_node=False)
    cbf_mu: float
    cbf_dt: float
    cbf_fd_eps: float
    cbf_max_iters: int = struct.field(pytree_node=False)
    cbf_grad_eps: float
    cbf_shrink_factor: float
    z_min: float
    z_max: float
    z_index: int = struct.field(pytree_node=False)

    dyn_m: float
    dyn_I: float
    dyn_g: float
    thrust_scale: float
    torque_scale: float
    action_low: float
    action_high: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        cbf_enabled: bool = True,
        cbf_mu: float = 0.2,
        cbf_dt: float = 1.0 / 60.0,
        cbf_fd_eps: float = 1e-3,
        cbf_max_iters: int = 8,
        cbf_grad_eps: float = 1e-8,
        cbf_shrink_factor: float = 0.5,
        z_min: float = 0.5,
        z_max: float = 1.5,
        z_index: int = 2,
        dyn_m: float = 1.0,
        dyn_I: float = 0.02,
        dyn_g: float = 9.81,
        thrust_scale: Optional[float] = None,
        torque_scale: float = 0.1,
        action_low: float = 0.0,
        action_high: float = 1.0,
    ) -> "SACCbfLearner":
        """Construct a SAC-CBF learner."""

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        if thrust_scale is None:
            thrust_scale = dyn_m * dyn_g

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            cbf_enabled=cbf_enabled,
            cbf_mu=cbf_mu,
            cbf_dt=cbf_dt,
            cbf_fd_eps=cbf_fd_eps,
            cbf_max_iters=cbf_max_iters,
            cbf_grad_eps=cbf_grad_eps,
            cbf_shrink_factor=cbf_shrink_factor,
            z_min=z_min,
            z_max=z_max,
            z_index=z_index,
            dyn_m=dyn_m,
            dyn_I=dyn_I,
            dyn_g=dyn_g,
            thrust_scale=thrust_scale,
            torque_scale=torque_scale,
            action_low=action_low,
            action_high=action_high,
        )

    def _cbf_params(self) -> Dict[str, float]:
        return {
            "m": float(self.dyn_m),
            "I": float(self.dyn_I),
            "g": float(self.dyn_g),
            "thrust_scale": float(self.thrust_scale),
            "torque_scale": float(self.torque_scale),
            "action_low": float(self.action_low),
            "action_high": float(self.action_high),
        }

    def _action_to_env(self, action_ext: np.ndarray) -> np.ndarray:
        a = np.asarray(action_ext, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        a01 = (a + 1.0) * 0.5
        return self.action_low + a01 * (self.action_high - self.action_low)

    def _h_value(self, z: float) -> float:
        return float(max(self.z_min - z, z - self.z_max, 0.0))

    def _predict_next_state_from_obs_action(
        self, obs: np.ndarray, action_ext: np.ndarray
    ) -> np.ndarray:
        state = np.asarray(obs[:6], dtype=np.float32)
        action_env = self._action_to_env(action_ext)
        return step_dynamics(state, action_env, float(self.cbf_dt), self._cbf_params())

    def _cbf_residual(self, obs: np.ndarray, action_ext: np.ndarray) -> float:
        if not self.cbf_enabled:
            return 0.0
        state = np.asarray(obs[:6], dtype=np.float32)
        z = float(state[self.z_index])
        h_curr = self._h_value(z)
        next_state = self._predict_next_state_from_obs_action(obs, action_ext)
        z_next = float(next_state[self.z_index])
        h_next = self._h_value(z_next)
        decay = max(0.0, 1.0 - float(self.cbf_mu) * float(self.cbf_dt))
        return float(h_next - decay * h_curr)

    def _cbf_correct_action(
        self, obs: np.ndarray, action_nom: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        a = np.asarray(action_nom, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        residual_nom = self._cbf_residual(obs, a)
        if not self.cbf_enabled or residual_nom <= 0.0:
            return a, {
                "residual_nom": residual_nom,
                "residual_safe": residual_nom,
                "success": float(residual_nom <= 0.0),
                "delta": 0.0,
                "iters": 0.0,
            }

        r = residual_nom
        iters = 0
        for it in range(int(self.cbf_max_iters)):
            iters = it + 1
            g = np.zeros_like(a)
            for i in range(a.shape[0]):
                a_eps = a.copy()
                a_eps[i] += float(self.cbf_fd_eps)
                a_eps = np.clip(a_eps, -1.0, 1.0)
                r_eps = self._cbf_residual(obs, a_eps)
                g[i] = (r_eps - r) / float(self.cbf_fd_eps)

            grad_norm = float(np.linalg.norm(g))
            if grad_norm < float(self.cbf_grad_eps):
                a = np.clip(a * float(self.cbf_shrink_factor), -1.0, 1.0)
            else:
                step = (r / (grad_norm ** 2 + float(self.cbf_grad_eps))) * g
                a = np.clip(a - step, -1.0, 1.0)

            r = self._cbf_residual(obs, a)
            if r <= 0.0:
                break

        delta = float(np.linalg.norm(a - action_nom))
        return a, {
            "residual_nom": residual_nom,
            "residual_safe": r,
            "success": float(r <= 0.0),
            "delta": delta,
            "iters": float(iters),
        }

    def _nominal_actions(self, observations: jnp.ndarray, rng: jax.random.PRNGKey):
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def _eval_nominal_actions(self, observations: jnp.ndarray):
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return dist.mode()

    def _format_obs(self, observations: jnp.ndarray):
        if observations.ndim == 1:
            return observations[None], True
        return observations, False

    def _apply_cbf_batch(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if not self.cbf_enabled:
            return actions, {
                "cbf/residual_nom_mean": 0.0,
                "cbf/residual_safe_mean": 0.0,
                "cbf/success_rate": 1.0,
                "cbf/delta_action_mean": 0.0,
            }

        residual_nom = []
        residual_safe = []
        deltas = []
        successes = []
        safe_actions = []

        for obs, act in zip(observations, actions):
            a_safe, info = self._cbf_correct_action(obs, act)
            safe_actions.append(a_safe)
            residual_nom.append(info["residual_nom"])
            residual_safe.append(info["residual_safe"])
            successes.append(info["success"])
            deltas.append(info["delta"])

        metrics = {
            "cbf/residual_nom_mean": float(np.mean(residual_nom)),
            "cbf/residual_safe_mean": float(np.mean(residual_safe)),
            "cbf/success_rate": float(np.mean(successes)),
            "cbf/delta_action_mean": float(np.mean(deltas)),
        }
        return np.asarray(safe_actions, dtype=np.float32), metrics

    def sample_actions(self, observations: jnp.ndarray):
        obs, single = self._format_obs(observations)
        key, rng = jax.random.split(self.rng)
        actions, _ = self._nominal_actions(obs, key)
        actions_np = np.asarray(actions, dtype=np.float32)
        safe_actions, _ = self._apply_cbf_batch(np.asarray(obs), actions_np)
        if single:
            safe_actions = safe_actions[0]
        return safe_actions, self.replace(rng=rng)

    def eval_actions(self, observations: jnp.ndarray):
        obs, single = self._format_obs(observations)
        actions = self._eval_nominal_actions(obs)
        actions_np = np.asarray(actions, dtype=np.float32)
        safe_actions, _ = self._apply_cbf_batch(np.asarray(obs), actions_np)
        if single:
            safe_actions = safe_actions[0]
        return safe_actions, self

    def update_actor(self, batch: DatasetDict) -> Tuple["SACCbfLearner", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )
            q = qs.mean(axis=0)
            alpha = self.temp.apply_fn({"params": self.temp.params})
            actor_loss = (alpha * log_probs - q).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean(),
                "logp_mean": log_probs.mean(),
                "alpha": alpha,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: jnp.ndarray) -> Tuple["SACCbfLearner", Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "alpha": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict) -> Tuple["SACCbfLearner", Dict[str, float]]:
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["next_observations"])

        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["not_terminated"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["not_terminated"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1_mean": qs[0].mean(),
                "q2_mean": qs[1].mean() if qs.shape[0] > 1 else qs[0].mean(),
                "target_q_mean": target_q.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update(self, batch: DatasetDict) -> Tuple["SACCbfLearner", Dict[str, float]]:
        new_agent, critic_info = self.update_critic(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        obs_np = np.asarray(batch["observations"], dtype=np.float32)
        nominal_actions = np.asarray(
            new_agent._eval_nominal_actions(batch["observations"]), dtype=np.float32
        )
        _, cbf_info = new_agent._apply_cbf_batch(obs_np, nominal_actions)

        metrics = {**critic_info, **actor_info, **temp_info, **cbf_info}
        return new_agent, metrics

    def save(self, path: str) -> None:
        """Serialize the learner to a file using Flax msgpack."""
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self))

    @classmethod
    def load(cls, path: str) -> "SACCbfLearner":
        """Load a learner checkpoint from disk."""
        with open(path, "rb") as f:
            data = f.read()
        restored = None
        if hasattr(serialization, "msgpack_restore"):
            restored = serialization.msgpack_restore(data)
            if isinstance(restored, cls):
                return restored
            if isinstance(restored, dict):
                for value in restored.values():
                    if isinstance(value, cls):
                        return value

        try:
            return serialization.from_bytes(cls, data)
        except Exception as exc:
            restored_type = type(restored) if restored is not None else None
            raise TypeError(
                f"Loaded checkpoint type {restored_type} is not {cls.__name__} or a container of it"
            ) from exc
