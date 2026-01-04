"""Implementations of algorithms for continuous control."""

import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import checkpoints
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhDeterministic
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(
    rng, apply_fn, params, observations: np.ndarray, action_noise: float
) -> np.ndarray:
    key, rng = jax.random.split(rng)
    actions = apply_fn({"params": params}, observations)
    noise = jax.random.normal(key, shape=actions.shape) * action_noise
    actions = actions + noise
    return jnp.clip(actions, -1.0, 1.0), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    return apply_fn({"params": params}, observations)


class TD3Learner(Agent):
    critic: TrainState
    target_critic: TrainState
    target_actor: TrainState
    tau: float
    discount: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    actor_delay: int = 2

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        actor_delay: int = 2,
    ):
        """
        An implementation of TD3: https://arxiv.org/abs/1802.09477
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhDeterministic(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )
        target_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
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

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            target_actor=target_actor,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            exploration_noise=exploration_noise,
            target_policy_noise=target_policy_noise,
            target_policy_noise_clip=target_policy_noise_clip,
            actor_delay=actor_delay,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng, num=2)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            actions = self.actor.apply_fn(
                {"params": actor_params}, batch["observations"]
            )
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = -q.mean()
            return actor_loss, {
                "actor_loss": actor_loss,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        target_actor_params = optax.incremental_update(
            actor.params, self.target_actor.params, self.tau
        )

        target_actor = self.target_actor.replace(params=target_actor_params)

        return self.replace(actor=actor, target_actor=target_actor, rng=rng), actor_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        next_actions = self.target_actor.apply_fn(
            {"params": self.target_actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        target_noise = (
            jax.random.normal(key, next_actions.shape) * self.target_policy_noise
        )
        target_noise = target_noise.clip(
            -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        next_actions += target_noise

        # Used only for REDQ.
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
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["not_terminated"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        true_steps = new_agent.critic.step / utd_ratio

        # Actor delay
        new_agent, actor_info = jax.lax.cond(
            true_steps % new_agent.actor_delay == 0,
            new_agent.update_actor,
            lambda _: (new_agent, {"actor_loss": 0.0}),
            mini_batch,
        )

        return new_agent, {**actor_info, **critic_info}

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            self.exploration_noise,
        )
        return np.asarray(actions), self.replace(rng=new_rng)

    def save(self, ckpt_dir: str, step: int) -> None:
        """Save learner state to ``ckpt_dir`` with step-specific naming.

        Uses ``flax.training.checkpoints`` to serialize a plain pytree dict
        containing all TrainStates, RNG, and hyperparameters needed to
        rehydrate the agent for evaluation.
        """

        state = {
            "actor": self.actor,
            "critic": self.critic,
            "target_actor": self.target_actor,
            "target_critic": self.target_critic,
            "rng": self.rng,
            "metadata": {
                "tau": float(self.tau),
                "discount": float(self.discount),
                "num_qs": int(self.num_qs),
                "num_min_qs": None if self.num_min_qs is None else int(self.num_min_qs),
                "exploration_noise": float(self.exploration_noise),
                "target_policy_noise": float(self.target_policy_noise),
                "target_policy_noise_clip": float(self.target_policy_noise_clip),
                "actor_delay": int(self.actor_delay),
            },
        }

        checkpoints.save_checkpoint(
            ckpt_dir,
            target=state,
            step=step,
            overwrite=True,
            keep=3,
            prefix="ckpt_",
        )

    @classmethod
    def load(cls, ckpt_dir: str, step: Optional[int] = None) -> "TD3Learner":
        """Load learner state from ``ckpt_dir`` (or checkpoint file).

        Args:
            ckpt_dir: Directory containing checkpoints or the checkpoint file.
            step: Optional explicit step; passed to ``restore_checkpoint``.
        """

        # Support both directory paths (with prefix) and direct checkpoint files.
        ckpt_kwargs = {"target": None, "step": step}
        if os.path.isfile(ckpt_dir):
            ckpt = checkpoints.restore_checkpoint(ckpt_dir, **ckpt_kwargs)
        else:
            ckpt = checkpoints.restore_checkpoint(
                ckpt_dir, prefix="ckpt_", **ckpt_kwargs
            )

        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir} (step={step})")

        meta = ckpt.get("metadata", {})
        return cls(
            rng=ckpt["rng"],
            actor=ckpt["actor"],
            critic=ckpt["critic"],
            target_actor=ckpt["target_actor"],
            target_critic=ckpt["target_critic"],
            tau=float(meta.get("tau", 0.005)),
            discount=float(meta.get("discount", 0.99)),
            num_qs=int(meta.get("num_qs", 2)),
            num_min_qs=meta.get("num_min_qs"),
            exploration_noise=float(meta.get("exploration_noise", 0.1)),
            target_policy_noise=float(meta.get("target_policy_noise", 0.2)),
            target_policy_noise_clip=float(meta.get("target_policy_noise_clip", 0.5)),
            actor_delay=int(meta.get("actor_delay", 2)),
        )
