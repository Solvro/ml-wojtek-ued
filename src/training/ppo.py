from typing import Any

import jax
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from env.functional import FunctionalEnv

# mujoco_playground's locomotion recipe
PPO: dict[str, Any] = {
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "clipping_epsilon": 0.2,
    "max_grad_norm": 1.0,
}

Normalizer = running_statistics.RunningStatisticsState


def make_network(env: FunctionalEnv) -> ppo_networks.PPONetworks:
    obs_size = env.observation_size
    return ppo_networks.make_ppo_networks(
        {"state": (obs_size.actor,), "privileged_state": (obs_size.critic,)},
        env.action_size,
        # brax types the hook narrower than its own normalizer
        preprocess_observations_fn=running_statistics.normalize,  # ty: ignore[invalid-argument-type]
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        activation=jax.nn.elu,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
