import jax
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import Activation, NetworkConfig
from env.functional import FunctionalEnv

Normalizer = running_statistics.RunningStatisticsState

ACTIVATIONS = {
    Activation.elu: jax.nn.elu,
    Activation.relu: jax.nn.relu,
    Activation.tanh: jax.nn.tanh,
    Activation.swish: jax.nn.swish,
}


def make_network(
    env: FunctionalEnv, network: NetworkConfig
) -> ppo_networks.PPONetworks:
    obs_size = env.observation_size
    return ppo_networks.make_ppo_networks(
        {"state": (obs_size.actor,), "privileged_state": (obs_size.critic,)},
        env.action_size,
        # brax types the hook narrower than its own normalizer
        preprocess_observations_fn=running_statistics.normalize,  # ty: ignore[invalid-argument-type]
        policy_hidden_layer_sizes=tuple(network.policy_hidden),
        value_hidden_layer_sizes=tuple(network.value_hidden),
        activation=ACTIVATIONS[network.activation],
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
