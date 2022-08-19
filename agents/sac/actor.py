import gym
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Flatten, Dense, LayerNormalization, Activation


class Actor(keras.Model):
    """
    Actor/Policy model (stochastic policy)
    """

    def __init__(self, config, env: gym.Env):
        super().__init__()

        # Dimensions
        self.input_dim = config["s_dim"]
        self.output_dim = env.action_space.shape[0]

        # Network
        self.policy = None
        self.layers_size = config["actor"]["layers"]

        # Create the actor network
        self.setup()

    def setup(self):
        """
        Build the stochastic policy model with mean and std
        """

        # Input layer
        input = keras.Input(shape=(self.input_dim,))
        output = Flatten()(input)

        # Hidden layers
        for i, layer_size in enumerate(self.layers_size):
            output = Dense(layer_size, activation=None, name=f"hidden_{i+1}")(output)
            output = LayerNormalization(center=True, scale=True, epsilon=1e-12, name=f"layer_norm_{i+1}")(output)
            output = Activation(tf.nn.relu, name=f"activation_{i+1}")(output)

        # Output layer: policy mean
        mean = Dense(self.output_dim, activation=None, name="output_mean")(output)

        # Output layer: policy (log) std
        log_std = Dense(self.output_dim, activation=None, name="output_std")(output)
        std = tf.exp(log_std)

        # Policy model
        self.policy = keras.Model(input, [mean, std])

    def call(self, s):
        """
        Give the policy mean and std for a given observation
        """

        mean, std = self.policy(s)
        std = tf.clip_by_value(std, 1e-6, 1)  # prevent std=0

        return mean, std

    @tf.function(jit_compile=True)
    def sample(self, s, deterministic=False, return_mean=False):
        """
        Sample stochastic policy and compute log probability
        """

        # Get mean and std
        mean, std = self(s)
        if deterministic:
            # Squash action to [-1, 1]
            action = tf.tanh(mean)
            return action, None

        # Sample action
        policy = tfp.distributions.Normal(mean, std)  # tfp.Normal implements reparam. trick
        action = policy.sample()

        # Log probability
        log_prob = policy.log_prob(action)

        # Squash action to [-1, 1]
        action = tf.tanh(action)

        # Squash correction log_prob
        log_prob -= tf.math.log(1 - action**2 + 1e-6)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        # TODO: try squash correction formula from CAPS_code sac_core.py line 55

        if return_mean:
            return action, tf.tanh(mean), log_prob
        else:
            return action, log_prob
