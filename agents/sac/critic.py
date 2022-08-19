import gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense, LayerNormalization, Activation


class Critic(keras.Model):
    """
    Critic Q-function
    """

    def __init__(self, config, env: gym.Env):
        super().__init__()

        # Dimensions
        self.input_dim = config["s_dim"] + env.action_space.shape[0]
        self.output_dim = 1

        # Network
        self.critic = None
        self.layers_size = config["critic"]["layers"]

        # Create the critic network
        self.setup()

    def setup(self):
        """
        Build the critic model
        """

        # Input layer
        input = keras.Input(shape=(self.input_dim,))
        output = Flatten()(input)

        # Hidden layers
        for i, layer_size in enumerate(self.layers_size):
            output = Dense(layer_size, activation=None, name=f"hidden_{i+1}")(output)
            output = LayerNormalization(center=True, scale=True, name=f"layer_norm_{i+1}")(output)
            output = Activation(tf.nn.relu)(output)

        # Output layer
        output = Dense(self.output_dim, activation=None)(output)

        # Critic model
        self.critic = keras.Model(input, output)

    def call(self, s, action):
        """
        Give the Q-value for a given observation and action
        """

        input = tf.concat([s, action], axis=1)

        return tf.squeeze(self.critic(input), 1)  # remove batch dimension

    def soft_update(self, source_weights, tau):
        """
        Update parameters from a source network
        """

        source = source_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)
