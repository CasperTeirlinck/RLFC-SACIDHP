import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import gym


class Critic(keras.Model):
    """
    Critic state-value network
    """

    def __init__(self, config, env: gym.Env):
        super().__init__()

        # Dimensions
        self.input_dim = config["critic"]["s_dim"]
        self.output_dim = env.observation_space.shape[0]

        # Network
        self.critic = None
        self.layers_size = config["critic"]["layers"]
        self.kernel_initializer = keras.initializers.truncated_normal(stddev=config["std_init"])
        if config["activation"] == "tanh":
            self.activation = tf.nn.tanh
        elif config["activation"] == "relu":
            self.activation = tf.nn.relu

        # Create the critic network
        self.setup()

    def setup(self):
        """
        Build the critic model
        """

        # Input
        input = keras.Input(shape=(self.input_dim,))
        output = Flatten()(input)

        # Hidden layers
        for layer_size in self.layers_size:
            output = Dense(
                layer_size,
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(output)

        # Output layer
        output = Dense(
            self.output_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

        # Critic model
        self.critic = keras.Model(inputs=input, outputs=output)

    def call(self, s):
        """
        Give the state-value
        """

        return self.critic(s)

    def soft_update(self, source_weights, tau):
        """
        Update parameters from a source network
        """

        source = source_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)
