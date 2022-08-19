import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import gym


class Actor(keras.Model):
    """
    Actor/Policy network
    """

    def __init__(self, config, env: gym.Env):
        super().__init__()
        self.env = env

        # Dimensions
        self.input_dim = config["actor"]["s_dim"]
        self.output_dim = env.action_space.shape[0]

        # Network
        self.actor = None
        self.layers_size = config["actor"]["layers"]
        if config["identity_init"]:
            self.kernel_initializer = keras.initializers.Identity()
        else:
            self.kernel_initializer = keras.initializers.truncated_normal(stddev=config["std_init"])
        if config["activation"] == "tanh":
            self.activation = tf.nn.tanh
        elif config["activation"] == "relu":
            self.activation = tf.nn.relu

        # Create the actor
        self.setup()

    def setup(self):
        """
        Build the policy model
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
            activation=tf.nn.tanh,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

        # Policy model
        self.policy = keras.Model(inputs=input, outputs=output)

    def call(self, s):
        """
        Give the policy action
        """

        action = self.policy(s)

        return action
