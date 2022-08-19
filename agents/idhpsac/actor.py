import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Activation, LayerNormalization, Lambda
import gym


class Actor(keras.Model):
    """
    Actor/Policy network
    """

    def __init__(self, config, env: gym.Env, policy_sac: keras.Model):
        super().__init__()
        self.env = env

        # Dimensions
        self.input_dim = config["actor"]["s_dim"]
        self.output_dim = env.action_space.shape[0]

        # Network
        self.actor = None
        if config["identity_init"]:
            self.kernel_initializer = keras.initializers.Identity()
        else:
            self.kernel_initializer = keras.initializers.truncated_normal(stddev=config["std_init"])
        self.activation = tf.nn.relu

        # SAC policy
        self.policy_sac = policy_sac
        self.policy_sac.trainable = False

        # Create the actor
        self.setup()

    def setup(self):
        """
        Build the policy model
        """

        # Input
        inputs = keras.Input(shape=(self.input_dim,))
        output = Flatten()(inputs)

        # Hybrid policy with pre-trained SAC layers
        for layer in self.policy_sac.layers:
            if isinstance(layer, (Dense, LayerNormalization, Activation, Lambda)):
                output = layer(output)
                if isinstance(layer, Activation):
                    output = Dense(
                        layer.output_shape[1],
                        activation=self.activation,
                        use_bias=False,
                        kernel_initializer=self.kernel_initializer,
                    )(output)

        output = tf.tanh(output)

        # Policy model
        self.policy = keras.Model(inputs=inputs, outputs=output)

    def call(self, s):
        """
        Give the policy action
        """

        action = self.policy(s)

        return action
