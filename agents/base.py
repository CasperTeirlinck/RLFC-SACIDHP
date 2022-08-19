import gym
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from tasks.base import TrackingTask
from tools.numpy import array32


class Agent(ABC):
    """
    Base class for RL agents with tracking task
    """

    def __init__(self, config, task: TrackingTask, env: gym.Env):
        self.config = config

        # Environment
        self.env = env

        # Tracking task
        self.task = task
        self.tracking_descr_external = None
        self.tracking_descr = None
        self.tracking_ref = None
        self.tracking_P = None
        self.tracking_P_external = None
        self.tracking_Q = None
        self.tracking_max = None
        if task is not None:
            self.set_task()

        # RL state
        self.s_states = config["s_states"] if "s_states" in config.keys() else []
        self.s_dim = config["s_dim"] if "s_dim" in config.keys() else 0

        # RL state actor
        self.s_a_states = config["actor"]["s_states"] if "s_states" in config["actor"].keys() else self.s_states
        self.s_a_dim = config["actor"]["s_dim"] if "s_dim" in config["actor"].keys() else self.s_dim

        # RL state critic
        self.s_c_states = config["critic"]["s_states"] if "s_states" in config["critic"].keys() else self.s_states
        self.s_c_dim = config["critic"]["s_dim"] if "s_dim" in config["critic"].keys() else self.s_dim

        # Training
        self.t = 0
        self.gamma = config["gamma"]
        self.tau = config["tau"]

    @abstractmethod
    @tf.function(jit_compile=True)
    def get_action(self, s, **kwargs):
        """
        Give the policy's action
        """

        raise NotImplementedError

    @abstractmethod
    def get_s_a(self, obs, tracking_ref, action_prev, t_f=False):
        """
        Get input state vector for actor
        """

        raise NotImplementedError

    @abstractmethod
    def get_s_c(self, obs, tracking_ref, action_prev, t_f=False):
        """
        Get input state vector for critic
        """

        raise NotImplementedError

    @abstractmethod
    def get_reward(self, tracking_err):
        """
        Calculate the reward
        """

        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs):
        """
        Training loop
        """

        raise NotImplementedError

    @abstractmethod
    @tf.function(experimental_relax_shapes=True)
    def update(self, *args):
        """
        Update the actor and critic
        """

        raise NotImplementedError

    def set_task(self):
        """
        Set tracking task related properties
        """

        # Externally tracked states
        self.tracking_descr_external = np.array(self.env.obs_descr)[
            np.in1d(self.env.obs_descr, [*self.task.tracking_ref.keys()])
        ]
        self.tracking_ref = array32([self.task.tracking_ref[_] for _ in self.tracking_descr_external])

        # Internally tracked states
        self.tracking_descr = np.array(self.env.obs_descr)[
            np.in1d(self.env.obs_descr, [*self.task.tracking_scale.keys()])
        ]

        # Selection and scaling matrices
        self.tracking_P = array32(
            [[1 if obs == ref else 0 for obs in self.env.obs_descr] for ref in self.tracking_descr]
        )
        self.tracking_P_external = array32(
            [[1 if obs == ref else 0 for obs in self.env.obs_descr] for ref in self.tracking_descr_external]
        )
        self.tracking_Q = np.diag(array32([self.task.tracking_scale[_] for _ in self.tracking_descr]))

        #
        self.tracking_max = array32([self.task.tracking_max[_] for _ in self.tracking_descr])
        # self.tracking_ref_max = np.max(abs(self.tracking_ref), axis=-1)
        # self.tracking_ref_max = np.where(self.tracking_ref_max == 0.0, 1.0, self.tracking_ref_max)
        # self.tracking_ref_max = np.where(self.tracking_ref_max == 0.0, d2r(5.0), self.tracking_ref_max)

        # Tracking RMSE thresholds for adaptive learning rate
        self.tracking_thresh = array32([self.task.tracking_thresh[_] for _ in self.tracking_descr])

    def reset_task(self):
        """
        Reset external tracking signals
        """

        self.task.set_tracking_ref()
        self.task.set_tracking_max()
        self.tracking_ref = array32([self.task.tracking_ref[_] for _ in self.tracking_descr_external])
        self.tracking_max = array32([self.task.tracking_max[_] for _ in self.tracking_descr])
