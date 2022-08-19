import importlib
import numpy as np
from pyperclip import determine_clipboard
import tensorflow as tf
import os
import glob
import gym
from gym.spaces import Box
import matplotlib.pyplot as plt
from agents.sac.sac import SAC
from envs.citation import Citation
from tasks.tracking_altitude import TrackAltitude

from tools import set_random_seed, clip
from tools.utils import d2r, incr_action, scale_action


class CitationAttitude(gym.Env):
    """
    Cessna Citation
    Includes pre-trained attitude controller
    """

    def __init__(self, config, task: TrackAltitude, inner_save_dir, dt):
        super().__init__()
        self.seed(config["seed"])

        # Env
        self.env_inner = Citation(config, dt=dt, obs_extra=[9])

        # Labels
        self.obs_descr = np.hstack([self.env_inner.obs_descr, "h"])
        self.action_descr = [
            "theta_ref",  # [rad]
        ]

        # Inner attitude controller
        self.agent_inner = SAC.load(inner_save_dir, task, self.env_inner)

        # Action space
        self.action_space = Box(
            low=d2r(np.array([-30])),
            high=d2r(np.array([30])),
            dtype=np.float64,
        )
        self.action_space_rates = Box(
            low=d2r(np.array([-10])),
            high=d2r(np.array([10])),
            dtype=np.float64,
        )

        # Observation space
        self.observation_space = Box(
            low=np.array([-np.inf]),
            high=np.array([np.inf]),
            dtype=np.float64,
        )

        # Task
        self.task = task

        # Running
        self.t = 0
        self.obs = None
        self.action = None
        self.action_inner = None

        # Logging
        self.action_history = []

    def __str__(self):
        return "citation_attitude"

    def reset(self):
        """
        Reset the environment to initial state
        """

        self.obs, obs_extra = self.agent_inner.env.reset()
        self.agent_inner.reset_task()
        self.action = 0
        self.action_inner = self.agent_inner.env.action
        self.t = 0

        self.action_history = []

        return np.hstack([self.obs, obs_extra])

    def step(self, action):
        """
        Excecute a single time step
        """

        # Action is internal reference signal
        theta_ref = action

        # Sample external tracking reference signal
        tracking_ref_inner = self.agent_inner.tracking_ref[:, self.t]

        # Construct complete inner loop reference signal
        tracking_ref_inner = np.hstack([theta_ref, tracking_ref_inner])

        # Get attitude controller action
        s = self.agent_inner.get_s(self.obs, tracking_ref_inner)
        action_pi = self.agent_inner.get_action(s[np.newaxis], deterministic=True)
        self.action_inner = scale_action(action_pi, self.env_inner.action_space)

        # Take action
        self.obs, obs_extra = self.agent_inner.env.step(self.action_inner)
        self.t += 1

        # Logging
        self.action_history.append(action)

        return np.hstack([self.obs, obs_extra])

    def render(self, task, idx_end=None, show_rmse=True, lr_warmup=None, state_simulink=None, rewards=None):
        """
        Visualize environment response
        """

        tracking_ref_internal = {"theta": self.action_history}
        self.env_inner.render(
            task,
            tracking_ref_internal=tracking_ref_internal,
            idx_end=idx_end,
            show_rmse=show_rmse,
            state_simulink=state_simulink,
            rewards=rewards,
        )
