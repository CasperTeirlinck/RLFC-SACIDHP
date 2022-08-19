import json
import os
import tensorflow.keras as keras
import gym

from agents.idhpsac import Actor
from agents.idhp import IDHP
from agents.sac import SAC
from tasks.base import TrackingTask
from tools.utils import concat


class IDHPSAC(IDHP):
    """
    Incremental Dual Heuristic Programming Soft Actor Critic (IDHPSAC)
    On-Policy Online Reinforcement Learning using Incremental Model
    using SAC Reference Policy
    """

    def __init__(self, config, task: TrackingTask, env: gym.Env, agent_sac: SAC, policy_sac: keras.Model = None):
        config["actor"]["s_dim"] = agent_sac.s_a_dim
        config["actor"]["s_states"] = agent_sac.s_a_states
        super().__init__(config, task, env)

        # SAC agent
        self.agent_sac = agent_sac

        # Actor
        if policy_sac is None:
            policy_sac = keras.Model(
                agent_sac.actor.policy.input,
                agent_sac.actor.policy.outputs[0],
            )
        self.actor = Actor(config, env, policy_sac)

    def get_s_a(self, obs, tracking_ref, t_f=False):
        """
        Get input state vector for actor
        """

        return self.agent_sac.get_s(obs, tracking_ref, t_f=t_f)

    def get_s_c(self, obs, tracking_ref, action_prev, t_f=False):
        """
        Get input state vector for critic
        """

        # Construct input state vector
        tracking_err = tracking_ref - (self.tracking_P @ obs)
        s = concat(
            [
                # tracking_err,
                tracking_err @ self.tracking_Q,
                [obs[_] for _ in self.s_c_states],
            ],
            axis=-1,
            t_f=t_f,
        )

        return s

    @classmethod
    def load(cls, save_dir, task: TrackingTask, env: gym.Env, agent_sac: SAC, config=None):
        """
        Load
        TODO: implement SAC agent reference in the saving and loading
        """

        # Config
        if not config:
            configfile = open(os.path.join(save_dir, "config.json"), "r")
            config = json.load(configfile)
        agent = cls(config, task, env, agent_sac)

        # IDHP weights
        agent.load_weights(save_dir)

        return agent
