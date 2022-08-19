import copy
import json
import os
import tensorflow.keras as keras
import numpy as np
from tqdm import tqdm
import gym
from agents.base import Agent

from agents.idhpsac import IDHPSAC
from agents.sac import SAC
from tasks.base import TrackingTask
from tools import scale_action
from tools.utils import incr_action_symm, low_pass


class IDHPSAC_DC(Agent):
    """
    Decoupled lon/lat wrapper for IDHPSAC with custom learning loop
    """

    def __init__(self, config, task: TrackingTask, env: gym.Env, agent_sac: SAC, lon_only=False, lat_only=False):
        super().__init__(config, task, env)

        # Decoupled training profile
        if lon_only and lat_only:
            raise Exception("Arguments 'lon_only' and 'lat_only' are mutually exclusive")
        self.lon_only = lon_only
        self.lat_only = lat_only
        self.lon_lat = bool(not lon_only and not lat_only)

        # SAC agent
        self.agent_sac = agent_sac
        self.policy_sac = agent_sac.actor.policy
        self.policy_sac_lon = keras.Model(
            self.policy_sac.input,
            keras.layers.Lambda(lambda x: x[:, env.action_lon_idxs])(self.policy_sac.outputs[0]),
        )
        self.policy_sac_lat = keras.Model(
            self.policy_sac.input,
            keras.layers.Lambda(lambda x: x[:, env.action_lat_idxs])(self.policy_sac.outputs[0]),
        )

        # Longitudinal agent
        config_lon = copy.deepcopy(config)
        config_lon["critic"]["s_dim"] = config_lon["critic"]["s_dim_lon"]
        config_lon["critic"]["s_states"] = config_lon["critic"]["s_states_lon"]
        config_lon["model"]["eps_thresh"] = list(env.obs_to_lon(np.array(config_lon["model"]["eps_thresh"])))

        self.task_lon = copy.deepcopy(task)
        self.task_lon.tracking_ref = {
            _: task.tracking_ref[_] for _ in task.tracking_ref.keys() if _ in env.obs_descr_lon
        }
        self.task_lon.tracking_scale = {
            _: task.tracking_scale[_] for _ in task.tracking_scale.keys() if _ in env.obs_descr_lon
        }
        self.task_lon.tracking_range = {
            _: task.tracking_range[_] for _ in task.tracking_range.keys() if _ in env.obs_descr_lon
        }
        self.task_lon.tracking_thresh = {
            _: task.tracking_thresh[_] for _ in task.tracking_thresh.keys() if _ in env.obs_descr_lon
        }

        self.agent_lon = IDHPSAC(config_lon, self.task_lon, env.env_lon, agent_sac, self.policy_sac_lon)

        # Lateral agent
        config_lat = copy.deepcopy(config)
        config_lat["critic"]["s_dim"] = config_lat["critic"]["s_dim_lat"]
        config_lat["critic"]["s_states"] = config_lat["critic"]["s_states_lat"]
        config_lat["model"]["eps_thresh"] = list(env.obs_to_lat(np.array(config_lat["model"]["eps_thresh"])))

        self.task_lat = copy.deepcopy(task)
        self.task_lat.tracking_ref = {
            _: task.tracking_ref[_] for _ in task.tracking_ref.keys() if _ in env.obs_descr_lat
        }
        self.task_lat.tracking_scale = {
            _: task.tracking_scale[_] for _ in task.tracking_scale.keys() if _ in env.obs_descr_lat
        }
        self.task_lat.tracking_range = {
            _: task.tracking_range[_] for _ in task.tracking_range.keys() if _ in env.obs_descr_lat
        }
        self.task_lat.tracking_thresh = {
            _: task.tracking_thresh[_] for _ in task.tracking_thresh.keys() if _ in env.obs_descr_lat
        }

        self.agent_lat = IDHPSAC(config_lat, self.task_lat, env.env_lat, agent_sac, self.policy_sac_lat)

        # Training
        self.lp_enable = config["lp_enable"]
        self.lp_w0 = config["lp_w0"]

    def learn(self):
        """
        Training loop (online, single episode)
        """

        # Initialize target networks
        self.agent_lon.critic_target.soft_update(self.agent_lon.critic.trainable_variables, tau=1.0)
        self.agent_lat.critic_target.soft_update(self.agent_lat.critic.trainable_variables, tau=1.0)

        # Initialize states
        obs = self.env.reset()
        obs_lon = self.env.obs_to_lon(obs)
        obs_lat = self.env.obs_to_lat(obs)
        tracking_ref = self.tracking_ref[:, 0]
        tracking_ref_lon = self.agent_lon.tracking_ref[:, 0]
        tracking_ref_lat = self.agent_lat.tracking_ref[:, 0]
        action_env_prev = np.float32(self.env.action)
        action_env_prev_filtered = action_env_prev
        action_env_prev_lon = np.float32(self.env.action_to_lon(self.env.action))
        action_env_prev_lat = np.float32(self.env.action_to_lat(self.env.action))
        s_a = self.agent_sac.get_s(obs, tracking_ref)
        obs_prev_lon = None
        obs_prev_lat = None

        # Start (online) training loop, single episode
        for t in (bar := tqdm(range(self.task.num_timesteps))):
            bar.set_description("Training IDHP")
            self.t = t
            self.agent_lon.t = t
            self.agent_lat.t = t

            # Get IDHP policy action
            if self.lon_only:
                action_pi_lon = self.agent_lon.get_action(s_a)
                action_pi_lat = np.zeros_like(action_env_prev_lat)
            if self.lat_only:
                action_pi_lon = np.zeros_like(action_env_prev_lon)
                action_pi_lat = self.agent_lat.get_action(s_a)
            if self.lon_lat:
                action_pi_lon = self.agent_lon.get_action(s_a)
                action_pi_lat = self.agent_lat.get_action(s_a)

            # Get action
            action_pi = np.hstack([action_pi_lon, action_pi_lat])
            if self.config["incr"]:
                action_env = incr_action_symm(action_env_prev, action_pi, self.env, dt=self.task.dt)
            else:
                action_env = scale_action(action_pi, self.env.action_space)

            # Decouple action
            if self.lon_only:
                action_env[self.env.action_lat_idxs] = self.env.action_trim[self.env.action_lat_idxs]
            if self.lat_only:
                action_env[self.env.action_lon_idxs] = self.env.action_trim[self.env.action_lon_idxs]
            action_env_lon = np.float32(self.env.action_to_lon(action_env))
            action_env_lat = np.float32(self.env.action_to_lat(action_env))

            # Low-pass filter (vhf oscillations due to SAC policy can break model identification)
            if self.lp_enable:
                action_env_filtered = low_pass(
                    action_env, action_env_prev_filtered, self.lp_w0 * 2 * np.pi, self.task.dt
                )
            else:
                action_env_filtered = action_env
            action_env_filtered_lon = np.float32(self.env.action_to_lon(action_env_filtered))
            action_env_filtered_lat = np.float32(self.env.action_to_lat(action_env_filtered))

            # Take action
            obs_next = self.env.step(action_env_filtered)
            obs_next_lon = self.env.obs_to_lon(obs_next)
            obs_next_lat = self.env.obs_to_lat(obs_next)

            # Check for crash
            if np.isnan(obs_next).sum() > 0:
                print("Crashed")

            # Sample tracking reference and error
            tracking_ref = self.tracking_ref[:, t]
            tracking_ref_lon = self.agent_lon.tracking_ref[:, t]
            tracking_ref_lat = self.agent_lat.tracking_ref[:, t]
            tracking_err_lon = tracking_ref_lon - (self.agent_lon.tracking_P @ obs_lon)
            tracking_err_lat = tracking_ref_lat - (self.agent_lat.tracking_P @ obs_lat)

            s_a_next = self.agent_sac.get_s(obs_next, tracking_ref)

            # Reward
            if self.lon_only or self.lon_lat:
                _, reward_grad_lon = self.agent_lon.get_reward(tracking_err_lon)
            if self.lat_only or self.lon_lat:
                _, reward_grad_lat = self.agent_lat.get_reward(tracking_err_lat)

            # Adaptive learning rate
            # if self.lon_only or self.lon_lat:
            #     tracking_rmse_lon = self.agent_lon.adaptive_lr(tracking_err_lon)
            # if self.lat_only or self.lon_lat:
            #     tracking_rmse_lat = self.agent_lat.adaptive_lr(tracking_err_lat)

            # Update the actor and critic
            if self.lon_only or self.lon_lat:
                actor_loss_grad_lon, critic_loss_grad_lon = self.agent_lon.update(
                    obs_lon,
                    obs_next_lon,
                    tracking_ref_lon,
                    action_env_lon,
                    action_env_prev_lon,
                    reward_grad_lon,
                    self.agent_lon.model.F,
                    self.agent_lon.model.G,
                    s_full=[obs, tracking_ref, action_env_prev],
                )
            if self.lat_only or self.lon_lat:
                actor_loss_grad_lat, critic_loss_grad_lat = self.agent_lat.update(
                    obs_lat,
                    obs_next_lat,
                    tracking_ref_lat,
                    action_env_lat,
                    action_env_prev_lat,
                    reward_grad_lat,
                    self.agent_lat.model.F,
                    self.agent_lat.model.G,
                    s_full=[obs, tracking_ref, action_env_prev],
                )

            # Update model
            if t > 0:
                if self.lon_only or self.lon_lat:
                    self.agent_lon.model.update(
                        obs_lon - obs_prev_lon,
                        action_env_filtered_lon - action_env_prev_filtered_lon,
                        obs_next_lon - obs_lon,
                    )
                if self.lat_only or self.lon_lat:
                    self.agent_lat.model.update(
                        obs_lat - obs_prev_lat,
                        action_env_filtered_lat - action_env_prev_filtered_lat,
                        obs_next_lat - obs_lat,
                    )

            # Update samples
            obs_prev_lon = obs_lon
            obs_prev_lat = obs_lat
            action_env_prev = action_env
            action_env_prev_lon = action_env_lon
            action_env_prev_lat = action_env_lat
            action_env_prev_filtered = action_env_filtered
            action_env_prev_filtered_lon = action_env_filtered_lon
            action_env_prev_filtered_lat = action_env_filtered_lat
            obs = obs_next
            obs_lon = obs_next_lon
            obs_lat = obs_next_lat
            s_a = s_a_next

            # Logging
            if self.lon_only or self.lon_lat:
                self.agent_lon.actor_weights_history.append(self.agent_lon.actor.get_weights())
                # self.agent_lon.actor_loss_grad_history.append(actor_loss_grad_lon)
                self.agent_lon.critic_weights_history.append(self.agent_lon.critic.get_weights())
                # self.agent_lon.critic_loss_grad_history.append(critic_loss_grad_lon)
                self.agent_lon.F_history.append(self.agent_lon.model.F)
                self.agent_lon.G_history.append(self.agent_lon.model.G)
                self.agent_lon.cov_history.append(self.agent_lon.model.Cov)
                self.agent_lon.epsilon_history.append(self.agent_lon.model.epsilon)
            if self.lat_only or self.lon_lat:
                self.agent_lat.actor_weights_history.append(self.agent_lat.actor.get_weights())
                # self.agent_lat.actor_loss_grad_history.append(actor_loss_grad_lat)
                self.agent_lat.critic_weights_history.append(self.agent_lat.critic.get_weights())
                # self.agent_lat.critic_loss_grad_history.append(critic_loss_grad_lat)
                self.agent_lat.F_history.append(self.agent_lat.model.F)
                self.agent_lat.G_history.append(self.agent_lat.model.G)
                self.agent_lat.cov_history.append(self.agent_lat.model.Cov)
                self.agent_lat.epsilon_history.append(self.agent_lat.model.epsilon)

    def save(self, save_dir):
        """
        Save
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Config
        configfile = open(os.path.join(save_dir, "config.json"), "wt+")
        json.dump(self.config, configfile)

        # Agents
        self.agent_lon.save(os.path.join(save_dir, "agent_lon"))
        self.agent_lat.save(os.path.join(save_dir, "agent_lat"))

    @classmethod
    def load(
        cls, save_dir, task: TrackingTask, env: gym.Env, agent_sac: SAC, lon_only=False, lat_only=False, config=None
    ):
        """
        Load
        TODO: implement SAC agent reference in the saving and loading
        """

        # Config
        if not config:
            configfile = open(os.path.join(save_dir, "config.json"), "r")
            config = json.load(configfile)
        agent = cls(config, task, env, agent_sac, lon_only, lat_only)

        # Agents
        agent.agent_lon.load_weights(os.path.join(save_dir, "agent_lon"))
        agent.agent_lat.load_weights(os.path.join(save_dir, "agent_lat"))

        return agent

    def update(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_s_a(self):
        raise NotImplementedError

    def get_s_c(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError
