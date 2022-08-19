import gym
import copy
import numpy as np
from tqdm import tqdm

from agents import Agent
from agents.idhp import IDHP
from tasks import TrackingTask
from tools import incr_action
from tools.utils import d2r, incr_action_symm, scale_action, scale_action_symm


class IDHP_DC(Agent):
    """
    Decoupled lon/lat wrapper for IDHP with custom learning loop
    """

    def __init__(self, config, task: TrackingTask, env: gym.Env, lon_only=False, lat_only=False):
        super().__init__(config, task, env)

        # Decoupled training profile
        if lon_only and lat_only:
            raise Exception("Arguments 'lon_only' and 'lat_only' are mutually exclusive")
        self.lon_only = lon_only
        self.lat_only = lat_only
        self.lon_lat = bool(not lon_only and not lat_only)

        # Longitudinal agent
        config_lon = copy.deepcopy(config)
        config_lon["actor"]["s_dim"] = config_lon["actor"]["s_dim_lon"]
        config_lon["actor"]["s_states"] = config_lon["actor"]["s_states_lon"]
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

        self.agent_lon = IDHP(config_lon, self.task_lon, env.env_lon)

        # Latheral agent
        config_lat = copy.deepcopy(config)
        config_lat["actor"]["s_dim"] = config_lat["actor"]["s_dim_lat"]
        config_lat["actor"]["s_states"] = config_lat["actor"]["s_states_lat"]
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

        self.agent_lat = IDHP(config_lat, self.task_lat, env.env_lat)

    def learn(self):
        """
        Training loop (online, single episode)
        """

        # Initialize target networks
        self.agent_lon.critic_target.soft_update(self.agent_lon.critic.trainable_weights, tau=1.0)
        self.agent_lat.critic_target.soft_update(self.agent_lat.critic.trainable_weights, tau=1.0)

        # Initialize states
        obs = self.env.reset()
        obs_lon = self.env.obs_to_lon(obs)
        obs_lat = self.env.obs_to_lat(obs)
        tracking_ref_lon = self.agent_lon.tracking_ref[:, 0]
        tracking_ref_lat = self.agent_lat.tracking_ref[:, 0]
        action_env_prev = np.float32(self.env.action)
        action_env_prev_lon = np.float32(self.env.action_to_lon(self.env.action))
        action_env_prev_lat = np.float32(self.env.action_to_lat(self.env.action))
        s_a_lon = self.agent_lon.get_s_a(obs_lon, tracking_ref_lon, action_env_prev_lon)
        s_a_lat = self.agent_lat.get_s_a(obs_lat, tracking_ref_lat, action_env_prev_lat)
        obs_prev_lon = None
        obs_prev_lat = None

        # Forced excitation
        tc = 5.0
        action_exc_A_lon = np.array([[d2r(5.0)]]).T
        action_exc_A_lon = action_exc_A_lon * 0.1 * np.exp(-self.task_lon.timevec / tc)
        action_exc_f_lon = 2 * np.pi * 0.2
        action_exc_lon = action_exc_A_lon * np.sin(action_exc_f_lon * self.task_lon.timevec)
        action_exc_A_lat = np.array([[d2r(5.0), d2r(0.0)]]).T
        action_exc_A_lat = action_exc_A_lat * 0.1 * np.exp(-self.task_lat.timevec / tc)
        action_exc_f_lat = 2 * np.pi * 0.2
        action_exc_lat = action_exc_A_lat * np.sin(action_exc_f_lat * self.task_lat.timevec)
        if self.lon_only:
            action_exc = np.vstack([action_exc_lon, np.zeros_like(action_exc_lat)])
        if self.lat_only:
            action_exc = np.vstack([np.zeros_like(action_exc_lon), action_exc_lat])
        if self.lon_lat:
            action_exc = np.vstack([action_exc_lon, action_exc_lat])

        # Start (online) training loop, single episode
        for t in (bar := tqdm(range(self.task.num_timesteps))):
            bar.set_description("Training IDHP")
            self.t = t
            self.agent_lon.t = t
            self.agent_lat.t = t

            # Get IDHP policy action
            if self.lon_only:
                action_pi_lon = self.agent_lon.get_action(s_a_lon)
                action_pi_lat = np.zeros_like(action_env_prev_lat)
            if self.lat_only:
                action_pi_lon = np.zeros_like(action_env_prev_lon)
                action_pi_lat = self.agent_lat.get_action(s_a_lat)
            if self.lon_lat:
                action_pi_lon = self.agent_lon.get_action(s_a_lon)
                action_pi_lat = self.agent_lat.get_action(s_a_lat)

            # Get action
            action_pi = np.hstack([action_pi_lon, action_pi_lat])
            # action_env = incr_action_symm(action_env_prev, action_pi, self.env, dt=self.task.dt)
            # action_env = action_env + action_exc[:, t]
            action_env = scale_action_symm(action_pi, self.env.action_space)
            action_env = action_env + self.env.action_trim + action_exc[:, t]
            action_env_lon = np.float32(self.env.action_to_lon(action_env))
            action_env_lat = np.float32(self.env.action_to_lat(action_env))

            # Take action
            obs_next = self.env.step(action_env)
            obs_next_lon = self.env.obs_to_lon(obs_next)
            obs_next_lat = self.env.obs_to_lat(obs_next)

            # Check for crash
            if np.isnan(obs_next).sum() > 0:
                print("Crashed")

            # Sample tracking reference and error
            tracking_ref_lon = self.agent_lon.tracking_ref[:, t]
            tracking_ref_lat = self.agent_lat.tracking_ref[:, t]
            tracking_err_lon = tracking_ref_lon - (self.agent_lon.tracking_P @ obs_lon)
            tracking_err_lat = tracking_ref_lat - (self.agent_lat.tracking_P @ obs_lat)

            s_a_next_lon = self.agent_lon.get_s_a(obs_next_lon, tracking_ref_lon, action_env_lon)
            s_a_next_lat = self.agent_lat.get_s_a(obs_next_lat, tracking_ref_lat, action_env_lat)

            # Reward
            reward_lon, reward_grad_lon = self.agent_lon.get_reward(tracking_err_lon)
            reward_lat, reward_grad_lat = self.agent_lat.get_reward(tracking_err_lat)

            # Adaptive learning rate
            tracking_rmse_lon = self.agent_lon.adaptive_lr(tracking_err_lon)
            tracking_rmse_lat = self.agent_lat.adaptive_lr(tracking_err_lat)

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
                )

            # Update model
            if t > 0:
                if self.lon_only or self.lon_lat:
                    self.agent_lon.model.update(
                        obs_lon - obs_prev_lon, action_env_lon - action_env_prev_lon, obs_next_lon - obs_lon
                    )
                if self.lat_only or self.lon_lat:
                    self.agent_lat.model.update(
                        obs_lat - obs_prev_lat, action_env_lat - action_env_prev_lat, obs_next_lat - obs_lat
                    )

            # Update samples
            obs_prev_lon = obs_lon
            obs_prev_lat = obs_lat
            action_env_prev = action_env
            action_env_prev_lon = action_env_lon
            action_env_prev_lat = action_env_lat
            obs_lon = obs_next_lon
            obs_lat = obs_next_lat
            s_a_lon = s_a_next_lon
            s_a_lat = s_a_next_lat

            # Logging
            if self.lon_only or self.lon_lat:
                # self.agent_lon.tracking_rmse_history.append(tracking_rmse)
                # self.agent_lon.reward_history.append(reward)
                # self.agent_lon.s_history.append(s)
                self.agent_lon.actor_weights_history.append(self.agent_lon.actor.get_weights())
                # self.agent_lon.actor_loss_grad_history.append(actor_loss_grad_lon)
                self.agent_lon.critic_weights_history.append(self.agent_lon.critic.get_weights())
                # self.agent_lon.critic_loss_grad_history.append(critic_loss_grad_lon)
                self.agent_lon.F_history.append(self.agent_lon.model.F)
                self.agent_lon.G_history.append(self.agent_lon.model.G)
                self.agent_lon.cov_history.append(self.agent_lon.model.Cov)
                self.agent_lon.epsilon_history.append(self.agent_lon.model.epsilon)
            if self.lat_only or self.lon_lat:
                # self.agent_lat.tracking_rmse_history.append(tracking_rmse)
                # self.agent_lat.reward_history.append(reward)
                # self.agent_lat.s_history.append(s)
                self.agent_lat.actor_weights_history.append(self.agent_lat.actor.get_weights())
                # self.agent_lat.actor_loss_grad_history.append(actor_loss_grad_lat)
                self.agent_lat.critic_weights_history.append(self.agent_lat.critic.get_weights())
                # self.agent_lat.critic_loss_grad_history.append(critic_loss_grad_lat)
                self.agent_lat.F_history.append(self.agent_lat.model.F)
                self.agent_lat.G_history.append(self.agent_lat.model.G)
                self.agent_lat.cov_history.append(self.agent_lat.model.Cov)
                self.agent_lat.epsilon_history.append(self.agent_lat.model.epsilon)

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
