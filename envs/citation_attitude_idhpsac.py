import numpy as np
import gym
from gym.spaces import Box
from agents.sac.sac import SAC
from agents.idhpsac.idhpsac import IDHPSAC
from envs.citation import Citation
from tasks.tracking_altitude import TrackAltitude

from tools.utils import d2r, low_pass, scale_action


class CitationAttitudeHybrid(gym.Env):
    """
    Cessna Citation
    Includes pre-trained attitude controller
    """

    def __init__(self, config, task: TrackAltitude, inner_save_dir, config_agent_idhpsac, dt):
        super().__init__()
        self.seed(config["seed"])

        # Inner attitude controller
        self.env_inner = Citation(config, dt=dt, obs_extra=[9])

        # Labels
        self.obs_descr = np.hstack([self.env_inner.obs_descr, "h"])
        self.action_descr = [
            "theta_ref",  # [rad]
        ]

        # Inner IDHPSAC agent
        self.agent_inner_sac: SAC = SAC.load(config_agent_idhpsac["agent_sac"], task, self.env_inner)
        self.agent_inner: IDHPSAC = IDHPSAC.load(
            inner_save_dir, task, self.env_inner, self.agent_inner_sac, config=config_agent_idhpsac
        )

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

        self.lp_enable = config_agent_idhpsac["lp_enable"]
        self.lp_w0 = config_agent_idhpsac["lp_w0"]

        # Logging
        self.action_history = []

    def __str__(self):
        return "citation_attitude"

    def reset(self):
        """
        Reset the environment to initial state
        """

        self.t = 0
        self.obs, obs_extra = self.agent_inner.env.reset()
        self.obs_prev = None
        self.agent_inner.reset_task()

        self.tracking_ref_inner = self.agent_inner.tracking_ref[:, self.t]
        self.tracking_ref_inner = np.hstack([0, self.tracking_ref_inner])
        self.s_a = self.agent_inner.get_s_a(self.obs, self.tracking_ref_inner)

        self.action = 0
        self.action_inner_prev = self.agent_inner.env.action
        self.agent_inner.critic_target.soft_update(self.agent_inner.critic.trainable_weights, tau=1.0)

        self.action_history = []

        return np.hstack([self.obs, obs_extra])

    def step(self, action):
        """
        Excecute a single time step
        """

        # Get attitude controller action
        action_pi = self.agent_inner.get_action(self.s_a)
        action_inner = scale_action(action_pi, self.env_inner.action_space)

        # Take action
        obs_next, obs_extra = self.agent_inner.env.step(action_inner)

        # Low-pass filter
        if self.lp_enable:
            obs_next = low_pass(obs_next, self.obs, self.lp_w0 * 2 * np.pi, self.task.dt)

        # Reward
        theta_ref = action
        self.tracking_ref_inner = self.agent_inner.tracking_ref[:, self.t]
        self.tracking_ref_inner = np.hstack([theta_ref, self.tracking_ref_inner])

        tracking_err = self.tracking_ref_inner - (self.agent_inner.tracking_P @ self.obs)  # ? obs_next
        s_a_next = self.agent_inner.get_s_a(obs_next, self.tracking_ref_inner)

        reward, reward_grad = self.agent_inner.get_reward(tracking_err)

        # Update the actor and critic
        self.agent_inner.update(
            self.obs,
            obs_next,
            self.tracking_ref_inner,
            action_inner,
            self.action_inner_prev,
            reward_grad,
            self.agent_inner.model.F,
            self.agent_inner.model.G,
        )

        # Update model
        if self.t > 0:
            self.agent_inner.model.update(
                self.obs - self.obs_prev,
                action_inner - self.action_inner_prev,
                obs_next - self.obs,
            )

        # Update samples
        self.obs_prev = self.obs
        self.action_inner_prev = action_inner
        self.obs = obs_next
        self.s_a = s_a_next
        self.t += 1

        # Logging
        self.action_history.append(action)

        # Logging
        self.agent_inner.actor_weights_history.append(self.agent_inner.actor.get_weights())
        self.agent_inner.critic_weights_history.append(self.agent_inner.critic.get_weights())
        self.agent_inner.F_history.append(self.agent_inner.model.F)
        self.agent_inner.G_history.append(self.agent_inner.model.G)
        self.agent_inner.cov_history.append(self.agent_inner.model.Cov)
        self.agent_inner.epsilon_history.append(self.agent_inner.model.epsilon)

        return np.hstack([self.obs, obs_extra])

    def render(
        self,
        task,
        env_sac=None,
        idx_end=None,
        show_rmse=True,
        lr_warmup=None,
        state_simulink=None,
        rewards=None,
    ):
        """
        Visualize environment response
        """

        tracking_ref_internal = {"theta": self.action_history}
        self.env_inner.render(
            task,
            env_sac=env_sac,
            tracking_ref_internal=tracking_ref_internal,
            idx_end=idx_end,
            show_rmse=show_rmse,
            state_simulink=state_simulink,
            rewards=rewards,
        )
