import json
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from tqdm import tqdm
import gym

from agents import Agent
from agents.idhp import Actor
from agents.idhp import Critic
from agents.idhp import Model
from tasks import TrackingTask
from tools import scale_action, set_random_seed, array32, clip
from tools.utils import concat, incr_action, incr_action_symm, low_pass, scale_action_symm


class IDHP(Agent):
    """
    Incremental Dual Heuristic Programming (IDHP)
    On-Policy Online Reinforcement Learning using Incremental Model
    Tracking controller
    """

    def __init__(self, config, task: TrackingTask, env: gym.Env):
        super().__init__(config, task, env)

        # Actor
        self.actor = Actor(config, self.env)

        # Critic
        self.critic = Critic(config, self.env)
        self.critic_target = Critic(config, self.env)

        # Incremental model
        self.model = Model(config, self.env)

        # Training
        self.reward_scale = config["reward_scale"]
        self.clip_norm_c = None
        self.clip_value_c = None
        self.clip_norm_a = None
        self.clip_value_a = None
        self.lp_enable = config["lp_enable"]
        self.lp_w0 = config["lp_w0"]
        self.rmse_thresh = self.tracking_thresh

        # Adaptive learning rate
        self.lr_adapt = config["lr_adapt"]
        self.lr_warmup = config["lr_warmup"]
        self.rmse_size = config["lr_thresh_rmse_size"]
        self.rmse_delay = config["lr_thresh_delay"]

        self.lr_actor_high = np.float32(config["actor"]["lr_high"])
        self.lr_actor_low = np.float32(config["actor"]["lr_low"])
        self.lr_critic_high = np.float32(config["critic"]["lr_high"])
        self.lr_critic_low = np.float32(config["critic"]["lr_low"])
        if config["identity_init"]:
            # self.lr_actor = tf.Variable(0.0, trainable=False)
            # self.lr_critic = tf.Variable(0.0, trainable=False)
            self.lr_actor = tf.Variable(self.lr_actor_high, trainable=False)
            self.lr_critic = tf.Variable(self.lr_critic_high, trainable=False)
        else:
            self.lr_actor = tf.Variable(self.lr_actor_high, trainable=False)
            self.lr_critic = tf.Variable(self.lr_critic_high, trainable=False)

        # Optimizers
        self.actor_optimizer = SGD(self.lr_actor, clipnorm=self.clip_norm_a, clipvalue=self.clip_value_a)
        self.critic_optimizer = SGD(self.lr_critic, clipnorm=self.clip_norm_c, clipvalue=self.clip_value_c)

        # Logging
        self.reward_history = []
        self.tracking_err_history = []
        self.tracking_rmse_history = []
        self.tracking_rmse_delay_history = []
        self.lr_scale_history = []
        self.actor_weights_history = []
        self.actor_loss_grad_history = []
        self.critic_weights_history = []
        self.critic_loss_grad_history = []
        self.F_history = []
        self.G_history = []
        self.cov_history = []
        self.epsilon_history = []
        self.s_history = []

    @tf.function(experimental_relax_shapes=True)
    def get_action(self, s):
        """
        Give the policy's action
        """

        action = self.actor(s[np.newaxis])
        action = tf.reshape(action, [-1])  # remove batch dimension

        return action

    def get_s_a(self, obs, tracking_ref, t_f=False):
        """
        Get input state vector for actor
        """

        # Construct input state vector (size = self.s_a_dim)
        tracking_err = tracking_ref - (self.tracking_P @ obs)

        s = concat(
            [
                [obs[_] for _ in self.s_a_states],
                tracking_err,
            ],
            axis=-1,
            t_f=t_f,
        )

        # Normalized, with incr control
        # s = concat(
        #     [
        #         [obs[_] / self.env.obs_norm[_] for _ in self.s_a_states],
        #         tracking_err / self.tracking_ref_max,
        #         action_prev,
        #     ],
        #     axis=-1,
        #     t_f=t_f,
        # )

        return s

    def get_s_c(self, obs, tracking_ref, action_prev, t_f=False):
        """
        Get input state vector for critic
        """

        # Construct input state vector (size = self.s_a_dim)
        tracking_err = tracking_ref - (self.tracking_P @ obs)

        s = concat(
            [
                [obs[_] for _ in self.s_c_states],
                tracking_err,
            ],
            axis=-1,
            t_f=t_f,
        )

        # Normalized, with incr control
        # s = concat(
        #     [
        #         [obs[_] / self.env.obs_norm[_] for _ in self.s_c_states],
        #         tracking_err / self.tracking_ref_max,
        #         action_prev,
        #     ],
        #     axis=-1,
        #     t_f=t_f,
        # )

        return s

    def get_reward(self, tracking_err):
        """
        Calculate the reward and gradient of the reward w.r.t the observed state
        Reward = negative weighted squared tracking error
        """

        # Calculate reward (Q = tracking error weight matrix)
        errQ = tracking_err.T @ self.tracking_Q
        reward = -errQ @ tracking_err
        reward = reward * self.reward_scale

        # Calculate reward gradient (P = tracking selection matrix)
        reward_grad = 2.0 * errQ @ self.tracking_P  # use -2.0 when tracking_err is x - x_ref
        reward_grad = reward_grad * self.reward_scale

        return np.float32(reward), np.float32(reward_grad)

    def learn(self):
        """
        Training loop (online, single episode)
        """

        # Initialize target networks
        self.critic_target.soft_update(self.critic.trainable_weights, tau=1.0)

        # Initialize states
        obs = self.env.reset()
        tracking_ref = self.tracking_ref[:, 0]
        action_env_prev = np.float32(self.env.action)
        action_env_prev_filtered = action_env_prev
        s_a = self.get_s_a(obs, tracking_ref)
        obs_prev = None
        success = True

        # Start (online) training loop, single episode
        episode_return = 0
        for t in (bar := tqdm(range(self.task.num_timesteps))):
            bar.set_description("Training IDHP")
            self.t = t

            # Get action
            action_pi = self.get_action(s_a)
            if self.config["incr"]:
                action_env = incr_action_symm(action_env_prev, action_pi, self.env, dt=self.task.dt)
            else:
                action_env = scale_action(action_pi, self.env.action_space)

            # Low-pass filter (vhf oscillations due to SAC policy can break model identification)
            if self.lp_enable:
                action_env_filtered = low_pass(
                    action_env, action_env_prev_filtered, self.lp_w0 * 2 * np.pi, self.task.dt
                )
            else:
                action_env_filtered = action_env

            # Take action
            obs_next = self.env.step(action_env_filtered)

            # Check for crash
            if np.isnan(obs_next).sum() > 0:
                print("Crashed")
                success = False
                break

            # Sample tracking reference and error
            tracking_ref = self.tracking_ref[:, t]
            tracking_err = tracking_ref - (self.tracking_P @ obs)
            s_a_next = self.get_s_a(obs_next, tracking_ref)

            # Reward
            reward, reward_grad = self.get_reward(tracking_err)
            episode_return += reward

            # Adaptive learning rate
            # tracking_rmse = self.adaptive_lr(tracking_err)

            # Update the actor and critic
            actor_loss_grad, critic_loss_grad = self.update(
                obs, obs_next, tracking_ref, action_env, action_env_prev, reward_grad, self.model.F, self.model.G
            )

            # Update model
            if t > 0:
                self.model.update(obs - obs_prev, action_env_filtered - action_env_prev_filtered, obs_next - obs)

            # Update samples
            obs_prev = obs
            action_env_prev = action_env
            action_env_prev_filtered = action_env_filtered
            obs = obs_next
            s_a = s_a_next

            # Logging
            self.actor_weights_history.append(self.actor.get_weights())
            # self.actor_loss_grad_history.append(actor_loss_grad)
            self.critic_weights_history.append(self.critic.get_weights())
            # self.critic_loss_grad_history.append(critic_loss_grad)
            self.F_history.append(self.model.F)
            self.G_history.append(self.model.G)
            self.cov_history.append(self.model.Cov)
            self.epsilon_history.append(self.model.epsilon)

        return success

    @tf.function(experimental_relax_shapes=True)
    def update(self, obs, obs_next, tracking_ref, action_env, action_env_prev, reward_grad, F, G, **kwargs):
        """
        Update the actor and critic
        """

        # Optionally get full samples when models are decoupled (for hybrid SAC actor input)
        if "s_full" in kwargs.keys():
            obs_full, tracking_ref_full, action_env_prev_full = kwargs["s_full"]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(obs)
            if "s_full" in kwargs.keys():
                tape.watch(obs_full)

            # Get input state for critic
            s_c = self.get_s_c(obs, tracking_ref, action_env_prev, t_f=True)
            s_c_next = self.get_s_c(obs, tracking_ref, action_env, t_f=True)

            # Get input state for actor
            if "s_full" in kwargs.keys():  # support for decoupled actor/critic inputs
                s_a = self.get_s_a(obs_full, tracking_ref_full, t_f=True)
            else:
                s_a = self.get_s_a(obs, tracking_ref, t_f=True)

            # Actor call
            action = self.actor(s_a[np.newaxis])
            action_nodes = tf.split(action, self.actor.output_dim, axis=1)

            # Critic call
            lmbda = self.critic(s_c[np.newaxis])
            lmbda_next = self.critic_target(s_c_next[np.newaxis])

        # Actor loss
        actor_loss_grad = -(reward_grad + self.gamma * lmbda_next) @ G
        actor_loss_grad = tape.gradient(action, self.actor.trainable_weights, output_gradients=actor_loss_grad)

        # Actor update
        self.actor_optimizer.apply_gradients(zip(actor_loss_grad, self.actor.trainable_weights))

        # Critic loss
        grads = []
        for i in range(len(action_nodes)):
            if "s_full" in kwargs.keys():
                grads.append(self.env.obs_to(tape.gradient(action_nodes[i], obs_full)))
            else:
                grads.append(tape.gradient(action_nodes[i], obs))
        dadx = tf.stack(grads, axis=0)
        td_err_ds = (reward_grad + self.gamma * lmbda_next) @ (F + G @ dadx) - lmbda
        critic_loss_grad = -td_err_ds
        critic_loss_grad = tape.gradient(lmbda, self.critic.trainable_weights, output_gradients=critic_loss_grad)

        # Critic update
        self.critic_optimizer.apply_gradients(zip(critic_loss_grad, self.critic.trainable_weights))
        self.critic_target.soft_update(self.critic.trainable_weights, tau=self.tau)

        del tape

        return actor_loss_grad, critic_loss_grad

    def adaptive_lr(self, tracking_err):
        """
        Change learning rates depending on tracking RMSE
        """

        # Add current tracking error
        self.tracking_err_history.append(tracking_err)

        # Tracking error RMSE of the last x measurements
        tracking_err_history = np.array(self.tracking_err_history).T
        tracking_rmse = np.sqrt(np.mean((tracking_err_history[:, -self.rmse_size :]) ** 2, axis=1))
        self.tracking_rmse_history.append(tracking_rmse)

        # Weighted sum
        # tracking_rmse = np.sum((tracking_rmse.T @ self.tracking_Q))

        # All of the rmse signals have been below the threshold for at least self.rmse_delay => lr_low
        tracking_rmse_history = np.array(self.tracking_rmse_history)[-self.rmse_delay :, :]
        tracking_rmse_history_lower = np.less(tracking_rmse_history, self.tracking_thresh)

        self.tracking_rmse_delay_history.append(np.all(tracking_rmse_history_lower, axis=0))
        lr_low = np.all(tracking_rmse_history_lower)

        # Determine if any tracking rmse are above the threshold
        # thresh = np.sum(np.greater(tracking_rmse, self.tracking_thresh)) == 1

        # Alternative nmae
        # y_err = tracking_err_history[:, -self.rmse_size :]
        # y_range = np.array([self.task.tracking_range[_] for _ in self.tracking_descr])

        # mae_vec = np.mean(np.abs(y_err), axis=1)
        # nmae_vec = mae_vec / y_range
        # nmae = np.mean(nmae_vec)

        # tracking_rmse = nmae

        # self.lr_actor.assign(self.lr_actor * (1 - 0.0002))
        # self.lr_critic.assign(self.lr_critic * (1 - 0.0002))

        # Set non-adaptive learning rates
        if not self.lr_adapt and self.t >= self.lr_warmup:
            if self.lr_actor != self.lr_actor_high:
                self.lr_actor.assign(self.lr_actor_high)
            if self.lr_critic != self.lr_critic_high:
                self.lr_critic.assign(self.lr_critic_high)

        # Set adaptive learning rates
        elif self.lr_adapt and self.t >= self.lr_warmup:
            if not lr_low:
                if self.lr_actor != self.lr_actor_high:
                    self.lr_actor.assign(self.lr_actor_high)
                if self.lr_critic != self.lr_critic_high:
                    self.lr_critic.assign(self.lr_critic_high)
            else:
                if self.lr_actor != self.lr_actor_low:
                    self.lr_actor.assign(self.lr_actor_low)
                if self.lr_critic != self.lr_critic_low:
                    self.lr_critic.assign(self.lr_critic_low)

        # if np.all(np.less(tracking_rmse, self.rmse_thresh)):
        #     # lr stays constant below the threshold
        #     pass
        # elif np.all(np.less(tracking_rmse, 2 * self.rmse_thresh)):
        #     self.lr_actor.assign(
        #         tf.clip_by_value(self.lr_actor - self.lr_actor_high * 0.01, self.lr_actor_low, self.lr_actor_high)
        #     )
        #     self.lr_critic.assign(
        #         tf.clip_by_value(
        #             self.lr_critic - self.lr_critic_high * 0.01, self.lr_critic_low, self.lr_critic_high
        #         )
        #     )
        # else:
        #     self.lr_actor.assign(
        #         tf.clip_by_value(self.lr_actor + self.lr_actor_high * 0.01, self.lr_actor_low, self.lr_actor_high)
        #     )
        #     self.lr_critic.assign(
        #         tf.clip_by_value(
        #             self.lr_critic + self.lr_critic_high * 0.01, self.lr_critic_low, self.lr_critic_high
        #         )
        #     )

        # rmse_error_scale = np.sum(np.abs(tracking_rmse - self.rmse_thresh)) * 0.1

        # if np.all(np.less(tracking_rmse, self.rmse_thresh)):
        #     # lr_actor = self.lr_actor - self.lr_actor_high * 0.0005
        #     # lr_critic = self.lr_critic - self.lr_critic_high * 0.0005
        #     lr_actor = self.lr_actor - self.lr_actor_high * rmse_error_scale * 2
        #     lr_critic = self.lr_critic - self.lr_critic_high * rmse_error_scale * 2

        #     self.lr_actor.assign(tf.clip_by_value(lr_actor, self.lr_actor_low, self.lr_actor_high))
        #     self.lr_critic.assign(tf.clip_by_value(lr_critic, self.lr_critic_low, self.lr_critic_high))
        # else:
        #     lr_actor = self.lr_actor + self.lr_actor_high * rmse_error_scale * 0.5
        #     lr_critic = self.lr_critic + self.lr_critic_high * rmse_error_scale * 0.5

        #     self.lr_actor.assign(tf.clip_by_value(lr_actor, self.lr_actor_low, self.lr_actor_high))
        #     self.lr_critic.assign(tf.clip_by_value(lr_critic, self.lr_critic_low, self.lr_critic_high))

        self.lr_scale_history.append(self.lr_actor / self.lr_actor_high)

        return tracking_rmse

    def save(self, save_dir):
        """
        Save
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Config
        configfile = open(os.path.join(save_dir, "config.json"), "wt+")
        json.dump(self.config, configfile)

        # Network weights
        self.actor.save_weights(os.path.join(save_dir, "actor"))
        self.critic.save_weights(os.path.join(save_dir, "critic"))
        self.critic_target.save_weights(os.path.join(save_dir, "critic_target"))

        # Model weights
        self.model.save_weights(os.path.join(save_dir, "model"))

    def load_weights(self, save_dir):
        """
        Load (only weights)
        """

        # Network weights
        self.actor.load_weights(os.path.join(save_dir, "actor")).expect_partial()
        self.critic.load_weights(os.path.join(save_dir, "critic")).expect_partial()
        self.critic_target.load_weights(os.path.join(save_dir, "critic_target")).expect_partial()

        # Model weights
        self.model.load_weights(os.path.join(save_dir, "model"))

    @classmethod
    def load(cls, save_dir, task: TrackingTask, env: gym.Env):
        """
        Load
        """

        # Config
        configfile = open(os.path.join(save_dir, "config.json"), "r")
        config = json.load(configfile)
        agent = cls(config, task, env)

        # Network weights
        agent.actor.load_weights(os.path.join(save_dir, "actor")).expect_partial()
        agent.critic.load_weights(os.path.join(save_dir, "critic")).expect_partial()
        agent.critic_target.load_weights(os.path.join(save_dir, "critic_target")).expect_partial()

        # Model weights
        agent.model.load_weights(os.path.join(save_dir, "model"))

        return agent
