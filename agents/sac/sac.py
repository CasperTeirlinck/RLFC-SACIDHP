import sys
import gym
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tqdm import tqdm

from agents import Agent
from tasks import TrackingTask
from agents.sac import Actor, Critic, ReplayBuffer
from tools import concat, incr_action
from tools.utils import d2r, low_pass, scale_action


class SAC(Agent):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
    """

    def __init__(self, config, task: TrackingTask, env: gym.Env):
        super().__init__(config, task, env)

        # Actor
        self.actor = Actor(config, env)

        # Critic
        self.critic_1 = Critic(config, env)
        self.critic_2 = Critic(config, env)
        self.critic_1_target = Critic(config, env)
        self.critic_2_target = Critic(config, env)

        # Replay buffer
        self.buffer = ReplayBuffer(config, env)
        self.batch_size = config["batch_size"]

        # Entropy coefficient
        self.log_ent_coef = tf.Variable(config["ent_coef_init"])
        self.ent_coef = tfp.util.DeferredTensor(self.log_ent_coef, tf.exp)
        self.ent_target = -np.prod(env.action_space.shape)

        # Training
        self.lr_init = config["lr_init"]
        self.lr = tf.Variable(self.lr_init, trainable=False)
        self.caps_coef_s = config["caps_coef_s"]
        self.caps_coef_t = config["caps_coef_t"]
        self.caps_std = config["caps_std"]
        self.crash_stop = False
        self.incr = bool(config["incr"])
        self.update_every = config["update_every"]
        self.logging = False

        # Optimizers
        self.actor_optimizer = Adam(self.lr)
        self.critic_1_optimizer = Adam(self.lr)
        self.critic_2_optimizer = Adam(self.lr)
        self.ent_coef_optimizer = Adam(self.lr)

        # Logging
        self.actor_loss_history = []
        self.actor_loss_t_history = []
        self.actor_loss_s_history = []
        self.critic_1_loss_history = []
        self.critic_2_loss_history = []

    @tf.function(jit_compile=True)
    def get_action(self, s, deterministic=False):
        """
        Give the policy's action
        """

        # Sample policy
        action, _ = self.actor.sample(s, deterministic)
        action = tf.reshape(action, [-1])  # remove batch dimension

        return action

    def get_s(self, obs, tracking_ref, t_f=False):
        """
        Get input state vector for actor and critic
        """

        # Construct input state vector
        tracking_err = tracking_ref - (self.tracking_P @ obs)
        s = concat(
            [
                tracking_err @ self.tracking_Q,
                [obs[_] for _ in self.s_states],
            ],
            axis=-1,
            t_f=t_f,
        )

        return s

    def get_s_a(self):
        raise NotImplementedError

    def get_s_c(self):
        raise NotImplementedError

    def get_reward(self, tracking_err):
        """
        Calculate the reward
        """

        # 1
        reward_vec = np.abs(
            np.clip(tracking_err @ self.tracking_Q, -np.ones(tracking_err.shape), np.ones(tracking_err.shape))
        )
        reward = -1 / 3 * reward_vec.sum()

        # 2
        # reward = -np.average(
        #     np.clip(
        #         np.abs(tracking_err / self.tracking_max), np.zeros(tracking_err.shape), np.ones(tracking_err.shape)
        #     ),
        #     weights=np.diagonal(self.tracking_Q),
        # )

        # 3
        # reward = -np.clip(np.mean(abs(tracking_err @ self.tracking_Q)), 0, 1.0)

        return np.float32(reward)

    def evaluate(self, verbose=True, lon_only=False, lat_only=False, crash_attitude=False, lp_w0=None):
        """
        Evaluation loop (online, single episode)
        """

        # Initialize states
        obs = self.env.reset()
        tracking_ref = self.tracking_ref[:, 0]
        action_env = self.env.action
        s = self.get_s(obs, tracking_ref)
        crash = False

        # Evaluation loop
        episode_return = 0
        if verbose:
            rnge = (bar := tqdm(range(self.task.num_timesteps)))
            bar.set_description("Evaluating SAC")
        else:
            rnge = range(self.task.num_timesteps)
        for t in rnge:

            # Get action
            action_pi = self.get_action(s[np.newaxis], deterministic=True)
            if self.incr:
                action_env = incr_action(action_env, action_pi, self.env, dt=self.task.dt)
            else:
                action_env = scale_action(action_pi, self.env.action_space)
            if lon_only:
                action_env = action_env.numpy()
                action_env[self.env.action_lat_idxs] = self.env.action_trim[self.env.action_lat_idxs]
            if lat_only:
                action_env = action_env.numpy()
                action_env[self.env.action_lon_idxs] = self.env.action_trim[self.env.action_lon_idxs] - [d2r(0.5)]

            # Take action
            obs_next = self.env.step(action_env)
            if lp_w0:
                obs = low_pass(obs_next, obs, lp_w0 * 2 * np.pi, self.task.dt)
            else:
                obs = obs_next

            # Check for crash
            if np.isnan(obs).sum() > 0:
                crash = True
                break
            if crash_attitude:
                if abs(obs[self.env.obs_descr.tolist().index("theta")]) > d2r(50) or abs(
                    obs[self.env.obs_descr.tolist().index("phi")]
                ) > d2r(75):
                    crash = True
                    break

            # Sample tracking reference and error
            tracking_ref = self.tracking_ref[:, t]
            tracking_err = tracking_ref - (self.tracking_P @ obs)
            s = self.get_s(obs, tracking_ref)

            # Reward
            reward = self.get_reward(tracking_err)
            episode_return += reward

        if crash:
            return None
        else:
            return episode_return

    def learn(self, num_timesteps, callback=None):
        """
        Training loop (offline, multiple episodes)
        """

        # Initialize callback
        if callback is not None:
            callback.init_callback(self)

        # Initialize target networks
        self.critic_1_target.soft_update(self.critic_1.trainable_weights, tau=1.0)
        self.critic_2_target.soft_update(self.critic_2.trainable_weights, tau=1.0)

        # Initialize states
        obs = self.env.reset()
        tracking_ref = self.tracking_ref[:, 0]
        action_env = np.float32(self.env.action)
        s = self.get_s(obs, tracking_ref)

        # Training loop
        t = 0
        update = 0
        episode = 1
        episode_return = 0
        crashes = 0
        crashes_eval = 0
        done = False
        crash = False
        for step in (bar := tqdm(range(num_timesteps))):
            self.t = step
            bar.set_postfix(
                {
                    "ep": episode,
                    "crashes": crashes if not self.crash_stop else "-",
                    "crashes_eval": crashes_eval,
                    "return_avg": callback.episode_return_avg if callback else "-",
                    "return_best": callback.return_eval_best if callback else "-",
                    "nmae_best": f"{callback.nmae_eval_best * 100 :.2f}%" if callback else "-",
                }
            )

            # Next episode
            if done:

                # Handle done
                if not crash:
                    if callback is not None:
                        crash_eval = not callback.on_done(episode_return)
                        if crash_eval:
                            crashes_eval += 1
                # Handle training crash
                else:
                    if self.crash_stop:
                        if callback is not None:
                            callback.on_crash()
                        break
                    else:
                        crash = False

                # Initialize states
                obs = self.env.reset()
                self.reset_task()
                tracking_ref = self.tracking_ref[:, 0]
                action_env = np.float32(self.env.action)
                s = self.get_s(obs, tracking_ref)
                t = 0

                episode_return = 0
                episode += 1

            # Get action
            action_pi = self.get_action(s[np.newaxis])
            if self.incr:
                action_env = incr_action(action_env, action_pi, self.env, dt=self.task.dt)
            else:
                action_env = scale_action(action_pi, self.env.action_space)

            # Take action
            obs = self.env.step(action_env)

            # Check for training crash
            if np.isnan(obs).sum() > 0:
                crashes += 1
                crash = True
                done = True
                continue

            # Sample tracking reference and error
            tracking_ref = self.tracking_ref[:, t]
            tracking_err = tracking_ref - (self.tracking_P @ obs)
            s_next = self.get_s(obs, tracking_ref)

            # Reward
            reward = self.get_reward(tracking_err)
            episode_return += reward

            # Store samples in replay buffer
            t += 1
            done = bool(t >= self.task.num_timesteps)
            self.buffer.store(s, action_pi, reward, s_next, done)
            s = s_next

            # Update
            if self.buffer.can_sample(self.batch_size):
                update += 1

                # Adaptive learning rate
                lr = self.lr_init * (1.0 - step / num_timesteps)
                self.lr.assign(lr)

                # Update step
                if update % self.update_every == 0:
                    batch = self.buffer.sample(self.batch_size)
                    actor_loss, actor_loss_t, actor_loss_s, critic_1_loss, critic_2_loss = self.update(batch)

                    # Logging
                    if self.logging:
                        self.actor_loss_history.append(actor_loss)
                        self.actor_loss_t_history.append(actor_loss_t)
                        self.actor_loss_s_history.append(actor_loss_s)
                        self.critic_1_loss_history.append(critic_1_loss)
                        self.critic_2_loss_history.append(critic_2_loss)

        # Stop crash
        if self.crash_stop and crash == True:
            return False

        return True

    @tf.function(jit_compile=True)
    def update(self, batch):
        """
        Update the actor and critic
        """

        # Sample replay buffer
        s_batch, action_batch, reward_batch, s_next_batch, done_batch = batch

        # Compute losses
        with tf.GradientTape(persistent=True) as tape:

            # Generate new actions
            action_pi, action_mean, log_prob = self.actor.sample(s_batch, return_mean=True)
            action_pi_next, action_mean_next, log_prob_next = self.actor.sample(s_next_batch, return_mean=True)
            action_mean_normal, _ = self.actor.sample(
                tf.random.normal(tf.shape(s_batch), mean=s_batch, stddev=self.caps_std), deterministic=True
            )

            # Call critics
            q1_pi = self.critic_1(s_batch, action_pi)
            q2_pi = self.critic_2(s_batch, action_pi)
            q = tf.minimum(q1_pi, q2_pi)

            q1_pi_target_next = self.critic_1_target(s_next_batch, action_pi_next)
            q2_pi_target_next = self.critic_2_target(s_next_batch, action_pi_next)
            q_target_next = tf.minimum(q1_pi_target_next, q2_pi_target_next)

            # Critic loss
            v_next = q_target_next - self.ent_coef * log_prob_next
            q_tdtarget = reward_batch + self.gamma * v_next * (1 - tf.cast(done_batch, tf.float32))
            critic_1_loss = 0.5 * keras.losses.MSE(self.critic_1(s_batch, action_batch), q_tdtarget)
            critic_2_loss = 0.5 * keras.losses.MSE(self.critic_2(s_batch, action_batch), q_tdtarget)

            # Actor loss
            actor_loss = tf.nn.compute_average_loss(self.ent_coef * log_prob - q)
            actor_loss_t = self.caps_coef_t * tf.nn.l2_loss(action_mean_next - action_mean) / tf.shape(action_mean)[0]
            actor_loss_s = self.caps_coef_s * tf.nn.l2_loss(action_mean_normal - action_mean) / tf.shape(action_mean)[0]
            actor_loss_total = actor_loss + actor_loss_t + actor_loss_s

            # Entropy coef loss
            ent_coef_loss = -tf.nn.compute_average_loss(self.ent_coef * tf.stop_gradient(log_prob + self.ent_target))

        # Critics update
        critic_1_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_weights)
        critic_2_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_weights)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_weights))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_weights))

        # Actor update
        actor_grad = tape.gradient(actor_loss_total, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_weights))

        # Entropy coeff update
        ent_coef_grad = tape.gradient(ent_coef_loss, [self.log_ent_coef])
        self.ent_coef_optimizer.apply_gradients(zip(ent_coef_grad, [self.log_ent_coef]))

        del tape

        # Update target networks
        self.critic_1_target.soft_update(self.critic_1.trainable_weights, tau=self.tau)
        self.critic_2_target.soft_update(self.critic_2.trainable_weights, tau=self.tau)

        return actor_loss, actor_loss_t, actor_loss_s, critic_1_loss, critic_2_loss

    def save(self, save_dir, minimal=False):
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
        self.critic_1.save_weights(os.path.join(save_dir, "critic_1"))
        self.critic_2.save_weights(os.path.join(save_dir, "critic_2"))
        self.critic_1_target.save_weights(os.path.join(save_dir, "critic_1_target"))
        self.critic_2_target.save_weights(os.path.join(save_dir, "critic_2_target"))

        # Logging
        if not minimal:
            np.savez(
                os.path.join(save_dir, "logging") + ".npz",
                actor_loss=self.actor_loss_history,
                actor_loss_t=self.actor_loss_t_history,
                actor_loss_s=self.actor_loss_s_history,
                critic_1_loss=self.critic_1_loss_history,
                critic_2_loss=self.critic_2_loss_history,
            )

    @classmethod
    def load(cls, save_dir, task: TrackingTask, env: gym.Env, logs=False):
        """
        Load
        """

        # Config
        configfile = open(os.path.join(save_dir, "config.json"), "r")
        config = json.load(configfile)
        agent = cls(config, task, env)

        # Network weights
        agent.actor.load_weights(os.path.join(save_dir, "actor")).expect_partial()
        agent.critic_1.load_weights(os.path.join(save_dir, "critic_1")).expect_partial()
        agent.critic_2.load_weights(os.path.join(save_dir, "critic_2")).expect_partial()
        agent.critic_1_target.load_weights(os.path.join(save_dir, "critic_1_target")).expect_partial()
        agent.critic_2_target.load_weights(os.path.join(save_dir, "critic_2_target")).expect_partial()

        # Logging
        if logs:
            npzfile = np.load(os.path.join(save_dir, "logging") + ".npz")
            agent.actor_loss_history = npzfile["actor_loss_history"]
            agent.actor_loss_t_history = npzfile["actor_loss_t_history"]
            agent.actor_loss_s_history = npzfile["actor_loss_s_history"]
            agent.critic_1_loss_history = npzfile["critic_1_loss_history"]
            agent.critic_2_loss_history = npzfile["critic_2_loss_history"]

        return agent

    @classmethod
    def load_npz(cls, save_dir, task: TrackingTask, env: gym.Env, config=None):
        """
        Load
        """

        # Config
        if not config:
            configfile = open(os.path.join(save_dir, "config.json"), "r")
            config = json.load(configfile)

        # Params npz
        npzfile = np.load(os.path.join(save_dir, "params.npz"))

        # Instantiate
        agent = cls(config, task, env)

        # Load weights
        for layer_name in ["hidden_1", "layer_norm_1", "hidden_2", "layer_norm_2", "output_mean", "output_std"]:
            weights = npzfile[layer_name + "_w"]
            biases = npzfile[layer_name + "_b"]
            agent.actor.policy.get_layer(layer_name).set_weights([weights, biases])

        return agent
