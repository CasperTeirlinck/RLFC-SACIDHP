import gym
import numpy as np


class ReplayBuffer:
    """
    Replay buffer to make samples more i.i.d. during critic and actor update.
    """

    def __init__(self, config, env: gym.Env):
        self.size = config["buffer_size"]
        self.s_dim = config["s_dim"]
        self.idx = 0
        self.s_storage = np.zeros((self.size, self.s_dim), dtype=np.float32)
        self.action_storage = np.zeros((self.size, env.action_space.shape[0]), dtype=np.float32)
        self.reward_storage = np.zeros(self.size, dtype=np.float32)
        self.s_next_storage = np.zeros((self.size, self.s_dim), dtype=np.float32)
        self.done_storage = np.zeros(self.size, dtype=bool)

    def store(self, s, action, reward, s_next, done):
        """
        Store new experience in the buffer
        """

        idx = self.idx % self.size  # override old storage when limit is reached

        self.s_storage[idx] = s
        self.action_storage[idx] = action
        self.reward_storage[idx] = reward
        self.s_next_storage[idx] = s_next
        self.done_storage[idx] = done

        self.idx += 1

    def sample(self, batch_size):
        """
        Sample random experience from the buffer
        """

        idx_max = min(self.idx, self.size)  # prevent sampling from empty storage

        batch = np.random.choice(idx_max, batch_size)

        s_batch = self.s_storage[batch]
        action_batch = self.action_storage[batch]
        reward_batch = self.reward_storage[batch]
        done_batch = self.done_storage[batch]
        s_next_batch = self.s_next_storage[batch]

        return s_batch, action_batch, reward_batch, s_next_batch, done_batch

    def can_sample(self, batch_size):
        """
        Check if batch_size samples can be sampled from the buffer.
        """

        return self.idx >= batch_size
