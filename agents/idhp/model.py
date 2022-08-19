import os
import numpy as np
import gym


class Model:
    """
    Recursive Least Squares (RLS) incremental environment model
    """

    def __init__(self, config, env: gym.Env):
        self.config = config["model"]

        # Dimensions
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        # Config
        self.gamma = config["model"]["gamma"]

        # Initialize measurement matrix
        self.X = np.ones((self.state_size + self.action_size, 1))

        # Initialise parameter matrix
        self.Theta = np.zeros((self.state_size + self.action_size, self.state_size))

        # Initialize covariance matrix
        self.Cov0 = config["model"]["cov0"] * np.identity(self.state_size + self.action_size)
        self.Cov = self.Cov0

        # Initial innovation (prediction error)
        self.epsilon = np.zeros((1, self.state_size))
        self.Cov_reset = False
        self.epsilon_thresh = np.array(config["model"]["eps_thresh"])

        self.t = 0

    @property
    def F(self):
        return np.float32(self.Theta[: self.state_size, :].T)

    @property
    def G(self):
        return np.float32(self.Theta[self.state_size :, :].T)

    def update(self, state, action, state_next):
        """
        Update RLS parameters based on state-action sample
        """

        # Predict next state
        state_next_pred = self.predict(state, action)

        # Error
        self.epsilon = (np.array(state_next)[np.newaxis].T - state_next_pred).T

        # Intermediate computations
        CovX = self.Cov @ self.X
        XCov = self.X.T @ self.Cov
        gammaXCovX = self.gamma + XCov @ self.X

        # Update parameter matrix
        self.Theta = self.Theta + (CovX @ self.epsilon) / gammaXCovX

        # Update covariance matrix
        self.Cov = (self.Cov - (CovX @ XCov) / gammaXCovX) / self.gamma

        # Check if Cov needs reset
        if self.Cov_reset == False:
            if np.sum(np.greater(np.abs(self.epsilon), self.epsilon_thresh)) == 1 and self.t > 1000:
                self.Cov_reset = True
                self.Cov = self.Cov0
        elif self.Cov_reset == True:
            if np.sum(np.greater(np.abs(self.epsilon), self.epsilon_thresh)) == 0:
                self.Cov_reset = False

        self.t += 1

    def predict(self, state, action):
        """
        Predict next state based on RLS model
        """

        # Set measurement matrix
        self.X[: self.state_size] = np.array(state)[np.newaxis].T
        self.X[self.state_size :] = np.array(action)[np.newaxis].T

        # Predict next state
        state_next_pred = (self.X.T @ self.Theta).T

        return state_next_pred

    def save_weights(self, save_dir):
        """
        Save current weights
        """

        np.savez(
            save_dir + ".npz",
            x=self.X,
            theta=self.Theta,
            cov=self.Cov,
            epsilon=self.epsilon,
        )

    def load_weights(self, save_dir):
        """
        Load weights
        """

        # Weights npz
        npzfile = np.load(save_dir + ".npz")

        # Load weights
        self.X = npzfile["x"]
        self.Theta = npzfile["theta"]
        self.Cov = npzfile["cov"]
        self.epsilon = npzfile["epsilon"]
