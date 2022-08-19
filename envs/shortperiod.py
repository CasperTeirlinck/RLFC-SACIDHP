import numpy as np
import tensorflow as tf
import os
import glob
import gym
from gym.spaces import Box
import matplotlib.pyplot as plt

from tasks import TrackingTask
from tools import set_random_seed, clip
from tools.utils import d2r, r2d


class ShortPeriod(gym.Env):
    """
    Linear short period model
    """

    def __init__(self, config, dt):
        super().__init__()
        self.seed(config["seed"])
        set_random_seed(config["seed"])

        # Simulation
        self.dt = dt
        self.t = 0

        # Labels
        self.state_descr = ["alpha", "q"]  # [rad], [rad/s]
        self.obs_descr = ["alpha", "q"]  # [rad], [rad/s]
        self.action_descr = ["de"]  # [rad]
        self.state_labels = [[r"\alpha", "[deg]"], [r"q", "[deg/s]"]]
        self.action_labels = [[r"\delta_e", "[deg]"]]

        # Action space
        self.action_space = Box(
            low=np.array([d2r(-20.0)]),
            high=np.array([d2r(20.0)]),
            dtype=np.float64,
        )
        self.action_space_rates = Box(
            low=d2r(np.array([-20])),
            high=d2r(np.array([20])),
            dtype=np.float64,
        )
        self.action_space.seed(config["seed"])

        # State space
        self.state = np.array([0.0, 0.0])
        self.trimmed = config["trimmed"]

        # Observation space
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float64,
        )
        self.obs_norm = [  # (max. acceptable magnitude)
            d2r(45.0),
            d2r(10.0),
        ]
        self.obs_scale = np.array([1.0, 1.0])

        # Inputs, outputs
        self.obs = None
        self.action = None

        # Faults
        self.fault = False
        self.fault_type = config["fault"]
        self.fault_timestep = config["fault_timestep"]

        # State transition model
        if config["dynamics"] == "ce500":  # [AE3202 Flight Dynamics Lecture Notes Table D-1]
            self.V = 59.9  # [m/s]
            self.c = 2.022  # [m]
            self.u_c = 102.7
            self.K2_Y = 0.98

            self.C_Za = -5.16
            self.C_Zadot = -1.43
            self.C_Zq = -3.86
            self.C_Zde = -0.6238
            self.C_ma = -0.43
            self.C_madot = -3.7
            self.C_mq = -7.04
            self.C_mde = -1.553

        elif config["dynamics"] == "ce172":  # [AE3202 Flight Dynamics Lecture Notes Table D-3]
            self.V = 66.75  # [m/s]
            self.c = 1.494  # [m]
            self.u_c = 47.05
            self.K2_Y = 0.6814

            self.C_Za = -4.631
            self.C_Zadot = -0.85
            self.C_Zq = -1.95
            self.C_Zde = -0.43
            self.C_ma = -0.89
            self.C_madot = -2.6
            self.C_mq = -6.2
            self.C_mde = -1.28

        else:
            raise Exception(f"Dynamics for '{config['dynamics']}' not found")

        self.A = self.get_A()
        self.B = self.get_B()

        # Logging
        self.state_history = []
        self.action_history = []

    def __str__(self):
        return "shortperiod"

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """

        self.t = 0

        # Trimmed/Untrimmed state
        if self.trimmed:
            self.state = np.array([0.0, 0.0])
        else:
            rng = np.random.default_rng()
            alpha_0 = d2r(rng.uniform(low=-5.0, high=5.0))
            q_0 = d2r(rng.uniform(low=-3.0, high=3.0))
            self.state = np.array([alpha_0, q_0])

        self.action = np.zeros(self.action_space.shape)

        self.state_history = []
        self.action_history = []

        # Return
        obs = self.get_obs()

        return np.float32(obs)

    def step(self, action=None):
        """
        Perform a single time step calculation
        """

        # System faults
        if self.fault_type == "de_reduce":
            # Reduce the elevator effectiveness by 50%
            if self.fault == False and self.t >= self.fault_timestep:
                self.C_mde = self.C_mde * 0.5
                self.C_Zde = self.C_Zde * 0.5
                self.A = self.get_A()
                self.B = self.get_B()
                self.fault = True

        elif self.fault_type == "de_invert":
            # Invert the elevator
            if self.fault == False and self.t >= self.fault_timestep:
                self.C_mde = -self.C_mde
                self.C_Zde = -self.C_Zde
                self.A = self.get_A()
                self.B = self.get_B()
                self.fault = True

        elif self.fault_type == "cg_shift":
            # Sudden shift in the c.g
            if self.fault == False and self.t >= self.fault_timestep:
                self.shift_cg(shift=1)
                self.fault = True

        self.action = action

        # Perform action and calculate new state by performing integration step
        self.state = self.state + (self.A.dot(self.state) + self.B.dot(action)) * self.dt
        self.t += 1

        # Get state observation
        obs = self.get_obs()

        # Logging
        self.state_history.append(self.state)
        self.action_history.append(action)

        return np.float32(obs)

    def get_obs(self):
        """
        Get the observed state vector
        """

        # Construct observed state vector
        obs = np.array([*self.state]) * self.obs_scale

        return obs

    def shift_cg(self, shift):
        """
        Shift the c.g with <shift> m (+ve backwards, -ve forwards)
        """

        # [AE3202 Flight Dynamics Lecture Notes Table 7-2]
        C_ma = self.C_ma - self.C_Za * shift / self.c
        C_Zq = self.C_Zq - self.C_Za * shift / self.c
        C_mq = self.C_mq - (self.C_Zq + self.C_ma) * shift / self.c + self.C_Za * (shift / self.c) ** 2
        C_madot = self.C_madot - self.C_Zadot * shift / self.c

        self.C_ma = C_ma
        self.C_Zq = C_Zq
        self.C_mq = C_mq
        self.C_madot = C_madot
        self.A = self.get_A()
        self.B = self.get_B()

    def get_A(self):
        """
        Construct the A-matrix
        [AE3202 Flight Dynamics Lecture Notes Eq. 4-48 and Table 4-9]
        """

        Vc = self.V / self.c
        u_cK2Y = self.u_c * self.K2_Y

        # fmt: off
        A = np.array([
            [
                Vc * self.C_Za/(2*self.u_c - self.C_Zadot),
                Vc * (2*self.u_c + self.C_Zq)/(2*self.u_c - self.C_Zadot)
            ],
            [
                Vc * (self.C_ma + self.C_Za * self.C_madot/(2*self.u_c - self.C_Zadot))/(2*u_cK2Y),
                Vc * (self.C_mq + self.C_madot * (2*self.u_c + self.C_Zq)/(2*self.u_c - self.C_Zadot))/(2*u_cK2Y)
            ]
        ]) 
        # fmt: on

        # Unnormalize the state qc/V
        A[1, :] *= Vc
        A[:, 1] /= Vc

        return A

    def get_B(self):
        """
        Construct the B-matrix
        [AE3202 Flight Dynamics Lecture Notes Eq. 4-48 and Table 4-9]
        """

        Vc = self.V / self.c
        u_cK2Y = self.u_c * self.K2_Y

        # fmt: off
        B = np.array([
            [Vc * self.C_Zde/(2*self.u_c - self.C_Zadot)],
            [Vc * (self.C_mde + self.C_Zde * self.C_madot/(2*self.u_c - self.C_Zadot))/(2*u_cK2Y)]
        ]) 
        # fmt: on

        # Unnormalize the state qc/V
        B[1, :] *= Vc

        return B

    def render(self, agent, task, idx_end=None, show_rmse=True, batch_dir=None):
        """
        Visualize environment response
        """

        # Load data
        state_history = []
        action_history = []
        rmse_history = []

        if batch_dir:
            for f in glob.glob(os.path.join(batch_dir, f"state_history_*.npy")):
                state_history.append(np.load(f))
            for f in glob.glob(os.path.join(batch_dir, f"action_history_*.npy")):
                action_history.append(np.load(f))
            for f in glob.glob(os.path.join(batch_dir, f"rmse_history_*.npy")):
                rmse_history.append(np.load(f))
            state_history = np.array(state_history)
            action_history = np.array(action_history)
            if show_rmse:
                rmse_history = np.array(rmse_history)
        else:
            state_history = np.array(self.state_history).T
            action_history = np.array(self.action_history).T
            if show_rmse:
                rmse_history = agent.tracking_rmse_history

        # * Fig: states and actions
        fig = plt.figure()
        fig.set_figwidth(8.27)
        if show_rmse:
            gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2, 1, 1])
            fig.set_figheight(5.5)
        else:
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1])
            fig.set_figheight(4)

        # Plot: state with ref
        ax = fig.add_subplot(gs[0, 0])
        if str(task) == "tracking_q":
            state_descr = "q"
            state_label = self.state_labels[1][0]
            state_unit = self.state_labels[1][1]
        elif str(task) == "tracking_alpha":
            state_descr = "alpha"
            state_label = self.state_labels[0][0]
            state_unit = self.state_labels[0][1]
        ax.set_ylabel(fr"${state_label}$ {state_unit}")

        if not batch_dir:
            ax.plot(
                task.timevec[:idx_end],
                r2d(state_history[self.state_descr.index(state_descr)])[:idx_end],
                label=fr"${state_label}$",
                color="C0",
            )
        else:
            state_history_max = r2d(np.max(state_history[:, self.state_descr.index(state_descr)], axis=0))
            state_history_min = r2d(np.min(state_history[:, self.state_descr.index(state_descr)], axis=0))
            state_history_mean = r2d(np.mean(state_history[:, self.state_descr.index(state_descr)], axis=0))

            ax.plot(task.timevec[:idx_end], state_history_max[:idx_end], color="C0")
            ax.plot(task.timevec[:idx_end], state_history_min[:idx_end], color="C0")
            ax.plot(
                task.timevec[:idx_end],
                state_history_mean[:idx_end],
                linestyle=":",
                label=fr"${state_label}$",
                color="C0",
            )
            ax.fill_between(
                task.timevec[:idx_end],
                state_history_max[:idx_end],
                state_history_min[:idx_end],
                color="C0",
                alpha=0.3,
            )
        ax.plot(
            task.timevec[:idx_end],
            r2d(task.tracking_ref[state_descr])[:idx_end],
            linestyle="--",
            color="k",
            label=fr"${state_label}_{{ref}}$",
        )
        if self.fault_type:
            ax.axvline(
                x=task.timevec[self.fault_timestep],
                color="k",
                linestyle="-.",
                label=r"$t_{fault}$",
            )
        ax.legend(bbox_to_anchor=(1, 1), loc=1)

        # Plot: action de
        ax = fig.add_subplot(gs[1, 0])
        ax.set_ylabel(r"$\delta_e$ [deg]")

        if not batch_dir:
            ax.plot(
                task.timevec[:idx_end],
                r2d(action_history[self.action_descr.index("de")])[:idx_end],
                color="C0",
            )
        else:
            action_history_max = r2d(
                np.max(action_history[:, self.action_descr.index("de")], axis=0),
            )
            action_history_min = r2d(
                np.min(action_history[:, self.action_descr.index("de")], axis=0),
            )
            action_history_mean = r2d(
                np.mean(action_history[:, self.action_descr.index("de")], axis=0),
            )

            ax.plot(task.timevec[:idx_end], action_history_max[:idx_end], color="C0")
            ax.plot(task.timevec[:idx_end], action_history_min[:idx_end], color="C0")
            ax.plot(
                task.timevec[:idx_end],
                action_history_mean[:idx_end],
                linestyle=":",
                label=r"$\delta_e$ [deg]",
                color="C0",
            )
            ax.fill_between(
                task.timevec[:idx_end],
                action_history_max[:idx_end],
                action_history_min[:idx_end],
                color="C0",
                alpha=0.3,
            )

        if self.fault_type:
            ax.axvline(
                x=task.timevec[self.fault_timestep],
                color="k",
                linestyle="-.",
            )

        # Plot: q rmse
        if show_rmse:
            ax = fig.add_subplot(gs[2, 0])
            ax.set_xlabel(r"time [s]")
            ax.set_ylabel(fr"${state_label}^{{RMSE}}$ {state_unit}")

            if not batch_dir:
                rmse = np.copy(rmse_history)
                rmse[rmse > agent.lr_thresh_rmse] = np.nan
                ax.plot(task.timevec[:idx_end], r2d(rmse)[:idx_end], color="C0")
                rmse = np.copy(rmse_history)
                rmse[rmse <= agent.lr_thresh_rmse] = np.nan
                ax.plot(task.timevec[:idx_end], r2d(rmse)[:idx_end], color="C3")
            else:
                rmse_history_max = np.max(rmse_history, axis=0)
                rmse_history_min = np.min(rmse_history, axis=0)
                rmse_history_mean = np.mean(rmse_history, axis=0)

                rmse_max = np.copy(rmse_history_max)
                rmse_max[rmse_max > agent.lr_thresh_rmse] = np.nan
                ax.plot(task.timevec[:idx_end], r2d(rmse_max)[:idx_end], color="C0")
                rmse_max_ = np.copy(rmse_history_max)
                rmse_max_[rmse_max_ <= agent.lr_thresh_rmse] = np.nan
                ax.plot(task.timevec[:idx_end], r2d(rmse_max_)[:idx_end], color="C3")

                rmse_min = np.copy(rmse_history_min)
                rmse_min[rmse_min > agent.lr_thresh_rmse] = np.nan
                ax.plot(task.timevec[:idx_end], r2d(rmse_min)[:idx_end], color="C0")
                rmse_min_ = np.copy(rmse_history_min)
                rmse_min_[rmse_min_ <= agent.lr_thresh_rmse] = np.nan
                ax.plot(task.timevec[:idx_end], r2d(rmse_min_)[:idx_end], color="C3")

                rmse_mean = np.copy(rmse_history_mean)
                rmse_mean[rmse_mean > agent.lr_thresh_rmse] = np.nan
                ax.plot(
                    task.timevec[:idx_end],
                    r2d(rmse_mean)[:idx_end],
                    color="C0",
                    linestyle=":",
                )
                rmse_mean_ = np.copy(rmse_history_mean)
                rmse_mean_[rmse_mean_ <= agent.lr_thresh_rmse] = np.nan
                ax.plot(
                    task.timevec[:idx_end],
                    r2d(rmse_mean_)[:idx_end],
                    color="C3",
                    linestyle=":",
                )

                ax.fill_between(
                    task.timevec[:idx_end],
                    r2d(rmse_history_max)[:idx_end],
                    r2d(rmse_history_min)[:idx_end],
                    color="C0",
                    alpha=0.3,
                )

            ax.axhline(
                y=r2d(agent.lr_thresh_rmse),
                color="C3",
                label=fr"${state_label}_{{thresh}}$",
                linestyle="--",
            )
            if agent.lr_warmup:
                ax.axvline(
                    x=task.timevec[agent.lr_warmup],
                    color="k",
                    linestyle="-",
                    label=r"$t_{warmup}$",
                )
            if self.fault_type:
                ax.axvline(
                    x=task.timevec[self.fault_timestep],
                    color="k",
                    linestyle="-.",
                )
            ax.legend(bbox_to_anchor=(1, 1), loc=1)

        plt.show(block=False)
