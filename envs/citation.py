import copy
import importlib
import numpy as np
import tensorflow as tf
import os
import glob
import gym
from gym.spaces import Box
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from tools import set_random_seed, clip
from tools.utils import d2r, r2d


class Citation(gym.Env):
    """
    Cessna Citation
    Implements the CitAST Matlab/Simulink environment build with the DASMAT model by the Delft University of Technology
    """

    def __init__(self, config, dt, obs_extra=[]):
        super().__init__()
        self.config = config
        self.seed(config["seed"])

        # Labels
        self.state_descr = np.array(
            [
                "p",  # [rad/s]
                "q",  # [rad/s]
                "r",  # [rad/s]
                "V",  # [m/s]
                "alpha",  # [rad]
                "beta",  # [rad]
                "phi",  # [rad]
                "theta",  # [rad]
                "psi",  # [rad]
                "h",  # [m]
                "x",  # [m]
                "y",  # [m]
            ]
        )
        self.obs_descr = np.array(
            [
                "p",  # [rad/s]
                "q",  # [rad/s]
                "r",  # [rad/s]
                "alpha",  # [rad]
                "theta",  # [rad]
                "phi",  # [rad]
                "beta",  # [rad]
                # "h",  # [m]
            ]
        )
        self.action_descr = np.array(
            [
                "de",  # [rad]
                "da",  # [rad]
                "dr",  # [rad]
            ]
        )
        self.state_labels = np.array(
            [
                r"p",
                r"q",
                r"r",
                r"V",
                r"\alpha",
                r"\beta",
                r"\phi",
                r"\theta",
                r"\psi",
                r"h",
                r"x",
                r"y",
            ]
        )
        self.action_labels = np.array(
            [
                r"\delta_e",
                r"\delta_a",
                r"\delta_r",
            ]
        )
        self.state_units = np.array(
            [
                "[rad/s]",
                "[rad/s]",
                "[rad/s]",
                "[m/s]",
                "[rad]",
                "[rad]",
                "[rad]",
                "[rad]",
                "[rad]",
                "[m]",
                "[m]",
                "[m]",
            ]
        )
        self.action_units = np.array(
            [
                "[rad]",
                "[rad]",
                "[rad]",
            ]
        )

        # Action space
        self.action_space = Box(
            low=d2r(np.array([-20.05, -37.24, -21.77])),
            high=d2r(np.array([14.90, 37.24, 21.77])),
            dtype=np.float64,
        )
        self.action_space_rates = Box(
            low=d2r(np.array([-20, -40, -20])),
            high=d2r(np.array([20, 40, 20])),
            dtype=np.float64,
        )
        self.action_to_cmd = lambda action: np.array([*action, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # State to obs
        self.state_obs_idxs = [0, 1, 2, 4, 7, 6, 5]
        self.state_to_obs = lambda state: np.array(state)[self.state_obs_idxs]

        # Observation space
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float64,
        )
        self.obs_norm = [  # (max. acceptable magnitude)
            d2r(10.0),
            d2r(10.0),
            d2r(10.0),
            d2r(45.0),
            d2r(45.0),
            d2r(45.0),
            d2r(45.0),
        ]
        self.obs_labels = self.state_to_obs(self.state_labels)
        self.obs_extra = obs_extra

        # Decoupled lon/lat
        self.obs_lon_idxs = [1, 3, 4]
        self.obs_lat_idxs = [0, 2, 6, 5]
        self.obs_to_lon = lambda obs: obs[self.obs_lon_idxs]
        self.obs_to_lat = lambda obs: obs[self.obs_lat_idxs]
        self.obs_descr_lon = self.obs_to_lon(self.obs_descr)
        self.obs_descr_lat = self.obs_to_lat(self.obs_descr)

        self.action_lon_idxs = [0]
        self.action_lat_idxs = [1, 2]
        self.action_to_lon = lambda action: action[self.action_lon_idxs]
        self.action_to_lat = lambda action: action[self.action_lat_idxs]

        self.env_lon = type(
            "CitationLon",
            (gym.Env,),
            {
                "obs_descr": self.obs_to_lon(self.obs_descr),
                "obs_labels": self.obs_to_lon(self.obs_labels),
                "action_descr": self.action_to_lon(self.action_descr),
                "action_labels": self.action_to_lon(self.action_labels),
                "action_space": Box(
                    low=self.action_to_lon(self.action_space.low),
                    high=self.action_to_lon(self.action_space.high),
                    dtype=self.action_space.dtype,
                ),
                "action_space_rates": Box(
                    low=self.action_to_lon(self.action_space_rates.low),
                    high=self.action_to_lon(self.action_space_rates.high),
                    dtype=self.action_space_rates.dtype,
                ),
                "observation_space": Box(
                    low=self.obs_to_lon(self.observation_space.low),
                    high=self.obs_to_lon(self.observation_space.high),
                    dtype=self.observation_space.dtype,
                ),
                "obs_to": self.obs_to_lon,
            },
        )
        self.env_lat = type(
            "CitationLat",
            (gym.Env,),
            {
                "obs_descr": self.obs_to_lat(self.obs_descr),
                "obs_labels": self.obs_to_lat(self.obs_labels),
                "action_descr": self.action_to_lat(self.action_descr),
                "action_labels": self.action_to_lat(self.action_labels),
                "action_space": Box(
                    low=self.action_to_lat(self.action_space.low),
                    high=self.action_to_lat(self.action_space.high),
                    dtype=self.action_space.dtype,
                ),
                "action_space_rates": Box(
                    low=self.action_to_lat(self.action_space_rates.low),
                    high=self.action_to_lat(self.action_space_rates.high),
                    dtype=self.action_space_rates.dtype,
                ),
                "observation_space": Box(
                    low=self.obs_to_lat(self.observation_space.low),
                    high=self.obs_to_lat(self.observation_space.high),
                    dtype=self.observation_space.dtype,
                ),
                "obs_to": self.obs_to_lat,
            },
        )

        # Plant model: CitAST
        h0 = config["h0"]
        v0 = config["v0"]
        try:
            self.plant = importlib.import_module(f"envs.citast.h{h0}v{v0}_failures.citation", package=None)
        except:
            raise Exception(f"Citation with initial altitude {h0} and airspeed {v0} not implemented")

        # Failures
        self.failure = config["failure"]
        self.failure_time = config["failure_time"]  # [s]

        # Noise, Disturbances
        self.sensor_noise = config["sensor_noise"] if "sensor_noise" in config.keys() else False
        self.atm_disturbance = config["atm_disturbance"] if "atm_disturbance" in config.keys() else False
        self.control_disturbance = config["control_disturbance"] if "control_disturbance" in config.keys() else None

        # Inputs, outputs
        self.state = None
        self.obs = None
        self.action = None
        self.trimmed = config["trimmed"]

        # Running
        self.dt = dt
        self.t = 0

        # Logging
        self.state_history = []
        self.action_history = []

    def __str__(self):
        return "citation"

    def step(self, action):
        """
        Excecute a single time step
        """

        # Get action
        if tf.is_tensor(action):
            action = action.numpy()
        self.action = copy.deepcopy(action)

        # Failures
        cg = 0.0  # c.g. shift x [m]
        ht = 0.0  # horizontal tail reduction [%]
        icing = 0  # icing [1, 0]

        if self.t * self.dt >= self.failure_time:
            # Jammed rudder
            if self.failure == "dr_stuck":
                self.action[2] = d2r(-15.0)
            # Reduced aileron effectiveness
            elif self.failure == "da_reduce":
                self.action[1] = 0.1 * self.action[1]
            # Reduced aileron range
            elif self.failure == "da_limit":
                self.action[1] = np.clip(self.action[1], d2r(-5.0), d2r(5.0))
            # Reduced elevator effectiveness
            elif self.failure == "de_reduce":
                self.action[0] = 0.3 * self.action[0]
            elif self.failure == "de_reduce_extreme":
                self.action[0] = 0.1 * self.action[0]
            # Reduced elevator range
            elif self.failure == "de_limit":
                self.action[0] = np.clip(self.action[0], d2r(-2.5), d2r(2.5))
            # C.G shift
            elif self.failure == "cg_shift":
                cg = 0.25
            # Reduced horizontal tail
            elif self.failure == "ht_reduce":
                ht = 0.70
            # Icing
            elif self.failure == "icing":
                icing = 1
            # Icing
            elif self.failure == "de_invert":
                self.action[0] = -1.0 * self.action[0]

        # Control disturbance
        if self.control_disturbance is not None:
            self.action += self.control_disturbance[:, self.t]

        # Get new state
        self.state, _ = self.plant.step(self.action_to_cmd(self.action), [cg, ht, icing])
        self.t += 1

        # Sensor noise
        self.state += self.get_sensor_noise()

        # Atmospheric disturbance (alpha step disturbance)
        if self.atm_disturbance:
            t = 20
            if (self.t * self.dt >= t) and (self.t * self.dt <= (t + 3 / 2)):
                self.state[4] += d2r(2.5)
            if (self.t * self.dt >= (t + 3 / 2)) and (self.t * self.dt <= (t + 3)):
                self.state[4] -= d2r(2.5)

            t = 80
            if (self.t * self.dt >= t) and (self.t * self.dt <= (t + 3 / 2)):
                self.state[4] += d2r(2.5)
            if (self.t * self.dt >= (t + 3 / 2)) and (self.t * self.dt <= (t + 3)):
                self.state[4] -= d2r(2.5)

        # Get observed state
        self.obs = self.state_to_obs(self.state)

        # Logging
        self.state_history.append(self.state)
        self.action_history.append(action)

        if len(self.obs_extra) > 0:
            return np.float32(self.obs), self.state[self.obs_extra]
        else:
            return np.float32(self.obs)

    def get_sensor_noise(self):
        """
        Generate normal sensor noise based on values from https://doi.org/10.2514/6.2018-0385
        """

        sensor_noise = np.zeros(self.state_descr.shape[0])
        if self.sensor_noise:
            # p, q, r
            sensor_noise[0:3] += np.random.normal(loc=3.0e-5, scale=np.sqrt(4.0e-7), size=3)

            # sideslip (estimate)
            sensor_noise[5] += np.random.normal(loc=1.8e-3, scale=np.sqrt(7.5e-8), size=1)

            # phi, theta
            sensor_noise[6:8] += np.random.normal(loc=4.0e-3, scale=np.sqrt(1e-9), size=2)

            # h (estimate)
            sensor_noise[9] += np.random.normal(loc=8.0e-3, scale=np.sqrt(4.5e-3), size=1)

        return sensor_noise

    def reset(self):
        """
        Reset the environment to initial state
        """

        self.plant.initialize()
        state, state_trim = self.plant.step(self.action_to_cmd(np.zeros(self.action_space.shape)), [0, 0, 0])
        self.state = state
        # self.state = np.zeros(len(self.state_descr))
        self.action_trim = state_trim[0:3]
        self.action = self.action_trim if self.trimmed else np.zeros(self.action_space.shape)
        self.obs = self.state_to_obs(self.state)

        self.state_history = []
        self.action_history = []

        if len(self.obs_extra) > 0:
            return np.float32(self.obs), self.state[self.obs_extra]
        else:
            return np.float32(self.obs)

    def close(self):
        """
        Close the environment
        """

        self.plant.terminate()

        return

    def render_batch(self, task, save_dirs):
        """
        Render a batch of state-action history
        """

        # Load batch data
        state_history = []
        action_history = []

        for save_dir in save_dirs:
            npzfile = np.load(os.path.join(save_dir, "stateaction_history.npz"))
            state_history.append(npzfile["state_history"])
            action_history.append(npzfile["action_history"])

        state_history = np.array(state_history)
        action_history = np.array(action_history)

        # Filter on temporal loss
        idxs_failed = []
        for i, save_dir in enumerate(save_dirs):
            action_loss_t = 0
            data = np.array(action_history[i])
            for j in range(1, data.shape[0]):
                action_loss_t += np.sum((data[j] - data[j - 1]) ** 2)
            if action_loss_t > 0.01:
                idxs_failed.append(i)

        state_history = np.delete(state_history, idxs_failed, axis=0)
        action_history = np.delete(action_history, idxs_failed, axis=0)
        success_rate = state_history.shape[0] / len(save_dirs)
        print(f"success rate = {success_rate*100 :.2f}%")

        # * Fig: states and actions
        fig = plt.figure()
        fig.set_figwidth(10)  # 8.27
        ncols = 2
        nrows = 7
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=[1] * (nrows - 1) + [0.2])
        fig.set_figheight(1.33 * (nrows - 1))

        # Plot: states with reference
        def plot_states(idxs, col):
            line_state, line_ref, line_ref_internal = None, None, None

            for i, j in enumerate(idxs):
                state_descr = self.state_descr[j]
                state_label = self.state_labels[j]
                state_unit = self.state_units[j]
                ax = fig.add_subplot(gs[i, col])

                # state_history_max = np.max(state_history[:, :, j], axis=0)
                # state_history_min = np.min(state_history[:, :, j], axis=0)
                state_history_max, state_history_min = np.percentile(state_history[:, :, j], [75, 25], axis=0)
                state_history_mean = np.mean(state_history[:, :, j], axis=0)

                if state_unit == "[rad]":
                    y_min = r2d(state_history_min)
                    y_max = r2d(state_history_max)
                    y_mean = r2d(state_history_mean)
                    state_unit = "[deg]"
                elif state_unit == "[rad/s]":
                    y_min = r2d(state_history_min)
                    y_max = r2d(state_history_max)
                    y_mean = r2d(state_history_mean)
                    state_unit = "[deg/s]"
                else:
                    y_min = state_history_min
                    y_max = state_history_max
                    y_mean = state_history_mean
                ax.set_ylabel(rf"${state_label}$ {state_unit}")

                # Reference
                if state_descr in task.tracking_ref:
                    if state_unit in ["[deg]", "[deg/s]"]:
                        yr = r2d(task.tracking_ref[state_descr])
                    else:
                        yr = task.tracking_ref[state_descr]
                    (line_ref,) = ax.plot(task.timevec, yr, linestyle="--", color="k", label=rf"${state_label}^{{r}}$")

                # State
                (line_state,) = ax.plot(task.timevec, y_mean, label=rf"${state_label}$", color="C0", linestyle=":")
                (line_state_minmax,) = ax.plot(task.timevec, y_min, color="C0", linewidth=0.5)
                (line_state_minmax,) = ax.plot(task.timevec, y_max, color="C0", linewidth=0.5)
                ax.fill_between(task.timevec, y_max, y_min, color="C0", alpha=0.2)

                # Failure time
                if self.failure:
                    ax.axvline(x=self.failure_time, color="C7", linestyle="--")

            return line_state, line_state_minmax, line_ref, line_ref_internal

        state_lon_idxs = [1, 4, 7, 3, 9]
        state_lat_idxs = [0, 2, 6, 5]
        line_state, line_state_minmax, line_ref, line_ref_internal = plot_states(state_lon_idxs, 0)
        line_state, line_state_minmax, line_ref, _ = plot_states(state_lat_idxs, 1)

        # Plot: actions
        def plot_actions(idxs, col, row):
            line_action = None

            for i, j in enumerate(idxs):
                action_label = self.action_labels[j]
                action_unit = self.action_units[j]
                ax = fig.add_subplot(gs[row + i, col])

                action_history_max = np.max(action_history[:, :, j], axis=0)
                action_history_min = np.min(action_history[:, :, j], axis=0)
                action_history_mean = np.mean(action_history[:, :, j], axis=0)

                if action_unit == "[rad]":
                    y_min = r2d(action_history_min)
                    y_max = r2d(action_history_max)
                    y_mean = r2d(action_history_mean)
                    action_unit = "[deg]"
                else:
                    y_min = action_history_min
                    y_max = action_history_max
                    y_mean = action_history_mean
                ax.set_ylabel(rf"${action_label}$ {action_unit}")
                if i == len(idxs) - 1:
                    ax.set_xlabel(r"$Time$ [s]")

                (line_action,) = ax.plot(task.timevec, y_mean, label=rf"${action_label}$", color="C2", linestyle=":")
                (line_action_minmax,) = ax.plot(task.timevec, y_min, color="C2", linewidth=0.5)
                (line_action_minmax,) = ax.plot(task.timevec, y_max, color="C2", linewidth=0.5)
                ax.fill_between(task.timevec, y_max, y_min, color="C2", alpha=0.2)

                # Failure time
                if self.failure:
                    ax.axvline(x=self.failure_time, color="C7", linestyle="--")

            return line_action, line_action_minmax

        action_lon_idxs = [0]
        action_lat_idxs = [1, 2]
        line_action, line_action_minmax = plot_actions(action_lon_idxs, 0, len(state_lon_idxs))
        line_action, line_action_minmax = plot_actions(action_lat_idxs, 1, len(state_lat_idxs))

        # Legend
        handles = [line_state, line_state_minmax, line_ref, line_ref_internal, line_action, line_action_minmax]
        labels = ["state mean", "state Q1/Q3", "reference", "reference (internal)", "action", "action Q1/Q3"]
        labels = [_ for i, _ in enumerate(labels) if handles[i] is not None]
        handles = [_ for _ in handles if _ is not None]

        lax = fig.add_subplot(gs[-1, :])
        lax.legend(
            handles=handles,
            labels=labels,
            borderaxespad=0,
            mode="expand",
            ncol=6,
        )
        lax.axis("off")

        plt.show(block=False)

    def render(
        self,
        task,
        agent=None,
        env_sac=None,
        tracking_ref_internal=None,
        idx_end=None,
        show_rmse=False,
        state_simulink=None,
        rewards=None,
        lon_only=False,
        lat_only=False,
    ):
        """
        Visualize environment response
        """

        # Load data
        state_history = np.array(self.state_history).T
        action_history = np.array(self.action_history).T
        if env_sac:
            state_history_sac = np.array(env_sac.state_history).T
            action_history_sac = np.array(env_sac.action_history).T
        state_history = np.array(self.state_history).T
        action_history = np.array(self.action_history).T
        if state_simulink is not None:
            state_simulink = state_simulink[1 : task.num_timesteps + 1].T
        if rewards is not None:
            rewards = np.array(rewards)

        # * Fig: states and actions
        fig = plt.figure()
        fig.set_figwidth(10)  # 8.27
        lon_lat = bool(not lon_only and not lat_only)
        if lon_only or lat_only:
            ncols = 1
        else:
            ncols = 2
        if show_rmse and lon_only:
            nrows = 8
        elif show_rmse and (lat_only or lon_lat):
            nrows = 9
        else:
            nrows = 7
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=[1] * (nrows - 1) + [0.2])
        fig.set_figheight(1.33 * (nrows - 1))

        # Plot: states with reference
        def plot_states(idxs, col):
            line_state, line_ref, line_ref_internal, line_simulink = None, None, None, None

            for i, j in enumerate(idxs):
                state_descr = self.state_descr[j]
                state_label = self.state_labels[j]
                state_unit = self.state_units[j]
                ax = fig.add_subplot(gs[i, col])

                if state_unit == "[rad]":
                    y = r2d(state_history[j])[:idx_end]
                    if env_sac:
                        y_sac = r2d(state_history_sac[j])[:idx_end]
                    state_unit = "[deg]"
                elif state_unit == "[rad/s]":
                    y = r2d(state_history[j])[:idx_end]
                    if env_sac:
                        y_sac = r2d(state_history_sac[j])[:idx_end]
                    state_unit = "[deg/s]"
                else:
                    y = state_history[j][:idx_end]
                    if env_sac:
                        y_sac = state_history_sac[j][:idx_end]
                ax.set_ylabel(rf"${state_label}$ {state_unit}")

                # Reference
                if state_descr in task.tracking_ref:
                    if state_unit in ["[deg]", "[deg/s]"]:
                        yr = r2d(task.tracking_ref[state_descr])[:idx_end]
                    else:
                        yr = task.tracking_ref[state_descr][:idx_end]
                    (line_ref,) = ax.plot(
                        task.timevec[:idx_end], yr, linestyle="--", color="k", label=rf"${state_label}^{{r}}$"
                    )

                    # ax.legend(bbox_to_anchor=(1, 1), loc=1)
                elif tracking_ref_internal and state_descr in tracking_ref_internal:
                    if state_unit in ["[deg]", "[deg/s]"]:
                        yr = r2d(np.array(tracking_ref_internal[state_descr]))[:idx_end]
                    else:
                        yr = tracking_ref_internal[state_descr][:idx_end]
                    (line_ref_internal,) = ax.plot(
                        task.timevec[:idx_end], yr, linestyle="-.", color="k", label=rf"${state_label}^{{r}}$"
                    )
                    # ax.legend(bbox_to_anchor=(1, 1), loc=1)

                # State
                line_state_sac = None
                if env_sac:
                    (line_state_sac,) = ax.plot(
                        task.timevec[:idx_end], y_sac, label=rf"${state_label}$", color="#C1C1C1", linewidth=2
                    )
                (line_state,) = ax.plot(task.timevec[:idx_end], y, label=rf"${state_label}$", color="C0")

                # Failure time
                if self.failure:
                    ax.axvline(x=self.failure_time, color="C7", linestyle="--")

                if state_simulink is not None:
                    if state_unit in ["[deg]", "[deg/s]"]:
                        ys = r2d(state_simulink[j])[:idx_end]
                    else:
                        ys = state_simulink[j][:idx_end]
                    (line_simulink,) = ax.plot(
                        task.timevec[:idx_end],
                        ys,
                        linestyle=(0, (2, 2)),
                        # linestyle="--",
                        linewidth=1.5,
                        color="C3",
                    )

            return line_state, line_ref, line_ref_internal, line_state_sac, line_simulink

        state_lon_idxs = [1, 4, 7, 3, 9]
        state_lat_idxs = [0, 2, 6, 5]
        if lon_only or lon_lat:
            line_state, line_ref, line_ref_internal, line_state_sac, line_simulink = plot_states(state_lon_idxs, 0)
        if lat_only:
            line_state, line_ref, _, _, _ = plot_states(state_lat_idxs, 0)
            line_ref_internal = None
        if lon_lat:
            line_state, line_ref, _, _, _ = plot_states(state_lat_idxs, 1)

        # Plot: actions
        def plot_actions(idxs, col, row):
            line_action = None
            line_action_disturbance = None

            for i, j in enumerate(idxs):
                action_label = self.action_labels[j]
                action_unit = self.action_units[j]
                ax = fig.add_subplot(gs[row + i, col])

                if action_unit == "[rad]":
                    y = r2d(action_history[j])[:idx_end]
                    if env_sac:
                        y_sac = r2d(action_history_sac[j])[:idx_end]
                    action_unit = "[deg]"
                else:
                    y = action_history[j][:idx_end]
                    if env_sac:
                        y_sac = action_history_sac[j][:idx_end]
                ax.set_ylabel(rf"${action_label}$ {action_unit}")
                if i == len(idxs) - 1:
                    ax.set_xlabel(r"$Time$ [s]")

                if env_sac:
                    (line_action_sac,) = ax.plot(
                        task.timevec[:idx_end], y_sac, label=rf"${action_label}$", color="#C1C1C1", linewidth=2
                    )
                (line_action,) = ax.plot(task.timevec[:idx_end], y, label=rf"${action_label}$", color="C2")

                if self.control_disturbance is not None:
                    (line_action_disturbance,) = ax.plot(
                        task.timevec[:idx_end], r2d(self.control_disturbance[j, :]), color="k"
                    )

                # Failure time
                if self.failure:
                    ax.axvline(x=self.failure_time, color="C7", linestyle="--")

                #
                # if action_label == r"\delta_a":
                #     axins = zoomed_inset_axes(ax, zoom=6, loc=10)
                #     axins.set_xlim(4.5, 6)
                #     axins.set_ylim(-1.25, -0.75)
                #     axins.set_yticklabels([])
                #     axins.set_xticklabels([])
                #     mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                #     axins.plot(task.timevec[:idx_end], y, color="C2")

            return line_action, line_action_disturbance

        action_lon_idxs = [0]
        action_lat_idxs = [1, 2]
        if lon_only or lon_lat:
            line_action, line_action_disturbance = plot_actions(action_lon_idxs, 0, len(state_lon_idxs))
        if lat_only:
            line_action, line_action_disturbance = plot_actions(action_lat_idxs, 0, len(state_lat_idxs))
        if lon_lat:
            line_action, line_action_disturbance = plot_actions(action_lat_idxs, 1, len(state_lat_idxs))

        # Plot: RMSE
        # line_thresh, line_warmup = None, None
        # if show_rmse:
        #     l = 0
        #     k = 0
        #     for i, tracking_descr in enumerate(agent.tracking_descr_external):
        #         state_label = self.state_labels[np.where(self.state_descr == tracking_descr)][0]
        #         state_unit = self.state_units[np.where(self.state_descr == tracking_descr)][0]

        #         if tracking_descr in self.obs_descr_lon:
        #             row = len(state_lon_idxs) + len(action_lon_idxs) + l
        #             l += 1
        #             col = 0
        #         if tracking_descr in self.obs_descr_lat:
        #             row = len(state_lat_idxs) + len(action_lat_idxs) + k
        #             col = 1 if lon_lat else 0
        #             k += 1

        #         rmse = np.array(agent.tracking_rmse_history)[:, i]
        #         thresh = r2d(np.array(agent.tracking_thresh)[i])
        #         if state_unit == "[rad]":
        #             rmse = r2d(rmse)
        #             state_unit = "[deg]"

        #         ax = fig.add_subplot(gs[row, col])
        #         ax.set_ylabel(rf"${state_label}^{{RMSE}}$ {state_unit}")

        #         rmse_top = np.copy(rmse)
        #         rmse_top[rmse_top <= thresh] = np.nan
        #         rmse_top[: agent.lr_warmup] = np.nan
        #         rmse_bottom = np.copy(rmse)
        #         rmse_bottom[rmse_bottom > thresh] = np.nan
        #         rmse_bottom[: agent.lr_warmup] = np.nan
        #         rmse_left = np.copy(rmse)
        #         rmse_left[agent.lr_warmup :] = np.nan
        #         ax.plot(task.timevec[:idx_end], rmse_bottom[:idx_end], color="C0")
        #         ax.plot(task.timevec[:idx_end], rmse_top[:idx_end], color="C0")
        #         ax.plot(task.timevec[:idx_end], rmse_left[:idx_end], color="#C1C1C1")

        #         # # fill above thresh
        #         # ax.fill_between(
        #         #     task.timevec,
        #         #     rmse_top,
        #         #     thresh * np.ones_like(task.timevec),
        #         #     color="C0",
        #         #     alpha=0.3,
        #         # )
        #         # # fill the delay below thresh
        #         # thresh_lr_high = thresh * np.ones_like(task.timevec)
        #         # thresh_lr_high[np.array(agent.tracking_rmse_delay_history)[:, i]] = np.nan
        #         # ax.fill_between(
        #         #     task.timevec,
        #         #     thresh_lr_high,
        #         #     rmse_bottom,
        #         #     color="C0",
        #         #     alpha=0.3,
        #         # )

        #         line_thresh = ax.hlines(
        #             y=thresh,
        #             xmin=task.timevec[agent.lr_warmup],
        #             xmax=task.timevec[-1],
        #             color="k",
        #             label=rf"${state_label}_{{thresh}}$",
        #             linestyle=":",
        #         )
        #         line_warmup = ax.axvline(
        #             x=task.timevec[agent.lr_warmup],
        #             color="k",
        #             linestyle="-",
        #             label=r"$t_{warmup}$",
        #         )

        # Legend
        handles = [
            line_state,
            line_simulink,
            line_ref,
            line_ref_internal,
            line_action,
            line_action_disturbance,
            line_state_sac,
        ]
        labels = [
            "state",
            "state Simulink",
            "reference",
            "reference (internal)",
            "action",
            "control disturbance",
            "state/action SAC-only",
        ]
        labels = [_ for i, _ in enumerate(labels) if handles[i] is not None]
        handles = [_ for _ in handles if _ is not None]

        lax = fig.add_subplot(gs[-1, :])
        lax.legend(
            handles=handles,
            labels=labels,
            borderaxespad=0,
            mode="expand",
            ncol=6,
        )
        lax.axis("off")

        # * Fig: reward
        if rewards is not None:
            fig = plt.figure()
            fig.set_figwidth(8.27)
            gs = fig.add_gridspec(nrows=1, ncols=1)
            fig.set_figheight(4)

            ax = fig.add_subplot(gs[0, 0])
            ax.plot(task.timevec[:idx_end], rewards[:idx_end])

        plt.show(block=False)
