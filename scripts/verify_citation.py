import copy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from envs import Citation
from tools import set_plot_styles
from tasks import VerificationTask
from tools.utils import d2r

set_plot_styles()

CONFIG_ENV_CITATION = {
    "seed": None,
    "h0": 2000,  # initial trimmed altitude
    "v0": 90,  # initial trimmed airspeed
    "trimmed": False,  # trimmed initial action is known
    "failure": None,  # failure type
    "failure_time": 30,  # failure time [s]
    "sensor_noise": False,
}
CONFIG_TASK = {
    "T": 10,  # task duration
    "dt": 0.01,  # time-step
}

config_env = copy.deepcopy(CONFIG_ENV_CITATION)
config_task = copy.deepcopy(CONFIG_TASK)

#
def get_control_disturbance(timevec):
    """
    Get control disturbance signal as 3211 signal
    """

    disturbance = np.zeros((3, len(timevec)))
    # Elevator
    de_t0 = 1  # [s]
    de_T = 1  # [s]
    de_A = d2r(1)  # [rad]
    disturbance[0, np.argwhere(timevec == de_t0)[0, 0] : np.argwhere(timevec == de_t0 + 3 * de_T)[0, 0]] = 0.8 * de_A
    disturbance[0, np.argwhere(timevec == de_t0 + 3 * de_T)[0, 0] : np.argwhere(timevec == de_t0 + 5 * de_T)[0, 0]] = (
        -1.2 * de_A
    )
    disturbance[0, np.argwhere(timevec == de_t0 + 5 * de_T)[0, 0] : np.argwhere(timevec == de_t0 + 6 * de_T)[0, 0]] = (
        1.1 * de_A
    )
    disturbance[0, np.argwhere(timevec == de_t0 + 6 * de_T)[0, 0] : np.argwhere(timevec == de_t0 + 7 * de_T)[0, 0]] = (
        -1.1 * de_A
    )

    # Aileron
    da_t0 = 2  # [s]
    da_T = 1  # [s]
    da_A = d2r(1)  # [rad]
    disturbance[1, np.argwhere(timevec == da_t0)[0, 0] : np.argwhere(timevec == da_t0 + 3 * da_T)[0, 0]] = 0.8 * da_A
    disturbance[1, np.argwhere(timevec == da_t0 + 3 * da_T)[0, 0] : np.argwhere(timevec == da_t0 + 5 * da_T)[0, 0]] = (
        -1.2 * da_A
    )
    disturbance[1, np.argwhere(timevec == da_t0 + 5 * da_T)[0, 0] : np.argwhere(timevec == da_t0 + 6 * da_T)[0, 0]] = (
        1.1 * da_A
    )
    disturbance[1, np.argwhere(timevec == da_t0 + 6 * da_T)[0, 0] : np.argwhere(timevec == da_t0 + 7 * da_T)[0, 0]] = (
        -1.1 * da_A
    )

    return disturbance


# Env
env = Citation(config_env, dt=0.01)

# Task
task = VerificationTask(config_task)

# Control signal
action_step = get_control_disturbance(task.timevec)

# Run
obs = env.reset()
action_trim = env.action
for t in range(task.num_timesteps):
    action = action_step[:, t]
    obs = env.step(action)

# Compare with Simulink
state_simulink = loadmat("./envs/citast/verify_2000_90.mat")["state"]
state_env = np.array(env.state_history).T

# RMSE
y = state_env[:, :]
y_ref = state_simulink[1 : task.num_timesteps + 1, :].T
rmse_vec = np.sqrt(np.mean(np.square(y - y_ref), axis=1))
nrmse_vec = rmse_vec / (np.max(y_ref, axis=1) - np.min(y_ref, axis=1))
nrmse = np.mean(nrmse_vec)

print(f"nRMSE = {nrmse * 100 :.2f}%")

# Plot
env.render(task, state_simulink=state_simulink)

input()
