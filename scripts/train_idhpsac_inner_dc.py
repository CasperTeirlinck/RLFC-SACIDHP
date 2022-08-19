import json
import os
import random
import sys
import copy
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

from envs.citation import Citation
from tasks import TrackAttitudeLong, TrackAttitude
from tasks.tracking_attitude_lat import TrackAttitudeLat
from tools import set_plot_styles
from agents import SAC, IDHPSAC_DC
from tools import create_dir, plot_weights_idhp, plot_incremental_model
from tools.plotting import plot_weights_and_model
from tools.utils import create_dir_time, d2r, nMAE, set_random_seed

# Config
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np_config.enable_numpy_behavior()
set_plot_styles()

CONFIG_AGENT_IDHPSAC_DC = {
    "agent_sac": "trained/SAC_citation_tracking_attitude_1659223622/496000",
    "seed": None,
    #
    "lr_adapt": False,  # enable/disable adaptive learning rate
    "lr_warmup": 0,  # nr of timesteps to hold lr 0 (identity init) or lr_high (random init)
    "lr_thresh_rmse_size": 100,  # nr of samples for the rmse calculation
    "lr_thresh_delay": 100,  # nr of samples required under the rmse threshold for switch to lr_low
    #
    "gamma": 0.8,  # discount factor
    "tau": 0.01,  # target critic mixing factor
    "reward_scale": 1.0,  # multiplier to the reward signal
    #
    "identity_init": True,  # enable/disable the use of identity init for actor
    "std_init": 0.05,  # std of normal random init for actor and critic
    "activation": "tanh",  # actor and critic hidden layer activation function [tanh, relu]
    #
    "lp_enable": False,  # enable/disable low-pass filter
    "lp_w0": 0.35,  # low-pass filter cut-off frequency [rad/s]
    #
    "incr": False,  # incremental control or absolute control
    "actor": {
        "s_dim_lon": None,  # defined by SAC actor config
        "s_states_lon": None,  # defined by SAC actor config
        "s_dim_lat": None,  # defined by SAC actor config
        "s_states_lat": None,  # defined by SAC actor config
        "lr_high": 0.1,  # high learning rate in "on-off" adaptive lr strategy
        "lr_low": 0.0,  # low learning rate in "on-off" adaptive lr strategy
        "layers": [],  # defined by SAC actor config
    },
    "critic": {
        "s_dim_lon": 1 + 3,  # size of input vector - lon
        "s_states_lon": [0, 1, 2],  # input vector observation states indices - lon
        "s_dim_lat": 2 + 4,  # size of input vector - lat
        "s_states_lat": [0, 1, 2, 3],  # input vector observation states indices - lat
        "lr_high": 1.0,  # high learning rate in "on-off" adaptive lr strategy
        "lr_low": 0.0,  # low learning rate in "on-off" adaptive lr strategy
        "layers": [8],  # hidden layer sizes
    },
    "model": {
        "gamma": 1.0,  # forgetting factor
        "cov0": 1.0e8,  # initial covariance matrix magnitude
        "eps_thresh": [  # threshold per observed state for incremental model error (cov reset)
            np.inf,  # 0.001, # p
            np.inf,  # 0.001, # q
            np.inf,  # 0.001, # r
            np.inf,  # 0.0005, # alpha
            np.inf,  # 0.0005, # theta
            np.inf,  # 0.0005, # phi
            np.inf,  # 0.0005, # beta
            np.inf,  # 100.0, # h
        ],
    },
}
CONFIG_ENV_CITATION = {
    "seed": None,
    "h0": 2000,  # initial trimmed altitude
    "v0": 90,  # initial trimmed airspeed
    "trimmed": False,  # trimmed initial action is known
    "failure": None,  # failure type
    "failure_time": 30,  # failure time [s]
    "sensor_noise": False,
}
CONFIG_TASK_ATTITUDE = {
    "T": 60,  # task duration
    "dt": 0.01,  # time-step
    "tracking_scale": {  # tracking scaling factors
        "theta": 1 / d2r(30),
        "phi": 1 / d2r(30),
        "beta": 1 / d2r(7),
    },
    "tracking_thresh": {  # tracking threshold on rmse used for adaptive lr
        "theta": d2r(1),
        "phi": d2r(2),
        "beta": d2r(5),
    },
}


def main():
    # Train
    train()
    # train_lon()
    # train_lat()

    input()


def train():
    """
    Train the IDHP-SAC agent
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_IDHPSAC_DC)
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)

    # Randomize
    seed = random.randrange(sys.maxsize)
    seed = 1807305904
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    task = TrackAttitude(config_task, train_online=True)

    # Environment
    env = Citation(config_env, dt=config_task["dt"])
    env_sac = Citation(config_env, dt=config_task["dt"])

    # Agent
    config_agent["actor"]["reward_scale"] = 10.0
    config_agent["actor"]["lr_high"] = 0.1
    config_agent["critic"]["lr_high"] = 1.0

    agent_sac = SAC.load("trained/SAC_citation_tracking_attitude_1659223622/496000", task, env_sac)
    agent = IDHPSAC_DC(config_agent, task, env, agent_sac)

    # Train
    save_dir = create_dir_time(f"trained/IDHPSAC_DC_{env}_{task}")
    print("Training IDHPSAC")
    print(f"Saving to {save_dir}")

    agent.learn()

    # Save
    agent.save(save_dir)
    config_env_f = open(os.path.join(save_dir, "config_env.json"), "wt+")
    config_task_f = open(os.path.join(save_dir, "config_task.json"), "wt+")
    json.dump(config_env, config_env_f)
    json.dump(config_task, config_task_f)

    # Evaluate SAC-only
    agent_sac.evaluate()

    # Metrics
    nmae = nMAE(agent, env)
    nmae_sac = nMAE(agent_sac, env_sac)
    print(f"nMAE Hybrid: {nmae * 100 :.2f}%")
    print(f"nMAE SAC: {nmae_sac * 100 :.2f}%")

    # Plot response
    env.render(task, env_sac=env_sac)

    # Plot weights
    plot_weights_and_model(agent.agent_lon, agent.task_lon)
    plot_weights_and_model(
        agent.agent_lat, agent.task_lat, zoom_F=[-1.0, 1.2], zoom_G=[-0.3, 0.15], zoom_F_y=0.4, zoom_G_y=0
    )


def train_lon():
    """
    Train the IDHP-SAC agent
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_IDHPSAC_DC)
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)

    # Randomize
    seed = random.randrange(sys.maxsize)
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    task = TrackAttitudeLong(config_task)

    # Environment
    # config_env["failure"] = "cg_shift"
    # config_env["failure_time"] = 30
    env = Citation(config_env, dt=config_task["dt"])

    # Agent
    agent_sac = SAC.load("trained/SAC_citation_tracking_attitude_1656931473/best_eval_nmae", task, env)
    agent = IDHPSAC_DC(config_agent, task, env, agent_sac, lon_only=True)

    # Train
    save_dir = create_dir_time(f"trained/IDHPSAC_DC_{env}_{task}")
    print("Training IDHPSAC")
    print(f"Saving to {save_dir}")

    agent.learn()

    # Save
    agent.save(save_dir)
    config_env_f = open(os.path.join(save_dir, "config_env.json"), "wt+")
    config_task_f = open(os.path.join(save_dir, "config_task.json"), "wt+")
    json.dump(config_env, config_env_f)
    json.dump(config_task, config_task_f)

    # Metrics
    nmae = nMAE(agent, env)
    print(f"nMAE: {nmae * 100 :.2f}%")

    # Plot response
    env.render(task, show_rmse=True, lon_only=True)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=1)
    fig.set_figwidth(8.27)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(task.timevec, agent.agent_lon.lr_scale_history)
    plt.show(block=False)

    # Plot weights
    plot_weights_idhp(agent.agent_lon, agent.task_lon)
    plot_incremental_model(agent.agent_lon, agent.task_lon)


def train_lat():
    """
    Train the IDHP-SAC agent
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_IDHPSAC_DC)
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)

    # Randomize
    seed = random.randrange(sys.maxsize)
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    task = TrackAttitudeLat(config_task)

    # Environment
    env = Citation(config_env, dt=config_task["dt"])

    # Agent
    agent_sac = SAC.load("trained/SAC_citation_tracking_attitude_1656931473/best_eval_nmae", task, env)
    agent = IDHPSAC_DC(config_agent, task, env, agent_sac, lat_only=True)

    # Train
    save_dir = create_dir_time(f"trained/IDHPSAC_DC_{env}_{task}")
    print("Training IDHPSAC")
    print(f"Saving to {save_dir}")

    agent.learn()

    # Save
    agent.save(save_dir)
    config_env_f = open(os.path.join(save_dir, "config_env.json"), "wt+")
    config_task_f = open(os.path.join(save_dir, "config_task.json"), "wt+")
    json.dump(config_env, config_env_f)
    json.dump(config_task, config_task_f)

    # Metrics
    nmae = nMAE(agent, env)
    print(f"nMAE: {nmae * 100 :.2f}%")

    # Plot response
    env.render(task, show_rmse=True, lat_only=True)

    # Plot weights
    plot_weights_and_model(agent.agent_lat, agent.task_lat)


if __name__ == "__main__":
    main()
