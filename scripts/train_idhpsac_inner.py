import json
import os
import random
import sys
import tensorflow.keras as keras
import time
import copy
from matplotlib import pyplot as plt
import numpy as np
from envs.citation import Citation
from tasks.tracking_attitude import TrackAttitude

from tools import set_plot_styles
from agents import IDHPSAC, SAC
from tools.plotting import plot_weights_and_model
from tools.utils import create_dir_time, d2r, nMAE, set_random_seed
from tensorflow.python.ops.numpy_ops import np_config

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np_config.enable_numpy_behavior()
set_plot_styles()

# Config
CONFIG_AGENT_IDHPSAC = {
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
        "s_dim": None,  # defined by SAC actor config
        "s_states": None,  # defined by SAC actor config
        "lr_high": 0.1,  # high learning rate in "on-off" adaptive lr strategy
        "lr_low": 0.0,  # low learning rate in "on-off" adaptive lr strategy
        "layers": [],  # defined by SAC actor config
    },
    "critic": {
        "s_dim": 3 + 7,  # size of input vector
        "s_states": [0, 1, 2, 3, 4, 5, 6],  # input vector observation states indices
        "lr_high": 1,  # high learning rate in "on-off" adaptive lr strategy
        "lr_low": 0.0,  # low learning rate in "on-off" adaptive lr strategy
        "layers": [8],  # hidden layer sizes
    },
    "model": {
        "obs_states": [0, 1, 2, 3, 4, 5, 6],  # select relevant states from full observation vector
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
        "phi": d2r(1),
        "beta": d2r(5),
    },
}


def main():
    # Train
    train()

    # Batch trainings:

    # std = 0.05: 50 runs
    # convergence rate: 4 failed = 92.00%
    # temporal loss success rate: 78.26%
    # => 72.00% total success rate

    # std = 0.01: 50 runs
    # convergence rate: 50/0 = 100%
    # temporal loss success rate: 100%
    # => 100% total success rate

    # std = 0.1: 50 runs
    # convergence rate: 16 failed = 68.00%
    # temporal loss success rate: 70.59%
    # => 48.00% total success rate

    # plot
    # config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)
    # config_task["T"] = 60
    # task = TrackAttitude(config_task, train_online=True)

    # config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    # env = Citation(config_env, dt=config_task["dt"])
    # env.render_batch(
    #     task,
    #     save_dirs=[
    #         # std 0.05
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425402",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425407",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425401",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425406",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425409",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425405",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425408",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425403",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659425404",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427751",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427747",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427744",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427745",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427749",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427742",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427743",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427748",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427750",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659427746",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428699",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428702",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428704",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428700",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428703",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428701",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428698",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428696",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428695",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428873",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428867",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428872",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428869",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428868",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428871",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428874",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428865",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428866",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428993",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428992",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428991",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428995",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428990",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428988",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428987",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428994",
    #         "trained/IDHPSAC_citation_tracking_attitude_1659428986",
    #         # std 0.01
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430132",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430135",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430130",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430133",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430131",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430137",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430129",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430128",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430136",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430134",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430312",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430315",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430314",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430313",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430311",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430308",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430310",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430309",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430316",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430307",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430908",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430909",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430906",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430910",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430907",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430913",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430912",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430905",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430911",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659430904",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431008",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431013",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431012",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431011",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431014",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431009",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431006",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431007",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431010",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431005",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431093",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431090",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431097",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431095",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431096",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431092",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431089",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431094",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431091",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431088",
    #         # std=0.1
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431536",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431520",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431519",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431532",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431531",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431533",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431617",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431628",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431616",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431626",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431631",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431614",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431525",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431522",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431521",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431530",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431523",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431534",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431625",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431624",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431629",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431620",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431627",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431615",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431630",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431907",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431910",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431908",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431913",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431911",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431919",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431916",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431915",
    #         # "trained/IDHPSAC_citation_tracking_attitude_1659431914",
    #     ],
    # )

    input()


def train():
    """
    Train the IDHP-SAC agent
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_IDHPSAC)
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)

    # Randomize
    seed = random.randrange(sys.maxsize)
    seed = 2759503318
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    config_task["T"] = 60
    task = TrackAttitude(config_task, train_online=True)

    # Environment
    env = Citation(config_env, dt=config_task["dt"])
    env_sac = Citation(config_env, dt=config_task["dt"])

    # Agent
    config_agent["actor"]["lr_high"] = 0.2
    config_agent["critic"]["lr_high"] = 1.0
    agent_sac = SAC.load("trained/SAC_citation_tracking_attitude_1659223622/496000", task, env_sac)
    agent = IDHPSAC(config_agent, task, env, agent_sac)

    # Train
    save_dir = create_dir_time(f"trained/IDHPSAC_{env}_{task}")
    print(f"Saving to {save_dir}")

    success = agent.learn()

    # Save state-action history
    # if success:
    #     np.savez(
    #         os.path.join(save_dir, "stateaction_history.npz"),
    #         state_history=env.state_history,
    #         action_history=env.action_history,
    #     )

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
    # plot_weights_and_model(agent, task, zoom_F=[-1.5, 1.0], zoom_G=[-0.35, 0.05])


if __name__ == "__main__":
    main()
