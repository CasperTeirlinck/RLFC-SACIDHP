import copy
import csv
import json
import time
import random
import shutil
import sys
import os
from termcolor import colored
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import signal
from tqdm import tqdm
from tasks.tracking_attitude_long import TrackAttitudeLong

from tools import set_plot_styles, nMAE, nRMSE
from tasks import TrackAttitude
from envs import Citation

from agents import SAC
from agents.sac import CallbackSAC
from tools import plot_training, create_dir_time, set_random_seed
from tools.plotting import plot_training_batch, plot_weights_sac
from tensorflow.python.ops.numpy_ops import np_config

from tools.utils import d2r

set_plot_styles()
os.system("color")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np_config.enable_numpy_behavior()

# Config
CONFIG_AGENT_SAC = {
    "seed": None,
    "s_states": [0, 1, 2],  # env observation idxs included in actor/critic input vector
    "s_dim": 3 + 3,  #
    "lr_init": 4.4e-4,  # initial learning rate (with linear decay)
    "gamma": 0.99,  # discount factor
    "tau": 0.005,  # target critic smoothing factor
    "update_every": 1,  # network update frequency
    "ent_coef_init": 1.0,  # initial entropy coefficient (0.0 from softlearning, 1.0 from kdally)
    "caps_coef_t": 400,  # policy smoothing temporal regularization scaling factor
    "caps_coef_s": 400,  # policy smoothing spatial regularization scaling factor
    "caps_std": 0.05,  # policy smoothing spatial sampling std
    "batch_size": 256,  # replay buffer batch size
    "buffer_size": 1000000,  # 50000 replay buffer max size
    "incr": False,  # incremental control or absolute control
    "actor": {
        "layers": [64, 64],  # hidden layer sizes
    },
    "critic": {
        "layers": [64, 64],  # hidden layer sizes
    },
}
CONFIG_ENV_CITATION = {
    "seed": None,
    "h0": 2000,  # initial trimmed altitude
    "v0": 90,  # initial trimmed airspeed
    "trimmed": False,  # trimmed initial action is known
    "failure": None,  # failure type
    "failure_time": 10,  # failure time [s]
    "sensor_noise": False,
}
CONFIG_TASK_ATTITUDE = {
    "T": 20,  # task duration
    "dt": 0.01,  # time-step
    "tracking_scale": {  # tracking scaling factors
        "theta": 1 / d2r(30),
        "phi": 1 / d2r(30),
        "beta": 1 / d2r(7),
    },
    "tracking_thresh": {  # tracking threshold on rmse used for adaptive lr
        "theta": None,
        "phi": None,
        "beta": None,
    },
}


def main():
    # Train
    train()

    input()


def train():
    """
    Train the SAC agent
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_SAC)
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)

    # Randomize
    seed = random.randrange(sys.maxsize)
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    config_task["T"] = 20
    task = TrackAttitude(config_task)
    task_eval = TrackAttitude(config_task, evaluate_tr=True)

    # Environment
    env = Citation(config_env, dt=config_task["dt"])

    # Agent
    agent = SAC(config_agent, task, env)

    # Training callback
    save_dir = create_dir_time(f"trained/SAC_{env}_{task}")
    callback = CallbackSAC(task_eval=task_eval, nmae_thresh=0.1, save_dir=save_dir)
    print("Training SAC")
    print(f"Saving to {save_dir}")

    # Train
    success = agent.learn(num_timesteps=int(1e6), callback=callback)

    # Success
    if success:
        print(f"Success")
    # Crash restart
    else:
        print(f"Crashed")
        train()

    # Save
    save_dir = os.path.join(save_dir, "latest")
    agent.save(save_dir)


if __name__ == "__main__":
    main()
