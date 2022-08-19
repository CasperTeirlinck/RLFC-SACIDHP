import copy
import json
import time
import random
import shutil
import sys
import os
import numpy as np
from termcolor import colored
import pandas as pd
from tqdm import tqdm
from envs.citation_attitude import CitationAttitude
from envs.citation_attitude_idhpsac import CitationAttitudeHybrid
from tasks.tracking_altitude import TrackAltitude
from tensorflow.python.ops.numpy_ops import np_config
from tools import set_plot_styles, nMAE

from agents import SAC
from tools.utils import d2r
from agents.sac import CallbackSAC
from tools import plot_training, create_dir_time, set_random_seed
from tools.plotting import plot_incremental_model, plot_training_batch, plot_weights_and_model, plot_weights_idhp

set_plot_styles()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np_config.enable_numpy_behavior()

# Config
CONFIG_AGENT_SAC = {
    "seed": None,
    "s_states": [],  # env observation idxs included in actor/critic input vector
    "s_dim": 1,  #
    "lr_init": 3.0e-4,  # initial learning rate (with linear decay)
    "gamma": 0.99,  # discount factor
    "tau": 0.005,  # target critic smoothing factor
    "update_every": 1,  # network update frequency
    "ent_coef_init": 1.0,  # initial entropy coefficient (0.0 from softlearning, 1.0 from kdally)
    "caps_coef_t": 10,  # policy smoothing temporal regularization scaling factor
    "caps_coef_s": 10,  # policy smoothing spatial regularization scaling factor
    "caps_std": 0.05,  # policy smoothing spatial sampling std
    "batch_size": 256,  # replay buffer batch size
    "buffer_size": 1000000,  # replay buffer max size
    "incr": False,  # incremental control or absolute controls
    "actor": {
        "layers": [32, 32],  # hidden layer sizes
    },
    "critic": {
        "layers": [32, 32],  # hidden layer sizes
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
CONFIG_TASK_ALTITUDE = {
    "T": 20,  # task duration
    "dt": 0.01,  # time-step
    "tracking_scale": {  # tracking scaling factors
        "h": 1 / 240,
        "theta": 1 / d2r(30),
        "phi": 1 / d2r(30),
        "beta": 1 / d2r(7),
    },
    "tracking_thresh": {  # tracking threshold on rmse used for adaptive lr
        "h": None,
        "theta": None,
        "phi": None,
        "beta": None,
    },
}


def main(inner_save_dir=None):
    # Train
    if inner_save_dir:
        train(inner_save_dir)
        return

    # Trained batch:
    # plot_training_batch(
    #     [  # inner
    #         "trained/SAC_citation_tracking_attitude_1659223622",
    #         "trained/SAC_citation_tracking_attitude_1659223623",
    #         "trained/SAC_citation_tracking_attitude_1659223621",
    #         "trained/SAC_citation_tracking_attitude_1659223620",
    #         "trained/SAC_citation_tracking_attitude_1659223619",
    #     ],
    #     [
    #         "trained/SAC_citation_attitude_tracking_altitude_1659276263",
    #         "trained/SAC_citation_attitude_tracking_altitude_1659276265",
    #         "trained/SAC_citation_attitude_tracking_altitude_1659276264",
    #         "trained/SAC_citation_attitude_tracking_altitude_1659276267",
    #         "trained/SAC_citation_attitude_tracking_altitude_1659276266",
    #     ],
    # )

    # [2:18:06<00:00, 120.68it/s, ep=500, crashes=0, crashes_eval=0, return_avg=-30.1, return_best=-18.9, nmae_best=6.16%]
    # evaluate("trained/SAC_citation_attitude_tracking_altitude_1659276263/best_eval_r")
    # [2:17:14<00:00, 121.44it/s, ep=500, crashes=0, crashes_eval=0, return_avg=-52.4, return_best=-15, nmae_best=5.22%]
    # evaluate("trained/SAC_citation_attitude_tracking_altitude_1659276265/best_eval_r")
    # [2:19:52<00:00, 119.15it/s, ep=500, crashes=0, crashes_eval=0, return_avg=-48.5, return_best=-10.9, nmae_best=4.43%]
    # evaluate("trained/SAC_citation_attitude_tracking_altitude_1659276264/best_eval_r")
    # [2:18:21<00:00, 120.47it/s, ep=500, crashes=0, crashes_eval=0, return_avg=-52, return_best=-15.8, nmae_best=5.47%]
    # evaluate("trained/SAC_citation_attitude_tracking_altitude_1659276267/best_eval_r")
    # [2:19:33<00:00, 119.43it/s, ep=500, crashes=0, crashes_eval=0, return_avg=-40.3, return_best=-12.4, nmae_best=5.41%]
    # evaluate("trained/SAC_citation_attitude_tracking_altitude_1659276266/best_eval_r")

    input()


def train(inner_save_dir):
    """
    Train the SAC cascaded agent
    """

    # Config
    configfile = open(os.path.join(inner_save_dir, "config.json"), "r")
    config_agent_inner = json.load(configfile)
    config_agent = copy.deepcopy(CONFIG_AGENT_SAC)
    config_agent["agent_inner"] = inner_save_dir
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ALTITUDE)

    # Randomize
    # seed = config_agent_inner["seed"]
    seed = random.randrange(sys.maxsize)
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    config_task["T"] = 20
    task = TrackAltitude(config_task)
    task_eval = TrackAltitude(config_task, evaluate_tr=True)

    # Environment: inner loop
    env = CitationAttitude(config_env, task.inner, inner_save_dir, dt=config_task["dt"])

    # Agent
    agent = SAC(config_agent, task.outer, env)

    # Training callback
    save_dir = create_dir_time(f"trained/SAC_{env}_{task}")
    callback = CallbackSAC(task_eval=task_eval.outer, nmae_thresh=0.1, save_dir=save_dir)
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
    if len(sys.argv) == 1:
        main()

    elif len(sys.argv) == 2:
        inner_save_dir = sys.argv[1]
        main(inner_save_dir=inner_save_dir)

    else:
        raise Exception("Incorrect number of arguments provided.")
