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


def main():
    # evaluate("trained/SAC_citation_attitude_tracking_altitude_1659276263/764000")
    evaluate_idhpsac(
        "trained/SAC_citation_attitude_tracking_altitude_1659276263/764000",  # SAC altitude controller
        "trained/IDHPSAC_citation_tracking_attitude_1659271870",  # IDHPSAC attitude controller
    )

    input()


def evaluate(save_dir, find_best=False, find_best_extensive=False):
    """
    Evaluate the SAC cascaded agent
    """

    # Config
    configfile = open(os.path.join(save_dir, "config.json"), "r")
    config_agent = json.load(configfile)
    inner_save_dir = config_agent["agent_inner"]
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ALTITUDE)

    # Randomize
    seed = config_agent["seed"]
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    config_task["T"] = 120
    task = TrackAltitude(config_task, evaluate=True)

    # Environment: inner loop
    # config_env["failure"] = "da_reduce"
    # config_env["failure_time"] = 30
    env = CitationAttitude(config_env, task.inner, inner_save_dir, dt=config_task["dt"])

    # Find best performing agent on task
    if find_best:
        print(save_dir)

        # Get all best saved agents
        save_dirs_all = [_.path for _ in os.scandir(os.path.dirname(save_dir)) if _.is_dir() and _.name.isnumeric()]
        save_dirs_best = [
            os.path.join(os.path.dirname(os.path.dirname(_)), _.name)
            for _ in os.scandir(save_dir)
            if _.is_file() and _.name.isnumeric()
        ]
        save_dirs_all.sort(key=lambda x: int(os.path.basename(x)), reverse=True)
        save_dirs_best.sort(key=lambda x: int(os.path.basename(x)), reverse=True)

        if find_best_extensive:
            save_dirs = save_dirs_all
        else:
            save_dirs = save_dirs_best

        for i, save_dir in enumerate(save_dirs):
            # Get new environment
            env = CitationAttitude(config_env, task.inner, inner_save_dir, dt=config_task["dt"])

            # Load agent
            agent = SAC.load(save_dir, task.outer, env)

            # Evaluate
            episode_return = agent.evaluate()

            print(f"agent {os.path.basename(save_dir)}\t[{i+1}/{len(save_dirs)}]:\t\t", end="")

            if episode_return:
                # Metrics: NMAE
                nmae = nMAE(agent, env, cascaded=True)
                color = "green" if nmae <= 0.05 else "red"
                print(f"nMAE: {colored(f'{nmae * 100 :.2f}', color)}%\t", end="")

                # Metrics: Temporal loss
                action_loss_t = 0
                action_history = np.array(env.action_history)
                for i in range(1, action_history.shape[0]):
                    action_loss_t += np.sum((action_history[i] - action_history[i - 1]) ** 2)
                print(f"loss_t: {action_loss_t :.3f}")
            else:
                print("crash")
    else:
        # Load agent
        agent = SAC.load(save_dir, task.outer, env)

        # Evaluate
        agent.evaluate()

        # Metrics: NMAE
        nmae = nMAE(agent, env, cascaded=True)
        color = "green" if nmae <= 0.05 else "red"
        print(f"nMAE: {colored(f'{nmae * 100 :.2f}', color)}%\t", end="")

        # Plots
        env.render(task, show_rmse=False)


def evaluate_idhpsac(save_dir, save_dir_idhpsac):
    """
    Evaluate the IDHP-SAC cascaded agent
    """

    # Config
    configfile_outer = open(os.path.join(save_dir, "config.json"), "r")
    config_agent_outer = json.load(configfile_outer)

    configfile_idhpsac = open(os.path.join(save_dir_idhpsac, "config.json"), "r")
    config_agent_idhpsac = json.load(configfile_idhpsac)

    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ALTITUDE)

    # Randomize
    seed = config_agent_idhpsac["seed"]
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    config_task["T"] = 120
    task = TrackAltitude(config_task, evaluate=True)
    # config_task["T"] = 20
    # task = TrackAltitude(config_task, evaluate_disturbance=True)

    # Environment: inner loop
    config_agent_idhpsac["actor"]["lr_high"] = 0.02
    config_agent_idhpsac["critic"]["lr_high"] = 0.1

    # config_agent_idhpsac["model"]["eps_thresh"][1] = 0.001  # q

    # config_agent_idhpsac["lp_enable"] = True
    # config_agent_idhpsac["lp_w0"] = d2r(40)

    config_env["failure"] = "cg_shift"
    config_env["failure_time"] = 30
    # config_env["control_disturbance"] = task.get_control_disturbance()
    # config_env["atm_disturbance"] = True
    # config_env["sensor_noise"] = True
    env = CitationAttitudeHybrid(config_env, task.inner, save_dir_idhpsac, config_agent_idhpsac, dt=config_task["dt"])

    # Load agent
    agent = SAC.load(save_dir, task.outer, env)

    # Evaluate
    agent.evaluate()

    # Evaluate SAC-only
    env_sac = CitationAttitude(config_env, task.inner, config_agent_outer["agent_inner"], dt=config_task["dt"])
    agent_sac = SAC.load(save_dir, task.outer, env_sac)
    agent_sac.evaluate()
    # agent_sac.evaluate(lp_w0=d2r(40))

    # Metrics
    nmae = nMAE(agent, env, cascaded=True)
    nmae_sac = nMAE(agent_sac, env_sac, cascaded=True)
    print(f"nMAE Hybrid: {nmae * 100 :.2f}%")
    print(f"nMAE SAC: {nmae_sac * 100 :.2f}%")

    # Plots
    env.render(task, env_sac=agent_sac.env.env_inner, show_rmse=False)
    # plot_weights_and_model(env.agent_inner, task)


if __name__ == "__main__":
    main()
