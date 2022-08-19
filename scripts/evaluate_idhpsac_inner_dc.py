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


def main():
    # Evaluate trained agent
    evaluate("trained/IDHPSAC_DC_citation_tracking_attitude_1659358080")

    input()


def evaluate(save_dir):
    """
    Evaluate
    """

    # Config
    config_agent_f = open(os.path.join(save_dir, "config.json"), "r")
    config_env_f = open(os.path.join(save_dir, "config_env.json"), "r")
    config_task_f = open(os.path.join(save_dir, "config_task.json"), "r")
    config_agent = json.load(config_agent_f)
    config_env = json.load(config_env_f)
    config_task = json.load(config_task_f)

    # Randomize
    seed = config_agent["seed"]
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    task = TrackAttitude(config_task, evaluate_hard=True)

    # Environment
    config_env["failure"] = "ht_reduce"
    config_env["failure_time"] = 30
    env = Citation(config_env, dt=0.01)
    env_sac = Citation(config_env, dt=0.01)

    # Load agent
    config_agent["actor"]["lr_high"] = 0.02
    config_agent["critic"]["lr_high"] = 0.1
    agent_sac = SAC.load("trained/SAC_citation_tracking_attitude_1659223622/496000", task, env_sac)
    agent: IDHPSAC_DC = IDHPSAC_DC.load(save_dir, task, env, agent_sac, config=config_agent)

    # Evaluate
    agent.learn()

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
    plot_weights_and_model(agent.agent_lat, agent.task_lat)


def evaluate_lon(save_dir):
    """
    Evaluate
    """

    # Config
    config_agent_f = open(os.path.join(save_dir, "config.json"), "r")
    config_env_f = open(os.path.join(save_dir, "config_env.json"), "r")
    config_task_f = open(os.path.join(save_dir, "config_task.json"), "r")
    config_agent = json.load(config_agent_f)
    config_env = json.load(config_env_f)
    config_task = json.load(config_task_f)

    # Randomize
    seed = config_agent["seed"]
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    # config_task[""] =
    task = TrackAttitudeLong(config_task, evaluate=True)

    # Environment
    # config_env["failure"] = "de_reduce"
    # config_env["failure_time"] = 10
    env = Citation(config_env, dt=0.01)

    # Load agent
    config_agent["actor"]["lr_high"] = 10
    config_agent["critic"]["lr_high"] = 2
    agent_sac = SAC.load_npz("trained/SAC_citation_tracking_attitude_GT0PLE", task, env)  # TODO in save
    agent: IDHPSAC_DC = IDHPSAC_DC.load(save_dir, task, env, agent_sac, lon_only=True, config=config_agent)

    # Evaluate
    agent.learn()

    # Metrics
    nmae = nMAE(agent, env)
    print(f"nMAE: {nmae * 100 :.2f}%")

    # Plot response
    env.render(task, show_rmse=False, lr_warmup=config_agent["lr_warmup"])

    # Plot weights
    plot_weights_and_model(agent.agent_lon, agent.task_lon)


if __name__ == "__main__":
    main()
