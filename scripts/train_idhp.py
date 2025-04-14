import os
import random
import sys
import time
import copy
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

from tools import set_plot_styles
from tasks import TrackPitchRate, TrackAoA
from envs import ShortPeriod
from agents import IDHP
from tools import create_dir, plot_weights_idhp, plot_incremental_model, plot_inputs
from tools.utils import d2r, set_random_seed

set_plot_styles()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np_config.enable_numpy_behavior()

# Config
CONFIG_AGENT_IDHP = {
    "seed": None,
    "lr_adapt": False,  # enable/disable adaptive learning rate
    "lr_warmup": 100,  # nr of timesteps to hold initial high learning rate
    "lr_thresh_rmse": d2r(1.0),  # threshold on the tracking error rmse for the adaptive lr
    "lr_thresh_rmse_size": 50,  # nr of samples for the rmse calculation
    "reward_scale": 20.0,  # multiplier to the reward signal
    "gamma": 0.6,  # discount factor
    "tau": 0.01,  # target critic mixing factor
    "identity_init": False,  # enable/disable the use of identity init for actor and critic
    "std_init": 0.1,  # std of normal random init for actor and critic
    "activation": "tanh",  # actor and critic hidden layer activation function [tanh, relu]
    "actor": {
        "s_dim": 1 + 1 + 1,  # size of input vector
        "s_states": [1],  # env observation idxs included in input vector
        "lr_high": 10.0,  # high learning rate in "on-off" adaptive lr strategy
        "lr_low": 10.0 / 10,  # low learning rate in "on-off" adaptive lr strategy
        "layers": [10],  # hidden layer sizes
    },
    "critic": {
        "s_dim": 1 + 1 + 1,  # size of input vector
        "s_states": [1],  # env observation idxs included in input vector
        "lr_high": 5.0,  # high learning rate in "on-off" adaptive lr strategy
        "lr_low": 5.0 / 10,  # low learning rate in "on-off" adaptive lr strategy
        "layers": [10],  # hidden layer sizes
    },
    "model": {
        "gamma": 1.0,  # forgetting factor
        "cov0": 1.0e8,  # initial covariance matrix magnitude
        "eps_thresh": [  # threshold per state for incremental model error (cov reset)
            d2r(0.0005),
            d2r(0.001),
        ],
    },
}
CONFIG_ENV_SHORTPERIOD = {
    "seed": None,
    "dynamics": "ce500",  # ce500 or ce172
    "trimmed": True,  # initialize at trimmed state
    "fault": "",  # "", "mild" or "severe"
    "fault_timestep": 1500,
}


def main():
    # Train
    train()

    # Plot batch
    # train_batch(100)
    # batch_dir = "trained/IDHP_shortperiod_tracking_alpha_batch"
    # task = TrackAoA(T=60, dt=0.01, ref_shape=2)
    # config_agent = copy.deepcopy(CONFIG_AGENT_IDHP)
    # config_env = copy.deepcopy(CONFIG_ENV_SHORTPERIOD)
    # env = ShortPeriod(config_env, task)
    # env.render(batch_dir=batch_dir, lr_warmup=config_agent["lr_warmup"], idx_end=3000)

    input()


def train():
    """
    Train the IDHP agent
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_IDHP)
    config_env = copy.deepcopy(CONFIG_ENV_SHORTPERIOD)

    # Randomize
    seed = random.randrange(sys.maxsize)
    config_agent["seed"] = seed
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    task = TrackAoA(T=60, dt=0.01, ref_shape=2)
    # task = TrackPitchRate(T=30, dt=0.01, ref_shape=2)

    # Environment
    env = ShortPeriod(config_env, dt=0.01)

    # Agent
    config_agent["lr_adapt"] = False
    # relu:
    config_agent["reward_scale"] = 4.0
    config_agent["actor"]["lr_high"] = 0.2
    config_agent["actor"]["lr_low"] = 0.2 / 10
    config_agent["critic"]["lr_high"] = 0.02
    config_agent["critic"]["lr_low"] = 0.02 / 10
    config_agent["actor"]["layers"] = [32, 32]
    config_agent["critic"]["layers"] = [32, 32]

    # config_agent["reward_scale"] = 10.0
    # config_agent["actor"]["lr_high"] = 0.08
    # config_agent["actor"]["lr_low"] = 0.2 / 10
    # config_agent["critic"]["lr_high"] = 0.008
    # config_agent["critic"]["lr_low"] = 0.02 / 10
    # config_agent["actor"]["layers"] = [8]
    # config_agent["critic"]["layers"] = [8]

    # config_agent["reward_scale"] = 10.0
    # config_agent["actor"]["lr_high"] = 0.1
    # config_agent["actor"]["lr_low"] = 0.2 / 10
    # config_agent["critic"]["lr_high"] = 0.01
    # config_agent["critic"]["lr_low"] = 0.02 / 10
    # config_agent["actor"]["layers"] = [8]
    # config_agent["critic"]["layers"] = [8]
    agent = IDHP(config_agent, task, env)

    # Train
    agent.learn()

    # Plot response
    env.render(agent, task)

    # Plot other
    plot_weights_idhp(agent, task)
    plot_incremental_model(agent, task)
    plot_inputs(agent)


def train_batch(episodes):
    """
    Train the IDHP agent on different initial conditions
    """

    # Config
    config_agent = copy.deepcopy(CONFIG_AGENT_IDHP)
    config_env = copy.deepcopy(CONFIG_ENV_SHORTPERIOD)

    # Task
    task = TrackAoA(T=60, dt=0.01, ref_shape=2)
    # task = TaskPitchRate(T=60, dt=0.01, ref_shape=1)

    # Environment
    config_env["trimmed"] = False
    env = ShortPeriod(config_env, task)

    # Agent
    agent = IDHP(config_agent, env)

    # Save directory
    save_dir = create_dir(f"trained/IDHP_{env}_{task}_batch_{str(int(time.time()))}")

    # Train multiple episodes
    for i in range(episodes):
        print(f"Episode {i+1} / {episodes}")

        # Train
        agent.learn()

        # Save response
        np.save(os.path.join(save_dir, f"state_history_{i}"), env.state_history)
        np.save(os.path.join(save_dir, f"action_history_{i}"), env.action_history)
        np.save(os.path.join(save_dir, f"rmse_history_{i}"), env.rmse_history)


if __name__ == "__main__":
    main()
