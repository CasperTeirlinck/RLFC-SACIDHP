import os
import time
import tensorflow as tf
import tensorflow.keras as keras
import random
import numpy as np
import gym
from scipy.signal import butter, lfilter
from gym.spaces import Box
import json


def set_random_seed(seed):
    """
    Set random seeds for python, numpy, tf and gym
    """

    if seed is None:
        return

    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(gym.spaces, "prng"):
        gym.spaces.prng.seed(seed)
    print(f"Using random seed {seed}")


def nMAE(agent, env, cascaded=False):
    """
    Calculate normalized Mean Absolute Error
    Normalized using treference signal tracking range
    Averaged over the tracking states
    """

    # If cascaded
    # if type(agent.task).__bases__[0] == TrackingTaskCascaded:
    if cascaded:
        agent_outer = agent
        agent_inner = env.agent_inner
        env = env.env_inner

        y_outer = (
            agent_outer.tracking_P_external
            @ np.hstack(
                [env.state_to_obs(np.array(env.state_history).T).T, (np.array(env.state_history).T)[env.obs_extra].T]
            ).T
        )
        y_inner = agent_inner.tracking_P_external @ env.state_to_obs(np.array(env.state_history).T)
        y = np.vstack([y_outer, y_inner])

        y_ref_outer = agent_outer.tracking_ref
        y_ref_inner = agent_inner.tracking_ref
        y_ref = np.vstack([y_ref_outer, y_ref_inner])

        y_range_outer = np.array([agent_outer.task.tracking_range[_] for _ in agent_outer.tracking_descr_external])
        y_range_inner = np.array([agent_inner.task.tracking_range[_] for _ in agent_inner.tracking_descr_external])
        y_range = np.hstack([y_range_outer, y_range_inner])

    # If simple
    else:
        y = agent.tracking_P @ env.state_to_obs(np.array(env.state_history).T)
        y_ref = agent.tracking_ref
        y_range = np.array([agent.task.tracking_range[_] for _ in agent.tracking_descr])

    mae_vec = np.mean(np.abs(y - y_ref), axis=1)
    nmae_vec = mae_vec / y_range
    nmae = np.mean(nmae_vec)

    return nmae


def nRMSE(agent, env):
    """
    Calculate normalized Root Mean Squared Error
    Normalized using treference signal tracking range
    Averaged over the tracking states
    """

    y = agent.tracking_P @ env.state_to_obs(np.array(env.state_history).T)
    y_ref = agent.tracking_ref
    y_range = np.array([agent.task.tracking_range[_] for _ in agent.tracking_descr])

    rmse_vec = np.sqrt(np.mean(np.square(y - y_ref), axis=1))
    nrmse_vec = rmse_vec / y_range
    nrmse = np.mean(nrmse_vec)

    return nrmse


def incr_action(action_prev, action_pi, env: gym.Env, dt):

    # Scale action increment
    action_incr = scale_action(action_pi, env.action_space_rates) * dt

    # Add to previous action
    action_env = action_prev + action_incr

    if tf.is_tensor(action_env):
        action_env = tf.clip_by_value(action_env, env.action_space.low, env.action_space.high)
    else:
        action_env = np.float32(np.clip(action_env, env.action_space.low, env.action_space.high))

    return action_env


def incr_action_symm(action_prev, action_pi, env: gym.Env, dt):

    # Scale action increment
    action_incr = scale_action_symm(action_pi, env.action_space_rates) * dt

    # Add to previous action
    action_env = action_prev + action_incr

    if tf.is_tensor(action_env):
        action_env = tf.clip_by_value(action_env, env.action_space.low, env.action_space.high)
    else:
        action_env = np.float32(np.clip(action_env, env.action_space.low, env.action_space.high))

    return action_env


def scale_action(action_pi, action_space: Box):
    """
    Rescale the action from [-1, 1] to [action_space.low, action_space.high]
    """

    # Scale to action space bounds
    low, high = np.float32(action_space.low), np.float32(action_space.high)
    action = low + 0.5 * (action_pi + 1.0) * (high - low)

    return action


def scale_action_symm(action_pi, action_space: Box):
    """
    Rescale the action from [-1, 1] to [action_space.low, action_space.high]
    Enforce symmetrical scaling and clip on asymmetrical bounds
    """

    # Scale to action space bounds
    low, high = np.float32(action_space.low), np.float32(action_space.high)
    scale = np.max(np.abs([low, high]), axis=0)
    action = scale * action_pi
    if tf.is_tensor(action):
        action = tf.clip_by_value(action, low, high)
    else:
        action = np.clip(action, low, high)

    return action


def unscale_action_symm(action_pi, action_space: Box):
    """
    Rescale the action from [action_space.low, action_space.high] to [-1, 1]
    Enforce symmetrical scaling and clip on asymmetrical bounds
    """

    # Scale to action space bounds
    low, high = np.float32(action_space.low), np.float32(action_space.high)
    scale = np.max(np.abs([low, high]), axis=0)
    action = action_pi / scale
    if tf.is_tensor(action):
        action = tf.clip_by_value(action, -np.ones_like(low), np.ones_like(high))
    else:
        action = np.clip(action, -np.ones_like(low), np.ones_like(high))

    return action


def unscale_action(action, action_space: Box):
    """
    Rescale the action from [action_space.low, action_space.high] to [-1, 1]
    Assumes the model operates on tanh squashed actions
    """

    # Scale to [-1, 1]
    low, high = action_space.low, action_space.high
    action = 2.0 * ((action - low) / (high - low)) - 1.0

    return action


def concat(arr, axis, t_f=False):
    """ """

    if t_f:
        a = tf.concat(arr, axis=axis)
    else:
        a = np.hstack(arr)

    return a


def clip(value, low, high):
    """
    Simple clipping function
    """

    return max(min(value, high), low)


def d2r(x):
    """
    Convert degrees to radians
    """

    return x * np.pi / 180


def r2d(x):
    """
    Convert radians to degrees
    """

    return x * 180 / np.pi


def create_dir(dir_path):
    """
    Create directory of it does not exist
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def create_dir_time(dir_path):
    """
    Append time to directory and make sure not to overwrite
    """

    dir_path_time = f"{dir_path}_{str(int(time.time()))}"

    if not os.path.exists(dir_path_time):
        try:
            os.makedirs(dir_path_time)
        except FileExistsError:
            time.sleep(10 / 1000)
            return create_dir_time(dir_path)
    else:
        time.sleep(10 / 1000)
        return create_dir_time(dir_path)

    return dir_path_time


def low_pass(x, x_prev, w_0, dt):
    """
    Low pass filter
    """

    alpha = dt / (1 / w_0 + dt)
    return alpha * x + (1 - alpha) * x_prev
