from cmath import isinf
from fnmatch import fnmatch
import os
from turtle import color
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from tools.utils import r2d


def set_plot_styles():
    """
    Configures the default styles for mpl
    """

    mpl.rcParams["axes.prop_cycle"] = cycler(
        "color",
        ["#00a6d6", "#ec6842", "#6cc24a", "#e03c31", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    )
    mpl.rcParams["figure.autolayout"] = True
    mpl.rcParams["lines.linewidth"] = 1.5
    mpl.rcParams["grid.color"] = "C1C1C1"
    mpl.rcParams["grid.linestyle"] = ":"
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.xmargin"] = 0
    mpl.rcParams["axes.ymargin"] = 0.1
    mpl.rcParams["axes.labelpad"] = 4.0
    mpl.rcParams["legend.framealpha"] = 1.0
    mpl.rcParams["legend.edgecolor"] = "k"
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"


def plot_training(save_dir):
    """
    Plot training metrics from file
    """

    file = os.path.join(save_dir, "training.csv")
    df = pd.read_csv(file, header=0)
    ep_return = df["return"]

    avg_size = 20
    return_avg = np.array(
        [np.mean(ep_return[-avg_size - (len(ep_return) - (i + 1)) : i + 1]) for i in range(len(ep_return))]
    )

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=1, height_ratios=[1])
    fig.set_figwidth(8.27)
    fig.set_figheight(5)

    # Reward curve
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"timestep [-]")
    ax.set_ylabel(r"episode return")

    ax.plot(df["timestep"], df["return"], linestyle="-")
    # ax.plot(df["timestep"], df["return"].rolling(window=20, min_periods=1).mean(), linestyle="-")
    # ax.plot(df["timestep"], return_avg, linestyle="-")
    ax.set_yscale("symlog")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([-2000, -1000, -500, -200, -100])

    fig.savefig(os.path.join(save_dir, "training.png"))

    plt.show(block=False)

    # Losses
    # npzfile = np.load(os.path.join(save_dir, "latest", "logging.npz"))

    # actor_loss = npzfile["actor_loss"]
    # actor_loss_t = npzfile["actor_loss_t"]
    # actor_loss_s = npzfile["actor_loss_s"]
    # critic_1_loss = npzfile["critic_1_loss"]
    # critic_2_loss = npzfile["critic_2_loss"]

    # fig = plt.figure()
    # gs = fig.add_gridspec(nrows=2, ncols=1)
    # fig.set_figwidth(8.27)

    # ax = fig.add_subplot(gs[0, 0])
    # ax.set_xlabel(r"timestep [-]")
    # ax.set_ylabel(r"$L_{\pi}$")

    # ax.plot(np.arange(0, actor_loss.shape[0], 1), actor_loss, linestyle="-")

    # ax = fig.add_subplot(gs[1, 0])
    # ax.set_xlabel(r"timestep [-]")
    # ax.set_ylabel(r"$L_{CAPS}$")

    # ax.plot(np.arange(0, actor_loss_t.shape[0], 1), actor_loss_t, linestyle="-", label=r"$L_T$")
    # ax.plot(np.arange(0, actor_loss_s.shape[0], 1), actor_loss_s, linestyle="-", label=r"$L_S$")
    # ax.legend(bbox_to_anchor=(1, 1), loc=1)

    # plt.show(block=False)


def plot_training_batch(save_dirs, save_dirs_2=None):
    """ """

    df = get_training_batch_df(save_dirs)
    if save_dirs_2:
        df2 = get_training_batch_df(save_dirs_2)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=1, height_ratios=[1])
    # fig.set_figwidth(8.27)
    # fig.set_figheight(4.135)
    fig.set_figwidth(4)
    fig.set_figheight(3)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"Timestep [-]")
    ax.set_ylabel(r"Episode return")

    ax.plot(df["timestep"], df["return_avg_smooth"], color="C0", label="Attitude Controller")
    ax.fill_between(df["timestep"], df["return_up_smooth"], df["return_down_smooth"], color="C0", alpha=0.3)
    # ax.fill_between(df["timestep"], df["return_max_smooth"], df["return_min_smooth"], color="C1", alpha=0.3)

    if save_dirs_2:
        ax.plot(
            df2["timestep"],
            df2["return_avg_smooth"],
            color="C2",
            label="Altitude Controller",
            # label="Altitude Controller\n(with pre-trained inner loop attitude controller)",
        )
        ax.fill_between(df2["timestep"], df2["return_up_smooth"], df2["return_down_smooth"], color="C2", alpha=0.3)

    ax.set_yscale("symlog")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([-2000, -1000, -500, -200, -100])
    xticks = ax.get_xticks()
    xticks[-1] = 1000000
    ax.set_xticks(xticks)
    if save_dirs_2:
        ax.legend(loc=4, ncol=1)

    plt.show(block=False)


def get_training_batch_df(save_dirs):
    """
    Get training batch dataframe for plotting
    """

    return_batch = []

    for save_dir in save_dirs:
        file = os.path.join(save_dir, "training.csv")
        df_single = pd.read_csv(file, header=0)
        return_batch.append(df_single["return"])
    idx_end = np.min([len(_) for _ in return_batch])
    return_batch = [_[:idx_end] for _ in return_batch]
    return_batch = np.array(return_batch)

    df = pd.DataFrame()
    df["timestep"] = df_single["timestep"][:idx_end]

    df["return_avg"] = return_batch.mean(axis=0)
    df["return_up"] = np.quantile(return_batch, 0.75, axis=0)
    df["return_down"] = np.quantile(return_batch, 0.25, axis=0)

    df["return_max"] = np.max(return_batch, axis=0)
    df["return_min"] = np.min(return_batch, axis=0)

    df["return_avg_smooth"] = df["return_avg"].rolling(window=20, min_periods=1).mean()
    df["return_up_smooth"] = df["return_up"].rolling(window=20, min_periods=1).mean()
    df["return_down_smooth"] = df["return_down"].rolling(window=20, min_periods=1).mean()

    df["return_max_smooth"] = df["return_max"].rolling(window=20, min_periods=1).mean()
    df["return_min_smooth"] = df["return_min"].rolling(window=20, min_periods=1).mean()

    return df


def plot_weights_sac(agent):
    """
    Plot weights and biases of trained SAC agent
    """

    fig1 = plt.figure()
    gs1 = fig1.add_gridspec(nrows=2, ncols=3)
    fig1.set_figwidth(8.27)
    fig2 = plt.figure()
    gs2 = fig2.add_gridspec(nrows=2, ncols=3)
    fig2.set_figwidth(8.27)
    fig3 = plt.figure()
    gs3 = fig3.add_gridspec(nrows=2, ncols=3)
    fig3.set_figwidth(8.27)
    fig4 = plt.figure()
    gs4 = fig4.add_gridspec(nrows=2, ncols=3)
    fig4.set_figwidth(8.27)

    layers = [
        layer
        for layer in agent.actor.policy.layers
        if any(fnmatch(layer.name, _) for _ in ["hidden_*", "layer_norm_*", "output_*"])
    ]
    for i in range(3):
        for j in range(2):
            layer = layers[i * 2 + j]

            # Axes
            ax1 = fig1.add_subplot(gs1[j, i])
            ax2 = fig2.add_subplot(gs2[j, i])
            ax3 = fig3.add_subplot(gs3[j, i])
            ax4 = fig4.add_subplot(gs4[j, i])

            # Get weights and biases
            weights, biases = layer.get_weights()
            weights = np.sort(weights.flatten())
            biases = np.sort(biases.flatten())

            # Weights
            ax1.bar(range(len(weights)), weights, width=1.0, color="C0")
            ax1.set_xlabel(rf"{layer.name}")
            if i == 0:
                ax1.set_ylabel(rf"$|w|$ [-]")

            # Biases
            ax2.bar(range(len(biases)), biases, width=1.0, color="C1")
            ax2.set_xlabel(rf"{layer.name}")
            if i == 0:
                ax2.set_ylabel(rf"$|b|$ [-]")

            # Weights hist
            ax3.hist(weights, bins=len(weights), density=True, color="C0")
            ax3.set_xlabel(rf"{layer.name}")
            if i == 0:
                ax3.set_ylabel(rf"$w$ [%]")

            # Biases hist
            ax4.hist(biases, bins=len(biases), density=True, color="C1")
            ax4.set_xlabel(rf"{layer.name}")
            if i == 0:
                ax4.set_ylabel(rf"$w$ [%]")

    plt.show(block=False)


def plot_grads_idhp(agent, task):
    """ """

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
    fig.set_figwidth(8.27)
    fig.set_figheight(4)

    actor_num_layers = len(agent.config["actor"]["layers"]) + 1
    actor_grads = np.array(
        [
            np.hstack([x.numpy().flatten() for x in np.array(_, dtype=object)[-actor_num_layers:]])
            for _ in agent.actor_loss_grad_history
        ]
    )
    critic_num_layers = len(agent.config["critic"]["layers"]) + 1
    critic_grads = np.array(
        [
            np.hstack([x.numpy().flatten() for x in np.array(_, dtype=object)[-critic_num_layers:]])
            for _ in agent.critic_loss_grad_history
        ]
    )

    # Plot: actor weights
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"time [s]", labelpad=10)
    ax.set_ylabel(r"Actor grads [-]", labelpad=10)

    for i in range(actor_grads.shape[1]):
        ax.plot(task.timevec, actor_grads[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlabel(r"time [s]", labelpad=10)
    ax.set_ylabel(r"Critic grads [-]", labelpad=10)

    for i in range(critic_grads.shape[1]):
        ax.plot(task.timevec, critic_grads[:, i])

    plt.show(block=False)


def plot_weights_and_model(agent, task, zoom_F=None, zoom_G=None, zoom_F_y=0.3, zoom_G_y=0.3):
    """
    Plot IDHP actor/critic weights and imcremental model parameters in one plot
    """

    fig = plt.figure()
    # gs = fig.add_gridspec(nrows=5, ncols=2, height_ratios=[1, 1, 1, 1, 0.2])
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 1])
    fig.set_figwidth(10)
    fig.set_figheight(6)

    obs_labels = agent.env.obs_labels

    # Plot: incremental model F
    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylabel(r"$F$", labelpad=10)

    ax.plot(
        task.timevec,
        [_.flatten() for _ in agent.F_history],
        label=np.array(
            [
                [
                    rf"$\frac{{\partial {obs_labels[i]}}}{{\partial {obs_labels[j]}}}$"
                    for j in range(len(agent.F_history[0]))
                ]
                for i in range(len(agent.F_history[0][0]))
            ]
        ).flatten(),
    )
    # ax.legend(bbox_to_anchor=(1, 1), loc=1, ncol=2)

    if zoom_F:
        axins = inset_axes(
            ax, "75%", "60%", loc="upper right", bbox_to_anchor=(0, zoom_F_y, 1, 1), bbox_transform=ax.transAxes
        )
        axins.set_xlim(0, task.T)
        axins.set_ylim(*zoom_F)
        axins.set_xticklabels([])
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0", ls=":")
        axins.plot(task.timevec, [_.flatten() for _ in agent.F_history])

    # Plot: incremental model G
    ax = fig.add_subplot(gs[1, 0])
    ax.set_ylabel(r"$G$", labelpad=10)

    ax.plot(
        task.timevec,
        [_.flatten() for _ in agent.G_history],
        label=np.array(
            [
                [
                    rf"$\frac{{\partial {obs_labels[j]}}}{{\partial {{{agent.env.action_labels[i]}}}}}$"
                    for j in range(len(agent.G_history[0]))
                ]
                for i in range(len(agent.G_history[0][0]))
            ]
        ).flatten(),
    )
    # ax.legend(bbox_to_anchor=(1, 1), loc=1)

    if zoom_G:
        axins = inset_axes(
            ax, "75%", "60%", loc="upper right", bbox_to_anchor=(0, zoom_G_y, 1, 1), bbox_transform=ax.transAxes
        )
        axins.set_xlim(0, task.T)
        axins.set_ylim(*zoom_G)
        axins.set_xticklabels([])
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0", ls=":")
        axins.plot(task.timevec, [_.flatten() for _ in agent.G_history])

    # Plot: incremental model parameter variances
    ax = fig.add_subplot(gs[2, 0])
    ax.set_ylabel(r"$Var$", labelpad=10)

    ax.plot(
        task.timevec,
        [np.diagonal(_) for _ in agent.cov_history],
        label=[rf"${label}$" for label in obs_labels] + [rf"${label}$" for label in agent.env.action_labels],
        # label=obs_labels, agent.env.action_labels,
    )
    ax.set_yscale("log")

    # Plot: incremental model error
    ax = fig.add_subplot(gs[3, 0])
    ax.set_xlabel(r"Time [s]", labelpad=10)
    ax.set_ylabel(r"$|\epsilon|$", labelpad=10)

    ax.plot(
        task.timevec,
        np.abs([_.flatten() for _ in agent.epsilon_history]),
        label=[rf"${label}$" for label in obs_labels],
    )
    for i, eps_thresh in enumerate(agent.model.epsilon_thresh):
        if not np.isinf(eps_thresh):
            ax.axhline(
                y=eps_thresh,
                color=f"C{i}",
                label=rf"${obs_labels[i]}_{{thresh}}$",
                linestyle="--",
            )

    # h, l = ax.get_legend_handles_labels()
    # lax = fig.add_subplot(gs[4, 0])
    # lax.legend(handles=h, labels=l, borderaxespad=0, mode="expand", ncol=4)
    # lax.axis("off")

    ax.set_ylim(bottom=0)
    ax.set_yscale("symlog", linthresh=1.0e-8)

    # Get actor/critic weights
    actor_num_layers = len(agent.config["actor"]["layers"]) + 1
    actor_weights = np.array(
        [
            # np.hstack([x.flatten() for x in np.array(_, dtype=object)[-actor_num_layers:]])
            np.hstack([x.flatten() for x in _[-actor_num_layers:]])
            for _ in agent.actor_weights_history
        ]
    )
    critic_num_layers = len(agent.config["critic"]["layers"]) + 1
    critic_weights = np.array(
        [
            # np.hstack([x.flatten() for x in np.array(_, dtype=object)[-critic_num_layers:]])
            np.hstack([x.flatten() for x in _[-critic_num_layers:]])
            for _ in agent.critic_weights_history
        ]
    )

    # Plot: actor weights
    actor_weights_1 = actor_weights[:, np.max(actor_weights, axis=0) > 0.5]

    j = 0
    ax = fig.add_subplot(gs[j, 1])
    ax.set_ylabel(r"Actor weights $\sim 1$", labelpad=10)

    for i in range(actor_weights_1.shape[1]):
        ax.plot(task.timevec, actor_weights_1[:, i])

    actor_weights_0 = actor_weights[:, np.max(actor_weights, axis=0) <= 0.5]

    j += 1
    ax = fig.add_subplot(gs[j, 1])
    ax.set_ylabel(r"Actor weights $\sim 0$", labelpad=10)

    for i in range(actor_weights_0.shape[1]):
        ax.plot(task.timevec, actor_weights_0[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[j + 1, 1])
    ax.set_xlabel(r"Time [s]", labelpad=10)
    ax.set_ylabel(r"Critic weights [-]", labelpad=10)

    for i in range(critic_weights.shape[1]):
        ax.plot(task.timevec, critic_weights[:, i])

    plt.show(block=False)


def plot_weights_idhp(agent, task, identity=True):
    """
    Plot the weights of actor and critic vs time
    """

    fig = plt.figure()
    if identity:
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 2])
    else:
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
    # fig.set_figwidth(8.27)
    fig.set_figwidth(8.27 / 2)
    fig.set_figheight(5)

    actor_num_layers = len(agent.config["actor"]["layers"]) + 1
    actor_weights = np.array(
        [
            # np.hstack([x.flatten() for x in np.array(_, dtype=object)[-actor_num_layers:]])
            np.hstack([x.flatten() for x in _[-actor_num_layers:]])
            for _ in agent.actor_weights_history
        ]
    )
    critic_num_layers = len(agent.config["critic"]["layers"]) + 1
    critic_weights = np.array(
        [
            # np.hstack([x.flatten() for x in np.array(_, dtype=object)[-critic_num_layers:]])
            np.hstack([x.flatten() for x in _[-critic_num_layers:]])
            for _ in agent.critic_weights_history
        ]
    )

    # Plot: actor weights
    j = 0
    if identity:
        ax = fig.add_subplot(gs[j, 0])
        # ax.set_xlabel(r"time [s]", labelpad=10)
        ax.set_ylabel(r"Actor weights [-]", labelpad=10)
        # ax.set_ylim(1 - 0.0025, 1 + 0.0025)
        # ax.set_ylim(1 - 0.025, 1 + 0.025)
        ax.set_ylim(1 - 0.05, 1 + 0.05)

        for i in range(actor_weights.shape[1]):
            ax.plot(task.timevec, actor_weights[:, i])

        j += 1
        ax = fig.add_subplot(gs[j, 0])
        # ax.set_xlabel(r"time [s]", labelpad=10)
        ax.set_ylabel(r"Actor weights [-]", labelpad=10)
        # ax.set_ylim(0 - 0.05, 0 + 0.05)
        ax.set_ylim(0 - 0.07, 0 + 0.07)

        for i in range(actor_weights.shape[1]):
            ax.plot(task.timevec, actor_weights[:, i])
    else:
        ax = fig.add_subplot(gs[j, 0])
        ax.set_xlabel(r"time [s]", labelpad=10)
        ax.set_ylabel(r"Actor weights [-]", labelpad=10)

        for i in range(actor_weights.shape[1]):
            ax.plot(task.timevec, actor_weights[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[j + 1, 0])
    ax.set_xlabel(r"Time [s]", labelpad=10)
    ax.set_ylabel(r"Critic weights [-]", labelpad=10)

    for i in range(critic_weights.shape[1]):
        ax.plot(task.timevec, critic_weights[:, i])

    plt.show(block=False)


def plot_weights_idhp_split(agent, task):
    """
    Plot the weights of actor and critic vs time
    """

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 1])
    fig.set_figwidth(8.27)
    fig.set_figheight(4)

    actor_weights_l1 = np.array([_[-2].flatten() for _ in agent.actor_weights_history])
    actor_weights_l2 = np.array([_[-1].flatten() for _ in agent.actor_weights_history])

    critic_weights_l1 = np.array([_[-2].flatten() for _ in agent.critic_weights_history])
    critic_weights_l2 = np.array([_[-1].flatten() for _ in agent.critic_weights_history])

    # Plot: actor weights: layer 1
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"time [s]", labelpad=10)
    ax.set_ylabel(r"Actor weights: layer 1 [-]", labelpad=10)

    for i in range(actor_weights_l1.shape[1]):
        ax.plot(task.timevec, actor_weights_l1[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlabel(r"time [s]", labelpad=10)
    ax.set_ylabel(r"Critic weights: layer 1 [-]", labelpad=10)

    for i in range(critic_weights_l1.shape[1]):
        ax.plot(task.timevec, critic_weights_l1[:, i])

    # Plot: actor weights: layer 2
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlabel(r"time [s]", labelpad=10)
    ax.set_ylabel(r"Actor weights: layer 2 [-]", labelpad=10)

    for i in range(actor_weights_l2.shape[1]):
        ax.plot(task.timevec, actor_weights_l2[:, i])

    # Plot: critic weights
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlabel(r"time [s]", labelpad=10)
    ax.set_ylabel(r"Critic weights: layer 2 [-]", labelpad=10)

    for i in range(critic_weights_l2.shape[1]):
        ax.plot(task.timevec, critic_weights_l2[:, i])

    plt.show(block=False)


def plot_incremental_model(agent, task):
    """
    Plot the incremental model metrics of IDHP with shortperiod model
    """

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[1, 1, 1, 1, 0.2])
    # fig.set_figwidth(8.27)
    fig.set_figwidth(8.27 / 2)
    fig.set_figheight(7)

    obs_labels = agent.env.obs_labels

    # Plot: incremental model F
    ax = fig.add_subplot(gs[0, :])
    ax.set_ylabel(r"$F$", labelpad=10)

    ax.plot(
        task.timevec,
        [_.flatten() for _ in agent.F_history],
        label=np.array(
            [
                [
                    rf"$\frac{{\partial {obs_labels[i]}}}{{\partial {obs_labels[j]}}}$"
                    for j in range(len(agent.F_history[0]))
                ]
                for i in range(len(agent.F_history[0][0]))
            ]
        ).flatten(),
    )
    # ax.legend(bbox_to_anchor=(1, 1), loc=1, ncol=2)

    # Plot: incremental model G
    ax = fig.add_subplot(gs[1, :])
    ax.set_ylabel(r"$G$", labelpad=10)

    ax.plot(
        task.timevec,
        [_.flatten() for _ in agent.G_history],
        label=np.array(
            [
                [
                    rf"$\frac{{\partial {obs_labels[j]}}}{{\partial {{{agent.env.action_labels[i]}}}}}$"
                    for j in range(len(agent.G_history[0]))
                ]
                for i in range(len(agent.G_history[0][0]))
            ]
        ).flatten(),
    )
    # ax.legend(bbox_to_anchor=(1, 1), loc=1)

    # Plot: incremental model parameter variances
    ax = fig.add_subplot(gs[2, :])
    ax.set_ylabel(r"$Var$", labelpad=10)

    ax.plot(
        task.timevec,
        [np.diagonal(_) for _ in agent.cov_history],
        label=[rf"${label}$" for label in obs_labels] + [rf"${label}$" for label in agent.env.action_labels],
        # label=obs_labels, agent.env.action_labels,
    )
    ax.set_yscale("log")

    # h, l = ax.get_legend_handles_labels()
    # lax = fig.add_subplot(gs[2, :])
    # lax.legend(handles=h, labels=l, borderaxespad=0, mode="expand", ncol=11)
    # lax.axis("off")

    # Plot: incremental model error
    ax = fig.add_subplot(gs[3, :])
    ax.set_xlabel(r"Time [s]", labelpad=10)
    ax.set_ylabel(r"$|\epsilon|$", labelpad=10)

    ax.plot(
        task.timevec,
        np.abs([_.flatten() for _ in agent.epsilon_history]),
        label=[rf"${label}$" for label in obs_labels],
    )
    for i, eps_thresh in enumerate(agent.model.epsilon_thresh):
        if not np.isinf(eps_thresh):
            ax.axhline(
                y=eps_thresh,
                color=f"C{i}",
                label=rf"${obs_labels[i]}_{{thresh}}$",
                linestyle="--",
            )

    h, l = ax.get_legend_handles_labels()
    lax = fig.add_subplot(gs[4, :])
    lax.legend(handles=h, labels=l, borderaxespad=0, mode="expand", ncol=4)
    lax.axis("off")

    ax.set_ylim(bottom=0)
    ax.set_yscale("symlog", linthresh=1.0e-8)

    plt.show(block=False)


def plot_inputs(agent):
    """
    Plot the inputs of the networks
    """

    s_history = np.array([_.flatten() for _ in agent.s_history])
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=1)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylabel(r"Actor/Critic input $s$")
    ax.plot(agent.task.timevec, s_history)

    plt.show(block=False)
