import numpy as np
import os
import csv
import time
from agents.sac import SAC
from tasks.base import TrackingTaskCascaded

from tools import nMAE


class CallbackSAC:
    """
    Callback for evaluating and saving the controller during training.
    """

    def __init__(self, save_dir="", task_eval=None, env_eval=None, nmae_thresh=None, avg_return_window_size=20):

        # Models
        self.agent: SAC = None
        self.cascaded = False

        # Evaluation
        self.episode_return_history = []
        self.episode_return_avg = 0
        self.episode_return_avg_size = avg_return_window_size

        self.task_eval = task_eval
        self.env_eval = env_eval
        self.return_eval_best = -np.inf
        self.nmae_eval_best = np.inf

        # Counters
        self.n_calls = 0

        # Dir
        self.save_dir = save_dir

        # Logger
        fieldnames = ["timestep", "return", "return_avg", "return_tr", "nMAE", "lr"]
        self.logfile = open(os.path.join(save_dir, "training.csv"), "wt+")
        self.logger = csv.DictWriter(self.logfile, fieldnames=fieldnames)
        self.logger.writeheader()
        self.logfile.flush()

        # Other
        self.t_start = time.time()

    def init_callback(self, agent: SAC):
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """

        self.agent = agent
        self.cascaded = type(agent.task).__bases__[0] == TrackingTaskCascaded
        return self

    def on_done(self, episode_return):
        """
        This method will be called after every training episode
        """

        # Save agent
        save_dir = os.path.join(self.save_dir, str(self.agent.t))
        self.agent.save(save_dir, minimal=True)

        # Evaluate on evaluation task
        if self.task_eval:
            nmae_eval = None
            episode_return_eval = None
            success = False

            env_eval = self.env_eval if self.env_eval else self.agent.env
            agent_eval = SAC.load(save_dir, self.task_eval, env_eval)

            episode_return_eval = agent_eval.evaluate(verbose=False)

            # Save best eval return
            if episode_return_eval:
                if episode_return_eval > self.return_eval_best:
                    self.return_eval_best = episode_return_eval
                    self.agent.save(os.path.join(self.save_dir, "best_eval_r"), minimal=True)
                    _ = open(os.path.join(self.save_dir, "best_eval_r", str(self.agent.t)), "w")
                    _.close()

                # Save best eval nMAE
                nmae_eval = nMAE(agent_eval, agent_eval.env, cascaded=self.cascaded)
                if nmae_eval < self.nmae_eval_best:
                    self.nmae_eval_best = nmae_eval
                    self.agent.save(os.path.join(self.save_dir, "best_eval_nmae"), minimal=True)
                    _ = open(os.path.join(self.save_dir, "best_eval_nmae", str(self.agent.t)), "w")
                    _.close()

            # Update running average
            if episode_return_eval and not np.isnan(episode_return_eval):
                success = True

                self.episode_return_history.append(episode_return_eval)
                self.episode_return_avg = np.mean(self.episode_return_history[-self.episode_return_avg_size :])

            # Log training
            self.logger.writerow(
                {
                    "timestep": self.agent.t,
                    "return": round(episode_return_eval, 6) if episode_return_eval else None,
                    "return_avg": round(self.episode_return_avg, 6),
                    "return_tr": round(episode_return, 6),
                    "nMAE": round(nmae_eval, 2) if nmae_eval else None,
                    "lr": round(float(self.agent.lr), 8),
                }
            )

        # No seperate evaluation run
        else:
            # Save best eval return
            if episode_return > self.return_eval_best:
                self.return_eval_best = episode_return
                self.agent.save(os.path.join(self.save_dir, "best_eval_r"), minimal=True)
                _ = open(os.path.join(self.save_dir, "best_eval_r", str(self.agent.t)), "w")
                _.close()

            # Update running average
            self.episode_return_history.append(episode_return)
            self.episode_return_avg = np.mean(self.episode_return_history[-self.episode_return_avg_size :])

            # Log training
            self.logger.writerow(
                {
                    "timestep": self.agent.t,
                    "return": round(episode_return, 6),
                    "return_avg": round(self.episode_return_avg, 6),
                    "return_tr": None,
                    "nMAE": None,
                    "lr": round(float(self.agent.lr), 8),
                }
            )

            success = True

        self.logfile.flush()

        return success

    def on_crash(self):
        """
        This method will be called after a failed episode
        """

        # Mark save file as crash
        _ = open(os.path.join(self.save_dir, "crash"), "w")
        _.close()
