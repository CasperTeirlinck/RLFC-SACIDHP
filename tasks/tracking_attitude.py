import numpy as np
import random

from tools import d2r
from tasks import TrackingTask


class TrackAttitude(TrackingTask):
    """
    Tracking attitude: pitch and roll angle with zero sideslip
    """

    def __str__(self):
        return "tracking_attitude"

    def __init__(self, config, evaluate=False, evaluate_tr=False, evaluate_hard=False, train_online=False):
        super().__init__(config)

        self.evaluate = evaluate
        self.evaluate_hard = evaluate_hard
        self.evaluate_tr = evaluate_tr
        self.train_online = train_online
        self.set_tracking_ref()
        self.set_tracking_scale()
        self.set_tracking_max()
        self.set_tracking_range()

    def set_tracking_ref(self):

        """
        Pitch angle reference signal
        """

        if self.train_online:  # online IDHP training task
            A_theta = d2r(5.0)  # [rad]
            # f_theta = 2 * np.pi * 0.2  # [rad/s]
            f_theta = 2 * np.pi * 0.15  # [rad/s]

            self.tracking_ref["theta"] = A_theta * np.sin(f_theta * self.timevec)

        elif self.evaluate_tr:  # Training callback evaluate
            A_theta = d2r(20)

            self.tracking_ref["theta"] = A_theta * self.cosstep(-1, 2) * 2 - A_theta
            self.tracking_ref["theta"] -= A_theta * self.cosstep(self.T * 0.25, 1)
            self.tracking_ref["theta"] -= A_theta * self.cosstep(self.T * 0.50, 1)
            self.tracking_ref["theta"] += A_theta * self.cosstep(self.T * 0.75, 1)

        elif self.evaluate:  # Evaluate
            A_theta = 20 * np.pi / 180  # [rad]
            self.tracking_ref["theta"] = A_theta * self.cosstep(-1, 2) * 2 - A_theta
            self.tracking_ref["theta"] -= A_theta * self.cosstep(self.T * 0.25, 1)
            self.tracking_ref["theta"] -= A_theta * self.cosstep(self.T * 0.50, 1)
            self.tracking_ref["theta"] += A_theta * self.cosstep(self.T * 0.75, 1)

        elif self.evaluate_hard:  # Evaluate, dr stuck failure case
            self.tracking_ref["theta"] = d2r(8) * np.ones_like(self.timevec)

        else:  # Training
            A_theta = random.choice([20, 10, -10, -20]) * np.pi / 180  # [rad]
            # A_theta = 20 * np.pi / 180  # [rad]

            self.tracking_ref["theta"] = A_theta * self.cosstep(-1, 2) * 2 - A_theta
            self.tracking_ref["theta"] -= A_theta * self.cosstep(self.T * 0.25, 1)
            self.tracking_ref["theta"] -= A_theta * self.cosstep(self.T * 0.50, 1)
            self.tracking_ref["theta"] += A_theta * self.cosstep(self.T * 0.75, 1)

        """
        Roll angle reference signal
        """
        if self.train_online:  # online IDHP training task
            self.tracking_ref["phi"] = A_theta * np.sin(f_theta * self.timevec + np.pi / 2)  # uncorrelated with theta

        elif self.evaluate_tr:  # Training callback evaluate
            A_phi = d2r(40)

            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60, 1)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85, 1)

        elif self.evaluate:  # Evaluate
            A_phi = 40 * np.pi / 180  # [rad]
            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60, 2)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85, 2)

        elif self.evaluate_hard:  # Evaluate, dr stuck failure case
            A_phi = 20 * np.pi / 180  # [rad]
            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60, 2)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85, 2)

        else:  # Training
            A_phi = random.choice([40, 20, -20, -40]) * np.pi / 180  # [rad]
            # A_phi = -40 * np.pi / 180  # [rad]

            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60, 1)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85, 1)

        """
        Sideslip angle reference signal
        """
        if self.train_online:  # online IDHP training task
            self.tracking_ref["beta"] = A_theta * np.exp(-self.timevec / 5.0) * np.sin(f_theta * self.timevec)
        else:  # Other
            self.tracking_ref["beta"] = 0 * self.timevec

    def set_tracking_scale(self):

        self.tracking_scale["theta"] = self.config["tracking_scale"]["theta"]
        self.tracking_scale["phi"] = self.config["tracking_scale"]["phi"]
        self.tracking_scale["beta"] = self.config["tracking_scale"]["beta"]

    def set_tracking_max(self):
        self.tracking_max["theta"] = d2r(10.0)
        self.tracking_max["phi"] = d2r(10.0)
        self.tracking_max["beta"] = d2r(2.0)

    def set_tracking_range(self):

        range_theta = self.tracking_ref["theta"].max() - self.tracking_ref["theta"].min()
        range_phi = self.tracking_ref["phi"].max() - self.tracking_ref["phi"].min()

        self.tracking_range["theta"] = range_theta if range_theta > 0 else d2r(5.0) - d2r(-5.0)
        self.tracking_range["phi"] = range_phi if range_theta > 0 else d2r(5.0) - d2r(-5.0)
        self.tracking_range["beta"] = d2r(5.0) - d2r(-5.0)
