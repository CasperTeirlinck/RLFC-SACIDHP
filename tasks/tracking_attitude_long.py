import numpy as np
import random

from tools import d2r
from tasks import TrackingTask


class TrackAttitudeLong(TrackingTask):
    """
    Tracking attitude long: pitch angle with zero roll and sideslip
    """

    def __str__(self):
        return "tracking_attitude_long"

    def __init__(self, config, evaluate=False):
        super().__init__(config)

        self.evaluate = evaluate
        self.set_tracking_ref()
        self.set_tracking_scale()
        self.set_tracking_range()

    def set_tracking_ref(self):

        # Pitch angle reference signal: sinusodial
        if not self.evaluate:
            A_theta = d2r(5.0)  # [rad]
            f_theta = 2 * np.pi * 0.2  # [rad/s]

            self.tracking_ref["theta"] = A_theta * np.sin(f_theta * self.timevec)
        else:
            A_theta = 20 * np.pi / 180  # [rad]

            self.tracking_ref["theta"] = A_theta * self.cosstep(-5, 5 * 2) * 2 - A_theta
            self.tracking_ref["theta"] -= A_theta * 0.5 * self.cosstep(self.T * 0.1625, 2)
            self.tracking_ref["theta"] -= A_theta * 0.5 * self.cosstep(self.T * 0.375, 2)
            self.tracking_ref["theta"] -= A_theta * 0.75 * self.cosstep(self.T * 0.7125, 2)
            self.tracking_ref["theta"] += A_theta * 0.75 * self.cosstep(self.T * 0.8875, 4)

        # Roll angle reference signal: zero
        self.tracking_ref["phi"] = 0 * self.timevec

        # Sideslip angle reference signal: zero
        self.tracking_ref["beta"] = 0 * self.timevec

    def set_tracking_scale(self):

        self.tracking_scale["theta"] = self.config["tracking_scale"]["theta"]
        self.tracking_scale["phi"] = self.config["tracking_scale"]["phi"]
        self.tracking_scale["beta"] = self.config["tracking_scale"]["beta"]

    def set_tracking_range(self):

        self.tracking_range["theta"] = self.tracking_ref["theta"].max() - self.tracking_ref["theta"].min()
        self.tracking_range["phi"] = d2r(5.0) - d2r(-5.0)
        self.tracking_range["beta"] = d2r(5.0) - d2r(-5.0)
