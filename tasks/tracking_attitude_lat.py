import numpy as np
import random

from tools import d2r
from tasks import TrackingTask


class TrackAttitudeLat(TrackingTask):
    """
    Tracking attitude lat: roll angle with zero pitch and sideslip
    """

    def __str__(self):
        return "tracking_attitude_lat"

    def __init__(self, config, evaluate=False):
        super().__init__(config)

        self.evaluate = evaluate
        self.set_tracking_ref()
        self.set_tracking_scale()
        self.set_tracking_range()

    def set_tracking_ref(self):

        # Pitch angle reference signal: zero
        self.tracking_ref["theta"] = 0 * self.timevec

        # Roll angle reference signal: sinusodial
        A_phi = d2r(5.0)  # [rad]
        f_phi = 2 * np.pi * 0.2  # [rad/s]

        self.tracking_ref["phi"] = A_phi * np.sin(f_phi * self.timevec)

        # Sideslip angle reference signal: zero
        # self.tracking_ref["beta"] = 0 * self.timevec
        tc = 5.0
        self.tracking_ref["beta"] = A_phi * np.exp(-self.timevec / tc) * np.sin(f_phi * self.timevec)

    def set_tracking_scale(self):

        self.tracking_scale["theta"] = self.config["tracking_scale"]["theta"]
        self.tracking_scale["phi"] = self.config["tracking_scale"]["phi"]
        self.tracking_scale["beta"] = self.config["tracking_scale"]["beta"]

    def set_tracking_range(self):

        self.tracking_range["theta"] = d2r(5.0) - d2r(-5.0)
        self.tracking_range["phi"] = self.tracking_ref["phi"].max() - self.tracking_ref["phi"].min()
        # self.tracking_range["beta"] = d2r(5.0) - d2r(-5.0)
        self.tracking_range["beta"] = self.tracking_ref["beta"].max() - self.tracking_ref["beta"].min()
