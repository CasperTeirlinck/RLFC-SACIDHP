import numpy as np
import random

from tasks import TrackingTaskCascaded
from tools.utils import d2r


class TrackAltitude(TrackingTaskCascaded):
    """
    Tracking altitude: altitude and roll angle with zero sideslip
    """

    def __str__(self):
        return "tracking_altitude"

    def __init__(self, config, evaluate=False, evaluate_tr=False, evaluate_disturbance=False):
        super().__init__(config)

        self.h_task = 0

        self.evaluate = evaluate
        self.evaluate_tr = evaluate_tr
        self.evaluate_disturbance = evaluate_disturbance
        self.set_tracking_ref()
        self.set_tracking_scale()
        self.set_tracking_max()
        self.set_tracking_range()
        self.set_tracking_level()

    def set_tracking_ref(self):

        # Altitude reference signal: linear step
        if self.evaluate_disturbance:
            h_0 = 2000

            self.tracking_ref["h"] = h_0 * np.ones_like(self.timevec)

        elif self.evaluate_tr:  # Training callback evaluate
            h_0 = 2000
            h_step = 50  # = 5m/s over 10s

            self.tracking_ref["h"] = h_0 + h_step * self.step(0, self.T * 0.5)

        elif not self.evaluate:  # Training
            h_0 = 2000
            h_step = [50, 0, -50]  # = 5m/s over 10s
            h_step = h_step[self.h_task % len(h_step)]  # Successive altitude selection
            self.h_task += 1

            self.tracking_ref["h"] = h_0 + h_step * self.step(0, self.T * 0.5)
        else:  # Evaluate
            h_0 = 2000

            self.tracking_ref["h"] = h_0 + 250 * self.step(5, 50)
            self.tracking_ref["h"] -= 250 * self.step(65, 50)

        # Roll angle reference signal: combined smooth step
        if self.evaluate_disturbance:
            self.tracking_ref["phi"] = 0 * np.ones_like(self.timevec)

        elif self.evaluate_tr:  # Training callback evaluate
            A_phi = d2r(40)  # [rad]

            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60, 1)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85, 1)

        elif not self.evaluate:  # Training
            A_phi = random.choice([40, 20, -20, -40]) * np.pi / 180  # [rad]

            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35, 1)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60, 1)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85, 1)

        else:  # Evaluate
            # A_phi = 40 * np.pi / 180  # [rad]

            # self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10 * 0.5, 2)
            # self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35 * 0.5, 2)
            # self.tracking_ref["phi"] -= A_phi / 2 * self.cosstep(self.T * 0.60 * 0.5, 2)
            # self.tracking_ref["phi"] += A_phi / 2 * self.cosstep(self.T * 0.85 * 0.5, 2)

            # self.tracking_ref["phi"] += A_phi / 2 * self.cosstep(self.T * 0.10 * 0.5 + 60, 2)
            # self.tracking_ref["phi"] -= A_phi / 2 * self.cosstep(self.T * 0.35 * 0.5 + 60, 2)
            # self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60 * 0.5 + 60, 2)
            # self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85 * 0.5 + 60, 2)

            A_phi = 20 * np.pi / 180  # [rad]

            self.tracking_ref["phi"] = A_phi * self.cosstep(self.T * 0.10 * 0.5, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35 * 0.5, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60 * 0.5, 2)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85 * 0.5, 2)

            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.10 * 0.5 + 60, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.35 * 0.5 + 60, 2)
            self.tracking_ref["phi"] -= A_phi * self.cosstep(self.T * 0.60 * 0.5 + 60, 2)
            self.tracking_ref["phi"] += A_phi * self.cosstep(self.T * 0.85 * 0.5 + 60, 2)

        # Sideslip angle reference signal: zero
        self.tracking_ref["beta"] = 0 * self.timevec

    def set_tracking_level(self):

        self.tracking_outer.append("h")

        self.tracking_inner.append("theta")
        self.tracking_inner.append("phi")
        self.tracking_inner.append("beta")

    def set_tracking_scale(self):

        self.tracking_scale["h"] = self.config["tracking_scale"]["h"]

        self.tracking_scale["theta"] = self.config["tracking_scale"]["theta"]
        self.tracking_scale["phi"] = self.config["tracking_scale"]["phi"]
        self.tracking_scale["beta"] = self.config["tracking_scale"]["beta"]

    def set_tracking_max(self):
        self.tracking_max["h"] = 250

        self.tracking_max["theta"] = d2r(20.0)
        self.tracking_max["phi"] = d2r(40.0)
        self.tracking_max["beta"] = d2r(5.0)

    def set_tracking_range(self):

        self.tracking_range["h"] = self.tracking_ref["h"].max() - self.tracking_ref["h"].min()

        self.tracking_range["phi"] = self.tracking_ref["phi"].max() - self.tracking_ref["phi"].min()
        self.tracking_range["beta"] = d2r(5.0) - d2r(-5.0)

    def get_control_disturbance(self):
        """
        Get control disturbance signal as 3211 signal
        """

        disturbance = np.zeros((3, self.num_timesteps))
        # Elevator
        de_t0 = 1  # [s]
        de_T = 1  # [s]
        de_A = d2r(1)  # [rad]
        disturbance[
            0, np.argwhere(self.timevec == de_t0)[0, 0] : np.argwhere(self.timevec == de_t0 + 3 * de_T)[0, 0]
        ] = (0.8 * de_A)
        disturbance[
            0, np.argwhere(self.timevec == de_t0 + 3 * de_T)[0, 0] : np.argwhere(self.timevec == de_t0 + 5 * de_T)[0, 0]
        ] = (-1.2 * de_A)
        disturbance[
            0, np.argwhere(self.timevec == de_t0 + 5 * de_T)[0, 0] : np.argwhere(self.timevec == de_t0 + 6 * de_T)[0, 0]
        ] = (1.1 * de_A)
        disturbance[
            0, np.argwhere(self.timevec == de_t0 + 6 * de_T)[0, 0] : np.argwhere(self.timevec == de_t0 + 7 * de_T)[0, 0]
        ] = (-1.1 * de_A)

        # Aileron
        da_t0 = 11  # [s]
        da_T = 1  # [s]
        da_A = d2r(1)  # [rad]
        disturbance[
            1, np.argwhere(self.timevec == da_t0)[0, 0] : np.argwhere(self.timevec == da_t0 + 3 * da_T)[0, 0]
        ] = (0.8 * da_A)
        disturbance[
            1, np.argwhere(self.timevec == da_t0 + 3 * da_T)[0, 0] : np.argwhere(self.timevec == da_t0 + 5 * da_T)[0, 0]
        ] = (-1.2 * da_A)
        disturbance[
            1, np.argwhere(self.timevec == da_t0 + 5 * da_T)[0, 0] : np.argwhere(self.timevec == da_t0 + 6 * da_T)[0, 0]
        ] = (1.1 * da_A)
        disturbance[
            1, np.argwhere(self.timevec == da_t0 + 6 * da_T)[0, 0] : np.argwhere(self.timevec == da_t0 + 7 * da_T)[0, 0]
        ] = (-1.1 * da_A)

        # Rudder
        dr_t0 = 9  # [s]
        dr_T = 1  # [s]
        dr_A = d2r(1)  # [rad]
        disturbance[
            2, np.argwhere(self.timevec == dr_t0)[0, 0] : np.argwhere(self.timevec == dr_t0 + 3 * dr_T)[0, 0]
        ] = (0.8 * dr_A)
        disturbance[
            2, np.argwhere(self.timevec == dr_t0 + 3 * dr_T)[0, 0] : np.argwhere(self.timevec == dr_t0 + 5 * dr_T)[0, 0]
        ] = (-1.2 * dr_A)
        disturbance[
            2, np.argwhere(self.timevec == dr_t0 + 5 * dr_T)[0, 0] : np.argwhere(self.timevec == dr_t0 + 6 * dr_T)[0, 0]
        ] = (1.1 * dr_A)
        disturbance[
            2, np.argwhere(self.timevec == dr_t0 + 6 * dr_T)[0, 0] : np.argwhere(self.timevec == dr_t0 + 7 * dr_T)[0, 0]
        ] = (-1.1 * dr_A)

        return disturbance
