from abc import ABC, abstractmethod
import copy
import numpy as np


class TrackingTask(ABC):
    """
    Base class for tracking task/episode
    """

    def __init__(self, config):

        # Time vector
        self.config = config
        self.T = config["T"]
        self.dt = config["dt"]
        self.timevec = np.arange(0, self.T, self.dt)
        self.num_timesteps = len(self.timevec)

        # Reference signals
        self.tracking_ref = {}
        self.tracking_scale = {}
        self.tracking_max = {}
        self.tracking_range = {}
        self.tracking_thresh = config["tracking_thresh"] if "tracking_thresh" in config.keys() else []

    @abstractmethod
    def set_tracking_ref(self):
        """
        Construct reference tracking signals
        """

        raise NotImplementedError

    @abstractmethod
    def set_tracking_scale(self):
        """
        Define tracking error scaling factors
        """

        raise NotImplementedError

    @abstractmethod
    def set_tracking_range(self):
        """
        Define normal operating ranges
        """

        raise NotImplementedError

    def cosstep(self, t0, w):
        """
        Smooth cosine step function starting at t0 with width w
        """

        t0_idx = np.abs(self.timevec - t0).argmin()
        t1 = t0 + w
        t1_idx = np.abs(self.timevec - t1).argmin()

        a = -(np.cos(1 / w * np.pi * (self.timevec - t0)) - 1) / 2
        if t0 >= 0:
            a[:t0_idx] = 0.0
        a[t1_idx:] = 1.0

        return a

    def step(self, t0, w):
        """
        Linear step function starting at t0 with width w
        """

        t1 = t0 + w

        a = np.piecewise(
            self.timevec,
            [
                self.timevec < t0,
                (self.timevec >= t0) & (self.timevec <= t1),
                self.timevec > t1,
            ],
            [
                0.0,
                lambda t: t / (t1 - t0) + t0 / (t0 - t1),
                1.0,
            ],
        )

        return a


class TrackingTaskCascaded(TrackingTask):
    """
    Base class for cascaded tracking task/episode with external inner and outer reference signals defined
    """

    def __init__(self, config):
        super().__init__(config)

        self.tracking_outer = []
        self.tracking_inner = []

    @property
    def outer(self):
        """
        Returns copy of the task with only outer reference signals
        """

        task_outer = copy.deepcopy(self)
        task_outer.tracking_ref = {
            _: task_outer.tracking_ref[_] for _ in self.tracking_outer if _ in task_outer.tracking_ref
        }
        task_outer.tracking_scale = {
            _: task_outer.tracking_scale[_] for _ in self.tracking_outer if _ in task_outer.tracking_scale
        }
        task_outer.tracking_range = {
            _: task_outer.tracking_range[_] for _ in self.tracking_outer if _ in task_outer.tracking_range
        }
        task_outer.tracking_max = {
            _: task_outer.tracking_max[_] for _ in self.tracking_outer if _ in task_outer.tracking_max
        }

        return task_outer

    @property
    def inner(self):
        """
        Returns copy of the task with only inner reference signals
        """

        task_inner = copy.deepcopy(self)
        task_inner.tracking_ref = {
            _: task_inner.tracking_ref[_] for _ in self.tracking_inner if _ in task_inner.tracking_ref
        }
        task_inner.tracking_scale = {
            _: task_inner.tracking_scale[_] for _ in self.tracking_inner if _ in task_inner.tracking_scale
        }
        task_inner.tracking_range = {
            _: task_inner.tracking_range[_] for _ in self.tracking_inner if _ in task_inner.tracking_range
        }
        task_inner.tracking_max = {
            _: task_inner.tracking_max[_] for _ in self.tracking_inner if _ in task_inner.tracking_max
        }

        return task_inner

    @abstractmethod
    def set_tracking_level(self):
        """
        Define control level per reference signal
        """

        raise NotImplementedError

    @abstractmethod
    def set_tracking_ref(self):
        raise NotImplementedError

    @abstractmethod
    def set_tracking_scale(self):
        raise NotImplementedError

    @abstractmethod
    def set_tracking_range(self):
        raise NotImplementedError


class VerificationTask(TrackingTask):
    """
    Verification task/episode
    """

    def __init__(self, config):
        super().__init__(config)

    def __str__(self):
        return "verification"

    def set_tracking_ref(self):
        pass

    def set_tracking_scale(self):
        pass

    def set_tracking_range(self):
        pass
