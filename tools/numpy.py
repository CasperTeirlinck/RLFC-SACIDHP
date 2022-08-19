import numpy as np


def array32(*args, **kwargs):
    """
    Numpy array overload for default float32
    """

    kwargs.setdefault("dtype", np.float32)

    return np.array(*args, **kwargs)


def zeros32(*args, **kwargs):
    """
    Numpy zeros overload for default float32
    """

    kwargs.setdefault("dtype", np.float32)

    return np.zeros(*args, **kwargs)


def ones32(*args, **kwargs):
    """
    Numpy ones overload for default float32
    """

    kwargs.setdefault("dtype", np.float32)

    return np.ones(*args, **kwargs)


def arange32(*args, **kwargs):
    """
    Numpy arange overload for default float32
    """

    kwargs.setdefault("dtype", np.float32)

    return np.arange(*args, **kwargs)


def identity32(*args, **kwargs):
    """
    Numpy identity overload for default float32
    """

    kwargs.setdefault("dtype", np.float32)

    return np.identity(*args, **kwargs)
