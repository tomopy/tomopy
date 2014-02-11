# -*- coding: utf-8 -*-
import numpy as np


def normalize(args):
    """
    Normalize raw projection data with
    the white field projection data.

    Parameters
    ----------
    data : ndarray
        Raw projection data.

    data_white : ndarray
        2-D white field projection data.

    cutoff : scalar
        Permitted maximum vaue of the
        normalized data.

    Returns
    -------
    data : ndarray
        Normalized data.
    """
    data, data_white, cutoff = args
    for m in range(data.shape[0]):
        data[m, :, :] = np.divide(data[m, :, :], data_white)
    if cutoff is not None:
        data[data > cutoff] = cutoff
    return data