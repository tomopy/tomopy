# -*- coding: utf-8 -*-
import numpy as np


def normalize(data, data_white, cutoff=None):
    """Normalize raw projection data with 
    the white field projection data.

    Parameters
    ----------
    data : ndarray
        Raw projection data.

    data_white : ndarray
        2-D white field projection data.

    cutoff : scalar, optional
        Permitted maximum vaue of the
        normalized data.

    Returns
    -------
    data : ndarray
        Normalized data.
    """
    data = np.divide(data, data_white)
    if cutoff is not None:
        data[data > cutoff] = cutoff
    return data
