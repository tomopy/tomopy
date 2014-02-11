# -*- coding: utf-8 -*-
from scipy.ndimage import filters


def median_filter(args):
    """
    Apply median filter to data.

    Parameters
    ----------
    data : ndarray
        Projection data.
        
    size : scalar or tuple
        The size of the filter. 

    Returns
    -------
    data : ndarray
        Median filtered data.
    """
    data, size = args
    for m in range(data.shape[2]):
        data[:, :, m] = filters.median_filter(data[:, :, m], size)
    return data