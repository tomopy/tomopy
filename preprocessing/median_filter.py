# -*- coding: utf-8 -*-
# Filename: median_filter.py
from scipy import ndimage
import numpy as np

def median_filter(data, axis=1, size=(1, 3)):
    """ Apply median filter to data.

    Parameters
    ----------
    data : ndarray
        Input data.

    axis : scalar, optional
        Specifies the axis that for filtering.
        0: slices-pixels plane
        1: projections-pixels plane
        2: projections-slices plans

    size : array-like, optional
    The size of the filter.

    Returns
    -------
    data : ndarray
        Output processed data.
    """
    print "Applying median filter to data..."

    # Override medianaxis if one dimension is null.
    if data.shape[0] == 1:
        axis = 0
    elif data.shape[1] == 1:
        axis = 1
    elif data.shape[2] == 1:
        axis = 2

    if axis is 0:
        for m in range(data.shape[0]):
            data[m, :, :] = ndimage.filters.median_filter(
                                    np.squeeze(data[m, :, :]), size=size)
    elif axis is 1:
        for m in range(data.shape[1]):
            data[:, m, :] = ndimage.filters.median_filter(
                                    np.squeeze(data[:, m, :]), size=size)
    elif axis is 2:
        for m in range(data.shape[2]):
            data[:, :, m] = ndimage.filters.median_filter(
                                    np.squeeze(data[:, :, m]), size=size)
    else: raise ValueError('Check median filter axes.')
    return data
