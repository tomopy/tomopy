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

def zinger_filter(data, cutoff=2):
    """ Apply zinger removal filter to data.

    Parameters
    ----------
    data : ndarray
        Input data (normalized).

    cutoff : scl, optional
        Permitted maximum vaue of the
        normalized data. Zinger otherwise.

    Returns
    -------
    data : ndarray
        Output processed data.
    """
    for m in range(data.shape[0]):
        zinger_mask = data[m, :, :] > cutoff
        print np.sum(zinger_mask)
        tmp = ndimage.filters.median_filter(data[m, :, :], size=3)
        data[m, zinger_mask] = 100
    return data
