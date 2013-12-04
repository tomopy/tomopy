# -*- coding: utf-8 -*-
# Filename: normalize.py
import numpy as np

def normalize(data, white, cutoff=None):
    """ Normalize using average white field images.

    Parameters:
    -----------
    data : ndarray
        Input data.

    white : ndarray
        White field data.

    cutoff : scl, optional
        Permitted maximum vaue of the
        normalized data.

    Returns
    -------
    data : ndarray
        Output processed data.
    """
    print "Normalizing data..."
    avg_white = np.mean(white, axis=0)
    for m in range(data.shape[0]):
        data[m, :, :] = np.divide(data[m, :, :], avg_white)

    if cutoff is not None:
        data.data[data.data > cutoff] = cutoff
    return data
