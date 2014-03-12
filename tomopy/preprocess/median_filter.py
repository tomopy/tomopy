# -*- coding: utf-8 -*-
from scipy.ndimage import filters

# --------------------------------------------------------------------

def _median_filter(args):
    """
    Apply median filter to data.

    Parameters
    ----------
    data : ndarray
        Projection data.
        
    size : scalar
        The size of the filter. 

    Returns
    -------
    data : ndarray
        Median filtered data.
    """
    data, args, ind_start, ind_end = args
    size = args
    
    for m in range(ind_end-ind_start):
        data[:, m, :] = filters.median_filter(data[:, m, :], (1, size))
    return ind_start, ind_end, data