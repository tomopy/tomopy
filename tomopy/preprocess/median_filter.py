# -*- coding: utf-8 -*-
from scipy.ndimage import filters
from tomopy.tools.multiprocess import worker


@worker
def median_filter(args):
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
    data, size, id = args
    
    data = filters.median_filter(data, (size, 1))
    return id, data