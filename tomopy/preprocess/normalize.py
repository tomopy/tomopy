# -*- coding: utf-8 -*-
import numpy as np

# --------------------------------------------------------------------

def _normalize(args):
    """
    Normalize raw projection data with
    the white field projection data.

    Parameters
    ----------
    data : ndarray
        Raw projection data.

    data_white : ndarray
        2-D white field projection data.
        
    data_dark : ndarray
        2-D dark field projection data.

    cutoff : scalar
        Permitted maximum vaue of the
        normalized data. 

    Returns
    -------
    data : ndarray
        Normalized data.
    """
    data, args, ind_start, ind_end = args
    data_white, data_dark, cutoff = args
    

    for m in range(ind_end-ind_start):
        data[m, :, :] = np.divide(data[m, :, :]-data_dark, 
                                  data_white-data_dark)
    if cutoff is not None:
        data[data > cutoff] = cutoff
    return ind_start, ind_end, data