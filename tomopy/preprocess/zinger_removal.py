# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import filters

# --------------------------------------------------------------------

def _zinger_removal(args):
    """
    Zinger removal.

    Parameters
    ----------
    data : ndarray
        Input data.
        
    zl : scalar
        Threshold of counts to cut zingers.
        
    mw : scalar
        Median filter width.
    """
    data, args, ind_start, ind_end = args
    zl, mw = args

    zinger_mask = np.zeros((1, data.shape[1], data.shape[2]))

    for m in range(ind_end-ind_start):
        tmp_img = filters.median_filter(data[m, :, :],(1, mw))
        zinger_mask = ((data[m, :, :]-tmp_img) >= zl).astype(int)
        data[m,:,:] = tmp_img*zinger_mask + data[m, :, :]*(1-zinger_mask)

    return ind_start, ind_end, data