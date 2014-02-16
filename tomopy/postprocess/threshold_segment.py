# -*- coding: utf-8 -*-
import numpy as np
from skimage.filter import threshold_otsu
from tomopy.tools.multiprocess import worker


@worker
def threshold_segment(args):
    """
    Threshold based segmentation.
    """
    data, args, ind_start, ind_end = args
    cutoff = args
    
    for m in range(ind_end-ind_start):
        img = data[m, :, :]
        cutoff = threshold_otsu(img)
        img = img > cutoff
        data[m, :, :] = img
    return ind_start, ind_end, data