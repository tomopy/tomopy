# -*- coding: utf-8 -*-
from skimage.filter import threshold_otsu

# --------------------------------------------------------------------

def _threshold_segment(args):
    """
    Threshold based segmentation.
    """
    data, args, ind_start, ind_end = args
    cutoff = args
    
    for m in range(ind_end-ind_start):
        img = data[m, :, :]
        if cutoff == None:
            cutoff = threshold_otsu(img)
        img = img > cutoff
        data[m, :, :] = img
    return ind_start, ind_end, data