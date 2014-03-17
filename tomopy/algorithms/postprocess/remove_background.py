# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import reconstruction

# --------------------------------------------------------------------

def _remove_background(args):
    """
    Remove background from reconstructions.
    """
    data, args, ind_start, ind_end = args
    
    for m in range(ind_end-ind_start):
        img = data[m, :, :]
        
        # first remove background.
        seed = np.copy(img)
        seed[1:-1, 1:-1] = img.min()
        img -= reconstruction(seed, img, method='dilation')
        
        data[m, :, :] = img
    return ind_start, ind_end, data