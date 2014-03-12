# -*- coding: utf-8 -*-
import numpy as np
from skimage import morphology
from skimage.filter import sobel

# --------------------------------------------------------------------

def _region_segment(args):
    """
    Region based segmentation.
    """
    data, args, ind_start, ind_end = args
    low, high = args
    
    for m in range(ind_end-ind_start):
        img = data[m, :, :]
        elevation_map = sobel(img)

        markers = np.zeros_like(img)
        markers[img < low] = 1
        markers[img > high] = 2

        img = morphology.watershed(elevation_map, markers)
        data[m, :, :] = img
    return ind_start, ind_end, data