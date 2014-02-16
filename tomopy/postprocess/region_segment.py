# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.filter import sobel
from tomopy.tools.multiprocess import worker


@worker
def region_segment(args):
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