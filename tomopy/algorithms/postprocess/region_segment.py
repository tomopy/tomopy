# -*- coding: utf-8 -*-
import numpy as np
from skimage import morphology
from skimage.filter import sobel

# --------------------------------------------------------------------

def region_segment(args):
    """
    Applies an region-based segementation to reconstructed data.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]
        
    low : scalar, int
       Lowest value for the marker.
        
    high : scalar, int
       Higest value for the marker.
         
    Returns
    -------
    output : ndarray
        Segmented data.
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