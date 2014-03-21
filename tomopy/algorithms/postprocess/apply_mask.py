# -*- coding: utf-8 -*-
import numpy as np

# --------------------------------------------------------------------

def apply_mask(data, ratio):
    """
    Apply circular mask to reconstructed data.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]
        
    ratio : scalar, int
        Ratio of the circular mask's diameter in pixels
        to the number of reconstructed image pixels 
        (i.e., the dimension of the images).
         
    Returns
    -------
    output : ndarray
        Masked data.
    """
    num_slices = data.shape[0]
    num_pixels = data.shape[1]

    # Apply circular mask.
    rad = num_pixels/2
    y, x = np.ogrid[-rad:rad, -rad:rad]
    mask = x*x + y*y > ratio*ratio*rad*rad
    for m in range(num_slices):
        data[m, mask] = 0
        
    return data