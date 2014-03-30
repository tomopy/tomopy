# -*- coding: utf-8 -*-
import numpy as np

# --------------------------------------------------------------------

def circular_roi(data, ratio):
    """
    Apply circular mask to projection data.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [projections, slices, pixels]
        
    ratio : scalar, int
        Ratio of the circular mask's diameter in pixels
        to the number of reconstructed image pixels 
        (i.e., the dimension of the images).
         
    Returns
    -------
    output : ndarray
        Masked data.
    """
    num_projections = data.shape[0]
    num_slices = data.shape[1]
    num_pixels = data.shape[2]
    
    if num_slices < num_pixels:
        ind1 = num_slices
        ind2 = num_pixels
    else:        
        ind1 = num_pixels
        ind2 = num_slices

    # Apply circular mask.
    rad1 = ind1/2
    rad2 = ind2/2
    y, x = np.ogrid[-rad1:rad1, -rad2:rad2]
    mask = x*x + y*y > ratio*ratio*rad1*rad2
    for m in range(num_projections):
        data[m, mask] = np.mean(data[m, ~mask])
        
    return data