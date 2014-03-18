# -*- coding: utf-8 -*-
import numpy as np

# --------------------------------------------------------------------

def _apply_mask(data, ratio):
    """
    Apply circular mask to reconstructed data.
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