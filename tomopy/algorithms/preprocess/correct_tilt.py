# -*- coding: utf-8 -*-
from scipy.ndimage.interpolation import rotate

# --------------------------------------------------------------------

def correct_tilt(data, angle):
    """
    Correct for tilt angle.
    """
    num_projections = data.shape[0]
    
    
    for m in range(num_projections):
        data[m, :, :] = rotate(data[m, :, :], angle, reshape=False)
        
    return data