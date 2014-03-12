# -*- coding: utf-8 -*-
import numpy as np
from scipy import misc

from gridrec import Gridrec

# --------------------------------------------------------------------

def _diagnose_center(data, theta, dir_path, slice_no, 
                     center_start, center_end, center_step):
    """ 
    Diagnostic tools to find rotation center.
    
    Helps finding the rotation center manually by
    visual inspection of the selected reconstructions
    with different centers. The outputs for different
    centers are put into ``data/diagnose`` directory
    and the corresponding center positions are printed
    so that one can skim through the images and
    select the best.
    
    Parameters
    ----------
    data : ndarray
        Input data.
    
    slice_no : scalar, optional
        The index of the slice to be used for diagnostics.
    
    center_start, center_end, center_step : scalar, optional
        Values of the start, end and step of the center values to
        be used for diagnostics.
    """
    num_projections =  data.shape[0]
    num_pixels =  data.shape[2]
    
    # Don't ask why. Just do that.
    center_step /= 2.
    
    # Make preperations for the slices and corresponding centers.
    slice_data = data[:, slice_no, :]
    center = np.arange(center_start, center_end, center_step, dtype=np.float32)
    num_center = center.size
    stacked_slices = np.zeros((num_projections, num_center, num_pixels),
                              dtype=np.float32)
                              
    for m in range(num_center):
        stacked_slices[:, m, :] = slice_data

    # Reconstruct the same slice with different centers.
    recon = Gridrec(stacked_slices)
    recon.reconstruct(stacked_slices, theta=theta, center=center)

    # Save it to a temporary directory for manual inspection.
    for m in range(center.size):
        if m % 2 == 0: # 2 slices same bec of gridrec.
            img = misc.toimage(recon.data_recon[m, :, :])
            file_name = dir_path + str(np.squeeze(center[m])) + ".tif"
            img.save(file_name)







