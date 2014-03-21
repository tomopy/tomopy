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
        
    Examples
    --------
    - Apply mask after reconstruction:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile,  slices_start=0, slices_end=1)
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> d.normalize()
        >>> 
        >>> # Reconstruct data
        >>> d.center=661.5
        >>> d.gridrec()
        >>> 
        >>> # Save reconstructed data before masking
        >>> output_file='tmp/before_masking_'
        >>> tomopy.xtomo_writer(d.data_recon, output_file)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>> 
        >>> # Apply masking
        >>> d.apply_mask()
        >>> 
        >>> # Save reconstructed data after masking
        >>> output_file='tmp/after_masking_'
        >>> tomopy.xtomo_writer(d.data_recon, output_file)
        >>> print "Images are succesfully saved at " + output_file + '...'
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