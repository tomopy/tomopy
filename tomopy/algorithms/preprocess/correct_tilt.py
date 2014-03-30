# -*- coding: utf-8 -*-
from scipy.ndimage.interpolation import rotate

# --------------------------------------------------------------------

def correct_tilt(data, angle):
    """
    Correct for tilt of the raw projection data. 

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    angle : scalar
        angle of rotation applied to data, data_white and data_dark data. 

    Returns
    -------
    data : ndarray
        rotated data.
        
    Examples
    --------
    - Rotate by 0.5 deg:
        
        >>> import tomopy
        >>> 
        >>> # Load sinogram
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile)
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> 
        >>> # Perform rotation
        >>> d.rotate(0.5)
        >>> 
        >>> # Save sinogram after rotation
        >>> output_file='tmp/after_rotation_'
        >>> tomopy.xtomo_writer(d.data, output_file)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    num_projections = data.shape[0]
    
    
    for m in range(num_projections):
        data[m, :, :] = rotate(data[m, :, :], angle, reshape=False)
        
    return data
