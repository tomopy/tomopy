# -*- coding: utf-8 -*-
import numpy as np
from skimage import io as skimage_io 
import warnings

from gridrec import Gridrec

# --------------------------------------------------------------------

def diagnose_center(data, theta, dir_path, slice_no, 
                    center_start, center_end, center_step, 
                    mask, ratio, dtype, data_min, data_max):
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
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]
        
    theta : ndarray, float32
        Projection angles.
        
    dir_path : str
        Directory to save output images.
    
    slice_no : scalar
        The index of the slice to be used for diagnostics.
    
    center_start, center_end, center_step : scalar
        Values of the start, end and step of the center values to
        be used for diagnostics.
        
    mask : bool
        If ``True`` applies a circular mask to the image.
        
    ratio : scalar
        The ratio of the radius of the circular mask to the
        edge of the reconstructed image.
        
    dtype : bool, optional
        Export data type precision.
        
    data_min, data_max : scalar, optional
        User defined minimum and maximum values
        of the reconstructions that will be used to scale 
        the images when saving.
        
    Examples
    --------
    - Finding rotation center by visual inspection:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile)
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> d.normalize()
        >>> 
        >>> # Perform reconstructions with different centers.
        >>> d.diagnose_center(center_start=640, center_end=670)
        >>> print "Images are succesfully saved at tmp/"
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
    
    # Apply circular mask.
    if mask is True:
        rad = num_pixels/2
        y, x = np.ogrid[-rad:rad, -rad:rad]
        msk = x*x + y*y > ratio*ratio*rad*rad
        for m in range(center.size):
            recon.data_recon[m, msk] = 0
        
    # Find max min of data for scaling
    if data_max is None:
        data_max = np.max(recon.data_recon)
    if data_min is None:
        data_min = np.min(recon.data_recon)
        
    if data_max < np.max(recon.data_recon):
        recon.data_recon[recon.data_recon>data_max] = data_max
    if data_min > np.min(recon.data_recon):
        recon.data_recon[recon.data_recon<data_min] = data_min

    # Save it to a temporary directory for manual inspection.
    for m in range(center.size):
        if m % 2 == 0: # 2 slices same bec of gridrec.
            file_name = dir_path + str(np.squeeze(center[m])) + ".tif"
            arr = recon.data_recon[m, :, :]
            print data_min, data_max
            print np.min(arr), np.max(arr)
            if dtype is 'uint8':
                arr = ((arr*1.0 - data_min)/(data_max-data_min)*255).astype('uint8')
            elif dtype is 'uint16':
                arr = ((arr*1.0 - data_min)/(data_max-data_min)*65535).astype('uint16')
            elif dtype is 'float32':
                arr = ((arr*1.0 - data_min)/(data_max-data_min)).astype('float32')
            print np.min(arr), np.max(arr)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage_io.imsave(file_name, arr, plugin='tifffile')
    
    return dtype, data_max, data_min