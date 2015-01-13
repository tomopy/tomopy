# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
if os.name == 'nt':
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'lib/librecon.pyd'))
    librecon = ctypes.CDLL(libpath)
else:
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..', 'lib/librecon.so'))
    librecon = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def sirt(data, theta, center, num_grid, iters, init_matrix):
    """
    Applies Simultaneous Iterative Reconstruction Technique (SIRT) 
    to obtain reconstructions.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]
        
    theta : ndarray, float32
        Projection angles in radians.
        
    center : scalar, float32
        Pixel index corresponding to the 
        center of rotation axis.
        
    num_grid : scalar, int32
        Grid size of the econstructed images.
        
    iters : scalar int32
        Number of mlem iterations.
    
    init_matrix : ndarray
       Initial guess for the reconstruction. Its
       shape is the same as the reconstructed data.
       
    Returns
    -------
    output : ndarray
        Reconstructed data with dimensions:
        [slices, num_grid, num_grid]
        
    Examples
    --------
    - Reconstruct using SIRT:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> d.normalize()
        >>> d.correct_drift()
        >>> d.center = 662
        >>> 
        >>> # Perform reconstruction
        >>> d.sirt()
        >>> 
        >>> # Save reconstructed data
        >>> output_file='tmp/recon_'
        >>> tomopy.xtomo_writer(d.data_recon, output_file)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """    
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    librecon.sirt.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon.sirt(data.ctypes.data_as(c_float_p),
                 theta.ctypes.data_as(c_float_p),
                 ctypes.c_float(center),
                 ctypes.c_int(num_projections),
                 ctypes.c_int(num_slices),
                 ctypes.c_int(num_pixels),
                 ctypes.c_int(num_grid),
                 ctypes.c_int(iters),
                 init_matrix.ctypes.data_as(c_float_p))
    return init_matrix