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

def pml3db(data, theta, center, num_grid, iters, beta, delta, beta1, delta1, regw1, regw2, init_matrix):
    """
    Applies Accelerated Penalized Maximum-Likelihood (APML)
    method to obtain reconstructions. 
    
    It is based on standard decoupled surrogate functions
    for the ML objective function assuming a Poisson model and 
    decoupled surrogate functions for a certain class of 
    penalty functions [1].
    
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
       
    beta : scalar, float32
        Regularization parameter. Determines the 
        amount of regularization.
        
    delta : scalar, float32
        Hyper-regularization parameter. A low value 
        preserves edgy reconstructions.
    
    init_matrix : ndarray
       Initial guess for the reconstruction. Its
       shape is the same as the reconstructed data.
       
    Returns
    -------
    output : ndarray
        Reconstructed data with dimensions:
        [slices, num_grid, num_grid]

    References
    ----------
    - `IEEE-TMI, Vol 23(9), 1165-1175(2004) \
    <http://dx.doi.org/10.1109/TMI.2004.831224>`_
    """
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    librecon.pml3db.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon.pml3db(data.ctypes.data_as(c_float_p),
                  theta.ctypes.data_as(c_float_p),
                  ctypes.c_float(center),
                  ctypes.c_int(num_projections),
                  ctypes.c_int(num_slices),
                  ctypes.c_int(num_pixels),
                  ctypes.c_int(num_grid),
                  ctypes.c_int(iters),
                  ctypes.c_float(beta),
                  ctypes.c_float(delta),
                  ctypes.c_float(beta1),
                  ctypes.c_float(delta1),
                  ctypes.c_float(regw1),
                  ctypes.c_float(regw2),
                  init_matrix.ctypes.data_as(c_float_p))
    return init_matrix