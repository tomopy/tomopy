# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '../..', 'lib/libtest.so'))
librecon = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def mlem_transmission(white, data, theta, center, num_grid, iters, init_matrix):
    """
    Applies Maximum-Likelihood Expectation-Maximization (MLEM)
    method to obtain reconstructions.
    
    Parameters
    ----------
    white : ndarray, float32
        White-field data with dimensions:
        [slices, pixels]
        
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
        
    References
    ----------
    - `http://en.wikipedia.org/wiki/Maximum_likelihood \
    <http://en.wikipedia.org/wiki/Maximum_likelihood>`_
    - `http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm \
    <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_
    - `Dempster, A. P.; Laird, N. M.; Rubin, D. B. Maximum likelihood from incomplete data via the EM algorithm. With discussion. J. Roy. Statist. Soc. Ser. B 39 (1977), no. 1, 1–38 \
    <http://www.jstor.org/discover/10.2307/2984875?uid=17682872&uid=3739656&uid=2&uid=3&uid=67&uid=16752384&uid=62&uid=3739256&sid=21103820705023>`_
    """
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    librecon.mlem_transmission.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon.mlem_transmission(
                  data.ctypes.data_as(c_float_p),
                  white.ctypes.data_as(c_float_p),
                  theta.ctypes.data_as(c_float_p),
                  ctypes.c_float(center),
                  ctypes.c_int(num_projections),
                  ctypes.c_int(num_slices),
                  ctypes.c_int(num_pixels),
                  ctypes.c_int(num_grid),
                  ctypes.c_int(iters),
                  init_matrix.ctypes.data_as(c_float_p)
                  )
    return init_matrix