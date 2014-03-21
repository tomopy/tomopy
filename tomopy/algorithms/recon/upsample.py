# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '../..', 'lib/librecon.so'))
librecon = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def upsample2d(data, level):
    """
    Upsample the slices.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]
        
    level : scalar, int32
        Upsampling level. For example level=2 
        means, the sinogram will be upsampled by 4,
        and level=3 means upsampled by 8.
    
    Returns
    -------
    output : ndarray
        Downsampled reconstructed 3-D data with dimensions:
        [slices, pixels*level^2, pixels*level^2]
    """
    num_slices = np.array(data.shape[0], dtype='int32')
    num_pixels = np.array(data.shape[1], dtype='int32')
    
    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    binsize = np.power(2, level)
    upsampled_data = np.zeros((num_slices, 
                               num_pixels*binsize, 
                               num_pixels*binsize),
                              dtype='float32')
                          
    librecon.upsample2d.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon.upsample2d(data.ctypes.data_as(c_float_p),
                        ctypes.c_int(num_slices),
                        ctypes.c_int(num_pixels),
                        ctypes.c_int(level),
                        upsampled_data.ctypes.data_as(c_float_p))
    return upsampled_data

# --------------------------------------------------------------------

def upsample3d(data, level):
    """
    Upsample the slices and pixels.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]
        
    level : scalar, int32
        Upsampling level. For example level=2 
        means, the sinogram will be upsampled by 4,
        and level=3 means upsampled by 8.
    
    Returns
    -------
    output : ndarray
        Downsampled reconstructed 3-D data with dimensions:
        [slices*level^2, pixels*level^2, pixels*level^2]
    """
    num_slices = np.array(data.shape[0], dtype='int32')
    num_pixels = np.array(data.shape[1], dtype='int32')
    
    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    binsize = np.power(2, level)
    upsampled_data = np.zeros((num_slices*binsize, 
                               num_pixels*binsize, 
                               num_pixels*binsize),
                              dtype='float32')
                          
    librecon.upsample3d.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon.upsample3d(data.ctypes.data_as(c_float_p),
                        ctypes.c_int(num_slices),
                        ctypes.c_int(num_pixels),
                        ctypes.c_int(level),
                        upsampled_data.ctypes.data_as(c_float_p))
    return upsampled_data

