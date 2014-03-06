# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '..', 'lib/libmlem.so'))
libmlem = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def _mlem(data, theta, center, num_grid, iters):
    
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    # Call C function.
    data_recon = np.ones((num_slices, num_grid, num_grid), dtype='float32')
    
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libmlem.mlem.restype = ctypes.POINTER(ctypes.c_void_p)
    libmlem.mlem(data.ctypes.data_as(c_float_p),
                 theta.ctypes.data_as(c_float_p),
                 ctypes.c_float(center),
                 ctypes.c_int(num_projections),
                 ctypes.c_int(num_slices),
                 ctypes.c_int(num_pixels),
                 ctypes.c_int(num_grid),
                 ctypes.c_int(iters),
                 data_recon.ctypes.data_as(c_float_p))
    return data_recon
