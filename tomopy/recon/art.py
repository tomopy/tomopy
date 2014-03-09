# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '..', 'lib/librecon.so'))
librecon = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def _art(data, theta, center, num_grid, iters, init_matrix):
    
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    librecon.art.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon.art(data.ctypes.data_as(c_float_p),
                 theta.ctypes.data_as(c_float_p),
                 ctypes.c_float(center),
                 ctypes.c_int(num_projections),
                 ctypes.c_int(num_slices),
                 ctypes.c_int(num_pixels),
                 ctypes.c_int(num_grid),
                 ctypes.c_int(iters),
                 init_matrix.ctypes.data_as(c_float_p))
    return init_matrix
