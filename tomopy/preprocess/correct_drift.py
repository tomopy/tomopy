# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '..', 'lib/libprep.so'))
libprep = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def _correct_drift(data, air_pixels):
    
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')
        

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    libprep.correct_drift.restype = ctypes.POINTER(ctypes.c_void_p)
    libprep.correct_drift(data.ctypes.data_as(c_float_p),
                          ctypes.c_int(num_projections),
                          ctypes.c_int(num_slices),
                          ctypes.c_int(num_pixels),
                          ctypes.c_int(air_pixels))
    return data