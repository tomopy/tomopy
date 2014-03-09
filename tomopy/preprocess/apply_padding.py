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

def _apply_padding(data, num_pad):
    
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')
        

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    padded_data = np.ones((num_projections, num_slices, num_pad), 
                           dtype='float32')
    
    libprep.apply_padding.restype = ctypes.POINTER(ctypes.c_void_p)
    libprep.apply_padding(data.ctypes.data_as(c_float_p),
                          ctypes.c_int(num_projections),
                          ctypes.c_int(num_slices),
                          ctypes.c_int(num_pixels),
                          ctypes.c_int(num_pad),
                          padded_data.ctypes.data_as(c_float_p))
    return padded_data
    
    