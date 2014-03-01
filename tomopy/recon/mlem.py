# -*- coding: utf-8 -*-
import os
import ctypes
import numpy as np


# Get the shared library for art.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libmlem.so'))
libmlem = ctypes.CDLL(libpath)


class Mlem():
    def __init__(self, data):
        
        num_projections = np.array(data.shape[0], dtype='int32')
        num_slices = np.array(data.shape[1], dtype='int32')
        num_pixels = np.array(data.shape[2], dtype='int32')
    
        self.data_recon = np.ones((num_slices,
                                   num_pixels,
                                   num_pixels), 
                                  dtype='float32')
        
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int_p = ctypes.POINTER(ctypes.c_int)

        libmlem.create.restype = ctypes.POINTER(ctypes.c_void_p)
        self.obj = libmlem.create(num_projections.ctypes.data_as(c_int_p),
                                  num_slices.ctypes.data_as(c_int_p),
                                  num_pixels.ctypes.data_as(c_int_p),
                                  data.ctypes.data_as(c_float_p))
    
    def reconstruct(self, iters, center, theta):
                               
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int_p = ctypes.POINTER(ctypes.c_int)
                               
        libmlem.reconstruct(self.obj,
                           iters.ctypes.data_as(c_int_p),
                           center.ctypes.data_as(c_float_p),
                           theta.ctypes.data_as(c_float_p),
                           self.data_recon.ctypes.data_as(c_float_p))
        return self.data_recon