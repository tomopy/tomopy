# -*- coding: utf-8 -*-
import os
import ctypes
import numpy as np


# Get the shared library for art.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libart.so'))
libart = ctypes.CDLL(libpath)


class Art():
    def __init__(self, data, theta, center, num_grid, num_air):
        """
        Constructor for Art reconstruction.
        
        Parameters
        ----------
        data : ndarray (float32)
            3-D omography data. [num_projections, num_slices, num_pixels]
            
        theta : ndarray (float32)
            Projection angles in radians.
            
        center : scalar (int32)
            Location of rotation axis on transverse (pixel) axis.
            
        num_grid : scalar (int32)
            Grid size of the reconstructed slices.
        """
        num_projections = np.array(data.shape[0], dtype='int32')
        num_slices = np.array(data.shape[1], dtype='int32')
        num_pixels = np.array(data.shape[2], dtype='int32')
        
        self.data_recon = np.zeros((num_slices, num_grid, num_grid),
                                   dtype='float32')
        
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int_p = ctypes.POINTER(ctypes.c_int)

        libart.create.restype = ctypes.POINTER(ctypes.c_void_p)
        self.obj = libart.create(data.ctypes.data_as(c_float_p),
                                 theta.ctypes.data_as(c_float_p),
                                 center.ctypes.data_as(c_float_p),
                                 num_projections.ctypes.data_as(c_int_p),
                                 num_slices.ctypes.data_as(c_int_p),
                                 num_pixels.ctypes.data_as(c_int_p),
                                 num_grid.ctypes.data_as(c_int_p),
                                 num_air.ctypes.data_as(c_int_p))
        

    def reconstruct(self, iters):
        """
        Perform Art reconstruction for slices starting from
        ``slice_start`` to ``slice_end``. 
        
        Parameters
        ----------
        iters : scalar (int32)
            Number of iterations.
        """
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_int_p = ctypes.POINTER(ctypes.c_int)
                               
        libart.reconstruct(self.obj,
                           self.data_recon.ctypes.data_as(c_float_p),
                           iters.ctypes.data_as(c_int_p))

        return self.data_recon














