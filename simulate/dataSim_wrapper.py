# -*- coding: utf-8 -*-
import ctypes
import numpy as np

lib = ctypes.CDLL('/local/dgursoy/GIT/tomopy/simulate/dataSim.so')
c_int_p = ctypes.POINTER(ctypes.c_int)
c_float_p = ctypes.POINTER(ctypes.c_float)


class DataSimCStruct(ctypes.Structure):
    _fields_ = [("sizex", ctypes.c_int),
                ("sizey", ctypes.c_int),
                ("sizez", ctypes.c_int),
                ("pixel_size", ctypes.c_float)]


class DataSim(object):
    def __init__(self, obj_vals, pixel_size):
        self.vals = DataSimCStruct()
        self.vals.sizex = obj_vals.shape[0]
        self.vals.sizey = obj_vals.shape[1]
        self.vals.sizez = obj_vals.shape[2]
        self.vals.pixel_size = pixel_size
        self.obj = lib.create(ctypes.byref(self.vals),
                              obj_vals.ctypes.data_as(c_float_p))

    def calc(self, srcx, srcy, srcz, detx, dety, detz):
        num_pts = srcx.size
        proj = np.zeros(srcx.shape, dtype='float32')
        lib.calc(self.obj,
                 num_pts,
                 srcx.ctypes.data_as(c_float_p),
                 srcy.ctypes.data_as(c_float_p),
                 srcz.ctypes.data_as(c_float_p),
                 detx.ctypes.data_as(c_float_p),
                 dety.ctypes.data_as(c_float_p),
                 detz.ctypes.data_as(c_float_p),
                 proj.ctypes.data_as(c_float_p))
        return proj
