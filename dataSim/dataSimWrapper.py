# -*- coding: utf-8 -*-
import ctypes
import numpy as np

lib = ctypes.CDLL('/local/dgursoy/GIT/tomopy/dataSim/dataSim.so')
c_int_p = ctypes.POINTER(ctypes.c_int)
c_float_p = ctypes.POINTER(ctypes.c_float)

class CStruct(ctypes.Structure):
    _fields_ = [("objSizeX", ctypes.c_int),
                ("objSizeY", ctypes.c_int),
                ("objSizeZ", ctypes.c_int),
                ("objPixelSize", ctypes.c_float)]

class DataSim(object):
    def __init__(self, objVals, objPixelSize):
        self.vals = CStruct()
        self.vals.objSizeX = objVals.shape[0]
        self.vals.objSizeY = objVals.shape[1]
        self.vals.objSizeZ = objVals.shape[2]
        self.vals.objPixelSize = objPixelSize
        self.obj = lib.create(ctypes.byref(self.vals),
                              objVals.ctypes.data_as(c_float_p))

    def calc(self, srcx, srcy, srcz, detx, dety, detz):
        numPts = srcx.size
        self.proj = np.zeros(srcx.shape, dtype='float32')
        lib.calc(self.obj,
                 numPts,
                 srcx.ctypes.data_as(c_float_p),
                 srcy.ctypes.data_as(c_float_p),
                 srcz.ctypes.data_as(c_float_p),
                 detx.ctypes.data_as(c_float_p),
                 dety.ctypes.data_as(c_float_p),
                 detz.ctypes.data_as(c_float_p),
                 self.proj.ctypes.data_as(c_float_p))
