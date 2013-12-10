# -*- coding: utf-8 -*-
import ctypes
import numpy as np

lib = ctypes.CDLL('/local/dgursoy/GIT/tomopy/simulate/simulate.so')
c_int_p = ctypes.POINTER(ctypes.c_int)
c_float_p = ctypes.POINTER(ctypes.c_float)

class SourceCStruct(ctypes.Structure):
    _fields_ = [("sizex", ctypes.c_int),
                ("sizey", ctypes.c_int),
                ("pixel_size", ctypes.c_float),
                ("energy", ctypes.c_float)]

class DetectorCStruct(ctypes.Structure):
    _fields_ = [("sizex", ctypes.c_int),
                ("sizey", ctypes.c_int),
                ("pixel_size", ctypes.c_float)]

class PhantomCStruct(ctypes.Structure):
    _fields_ = [("sizex", ctypes.c_int),
                ("sizey", ctypes.c_int),
                ("sizez", ctypes.c_int),
                ("pixel_size", ctypes.c_float)]

class Simulate():
    def __init__(self, src, det, obj):
        self.source = SourceCStruct()
        self.detector = DetectorCStruct()
        self.phantom = PhantomCStruct()
        self.source.sizex = src.sizex
        self.source.sizey = src.sizey
        self.source.pixel_size = src.pixel_size
        self.source.energy = src.energy
        self.detector.sizex = det.sizex
        self.detector.sizey = det.sizey
        self.detector.pixel_size = det.pixel_size
        self.phantom.sizex = obj.sizex
        self.phantom.sizey = obj.sizey
        self.phantom.sizez = obj.sizez
        self.phantom.pixel_size = obj.pixel_size
        self.obj = lib.create(ctypes.byref(self.source),
                              ctypes.byref(self.detector),
                              ctypes.byref(self.phantom),
                              obj.values.ctypes.data_as(c_float_p))

    def calc3d(self, srcx, srcy, srcz, detx, dety, detz):
        srcx = np.array(srcx, dtype='float32')
        srcy = np.array(srcy, dtype='float32')
        srcz = np.array(srcz, dtype='float32')
        detx = np.array(detx, dtype='float32')
        dety = np.array(dety, dtype='float32')
        detz = np.array(detz, dtype='float32')
        proj = np.zeros((self.detector.sizex,
                         self.detector.sizey), dtype='float32')

        import time
        t = time.time()
        lib.calc3d(self.obj,
                   srcx.ctypes.data_as(c_float_p),
                   srcy.ctypes.data_as(c_float_p),
                   srcz.ctypes.data_as(c_float_p),
                   detx.ctypes.data_as(c_float_p),
                   dety.ctypes.data_as(c_float_p),
                   detz.ctypes.data_as(c_float_p),
                   proj.ctypes.data_as(c_float_p))
        print time.time() - t
        return proj

    def calc2d(self, srcx, srcy, detx, dety):
        srcx = np.array(srcx, dtype='float32')
        srcy = np.array(srcy, dtype='float32')
        detx = np.array(detx, dtype='float32')
        dety = np.array(dety, dtype='float32')
        proj = np.zeros((self.detector.sizex,
                         self.detector.sizey), dtype='float32')

        import time
        t = time.time()
        lib.calc2d(self.obj,
                   srcx.ctypes.data_as(c_float_p),
                   srcy.ctypes.data_as(c_float_p),
                   detx.ctypes.data_as(c_float_p),
                   dety.ctypes.data_as(c_float_p),
                   proj.ctypes.data_as(c_float_p))
        print time.time() - t
        return proj
