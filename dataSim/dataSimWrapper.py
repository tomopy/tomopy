# -*- coding: utf-8 -*-
import ctypes
import numpy as np


lib = ctypes.CDLL('/local/dgursoy/GIT/tomopy/dataSim/dataSim.so')
c_int_p = ctypes.POINTER(ctypes.c_int)
c_float_p = ctypes.POINTER(ctypes.c_float)


class DataSimCStruct(ctypes.Structure):
    _fields_ = [("objSizeX", ctypes.c_int),
                ("objSizeY", ctypes.c_int),
                ("objSizeZ", ctypes.c_int),
                ("objPixelSize", ctypes.c_float)]


class DataSim(object):
    def __init__(self, objVals, objPixelSize):
        self.vals = DataSimCStruct()
        self.vals.objSizeX = objVals.shape[0]
        self.vals.objSizeY = objVals.shape[1]
        self.vals.objSizeZ = objVals.shape[2]
        self.vals.objPixelSize = objPixelSize
        self.obj = lib.create(ctypes.byref(self.vals),
                              objVals.ctypes.data_as(c_float_p))

    def calc(self, srcx, srcy, srcz, detx, dety, detz):
        numPts = srcx.size
        proj = np.zeros(srcx.shape, dtype='float32')
        lib.calc(self.obj,
                 numPts,
                 srcx.ctypes.data_as(c_float_p),
                 srcy.ctypes.data_as(c_float_p),
                 srcz.ctypes.data_as(c_float_p),
                 detx.ctypes.data_as(c_float_p),
                 dety.ctypes.data_as(c_float_p),
                 detz.ctypes.data_as(c_float_p),
                 proj.ctypes.data_as(c_float_p))
        return proj


class AreaDetector:
    def __init__(self, resolution, pixelSize):
        """ Area detector constructor.
        """
        self.resolution = resolution
        self.pixelSize = float(pixelSize)

    def getPixelCoords(self, dist=1e12, alpha=0, beta=0, gamma=0):
        """ Positions the detector in space and
        returns the pixel coordinates.
        """
        lenx = self.resolution[0] * self.pixelSize
        leny = self.resolution[1] * self.pixelSize
        yi = np.arange(-(lenx - self.pixelSize)/2,
                        (lenx - self.pixelSize)/2 + self.pixelSize,
                        self.pixelSize)
        zi = np.arange(-(leny - self.pixelSize)/2,
                        (leny - self.pixelSize)/2 + self.pixelSize,
                        self.pixelSize)
        y0, z0 = np.meshgrid(yi, zi)
        x0 = dist * np.ones(self.resolution)

        x1 = x0
        y1 = y0 * np.cos(alpha) + z0 * np.sin(alpha)
        z1 = y0 * np.sin(alpha) - z0 * np.cos(alpha)

        x2 = z1 * np.sin(beta) - x1 * np.cos(beta)
        y2 = y1
        z2 = z1 * np.cos(beta) + x1 * np.sin(beta)

        coordx = y2 * np.sin(gamma) - x2 * np.cos(gamma)
        coordy = y2 * np.cos(gamma) + x2 * np.sin(gamma)
        coordz = z2

        coordx = coordx.astype('float32')
        coordy = coordy.astype('float32')
        coordz = coordz.astype('float32')
        return coordx, coordy, coordz

    def getAngles(self, phi, numProj):
        """ Calculates the detecto-source pair
        alignments from object alignment parameters.
        """
        gamma = np.arange(0, np.pi, np.pi / numProj)
        alpha = phi * np.sin(2 * gamma)
        beta = phi * np.cos(2 * gamma)
        return alpha, beta, gamma
