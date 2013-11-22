# -*- coding: utf-8 -*-
# Filename: metrics.py
""" Module for setup of the objective function.
"""
import numpy as np
import interpolate
import geometry


class similarityMeasure(object):
    """ Template class.
    """
    def __init__(self, from_img, from_grid, to_img, to_grid, transformType):
        self.transformType = transformType
        self.from_img = from_img
        self.from_grid = from_grid
        self.to_img = to_img
        self.to_grid = to_grid


class mutualInformation(similarityMeasure):
    def calc(self, x0):
        tform = geometry.transform2d(self.transformType, x0)
        to_img = interpolate.interp2d(self.from_img, 
                                     self.from_grid,
                                     tform, 
                                     self.to_grid)
        histogram, x, y = np.histogram2d(self.to_img.flatten(), 
                                        to_img.flatten(), bins=64)
        pab = histogram / np.sum(histogram)
        pa = np.sum(pab, axis=0)
        pb = np.sum(pab, axis=1)
        tmp = pab / np.multiply(np.array(pa,ndmin=2),np.array(pb,ndmin=2).transpose())
        #tmp = np.empty((64, 64))
        #for m in range(64):
        #    for n in range(64):  
        #        tmp[m, n] = pab[m, n] / np.dot(pa[m], pb[n])
        tmp[pab == 0] = 1
        mutInf = np.multiply(pab, np.log2(tmp))
        return -np.sum(mutInf)

class crossCorrelation(similarityMeasure):
    def calc(self, x0):
        tform = geometry.transform2d(self.transformType, x0)
        to_img = interpolate.interp2d(self.from_img, 
                                     self.from_grid,
                                     tform, 
                                     self.to_grid)
        return -np.correlate(self.to_img.flatten(), to_img.flatten())
        
class norm2(similarityMeasure):
    def calc(self, x0):
        tform = geometry.transform2d(self.transformType, x0)
        to_img = interpolate.interp2d(self.from_img, 
                                     self.from_grid,
                                     tform, 
                                     self.to_grid)
        return np.mean((self.to_img.flatten() - to_img.flatten())**2)
        
        
class norm1(similarityMeasure):
    def calc(self, x0):
        tform = geometry.transform2d(self.transformType, x0)
        to_img = interpolate.interp2d(self.from_img, 
                                     self.from_grid,
                                     tform, 
                                     self.to_grid)
        return np.mean(np.abs(self.to_img.flatten() - to_img.flatten()))
        
