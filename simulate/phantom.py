# -*- coding: utf-8 -*-
import numpy as np

class Phantom():
    def __init__(self):
        pass

    def values(self, values):
        self.values = np.array(values, dtype='float32')
        self.sizex = values.shape[0]
        self.sizey = values.shape[1]
        self.sizez = values.shape[2]

    def pixel_size(self, pixel_size):
        self.pixel_size = np.array(pixel_size, dtype='float32')
