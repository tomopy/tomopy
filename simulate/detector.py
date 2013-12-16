# -*- coding: utf-8 -*-
import numpy as np

class Detector():
    def __init__(self):
        pass

    def pixel_size(self, pixel_size):
        self.pixel_size = np.array(pixel_size, dtype='float32')

    def num_pixels(self, sizex, sizey):
        self.sizex = np.array(sizex, dtype='int32')
        self.sizey = np.array(sizey, dtype='int32')

    def pixel_coords(self, dist=-1e4, alpha=0, beta=0, gamma=0):
        """ Positions the detector in space and
        returns the pixel coordinates
        (pixel centers not edges).

        Parameters
        ----------
        dist : scalar
            The position of the area detector on X axis.
            It is positioned orthogonal to X axis,
            parallel to ZY transverse plane.

        alpha, beta, gamma : scalar
            Counter-clockwise rotation about X, Y and Z
            axes in radians. First X, then Y and then
            Z rotation.
        """
        lenx = self.sizex * self.pixel_size
        leny = self.sizey * self.pixel_size
        yi = np.arange(-(lenx - self.pixel_size)/2,
                        (lenx - self.pixel_size)/2 + self.pixel_size,
                        self.pixel_size)
        zi = np.arange(-(leny - self.pixel_size)/2,
                        (leny - self.pixel_size)/2 + self.pixel_size,
                        self.pixel_size)
        y0, z0 = np.meshgrid(zi, yi)
        x0 = dist * np.ones((self.sizex, self.sizey))

        # Perform rotation.
        c1, c2, c3 = np.cos([alpha, beta, gamma])
        s1, s2, s3 = np.sin([alpha, beta, gamma])

        coordx = x0*c2*c3 - y0*(s1*s2*c3 + c1*s3) - z0*(c1*s2*c3 - s1*s3)
        coordy = x0*c2*s3 - y0*(s1*s2*s3 - c1*c3) - z0*(c1*s2*s3 + s1*c3)
        coordz = x0*s2 + y0*s1*c2 + z0*c1*c2
        return coordx, coordy, coordz
