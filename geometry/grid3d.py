# -*- coding: utf-8 -*-
# Filename: grid3d.py
""" Module for spatial referencing of objects in 3-D space.
"""
from abc import ABCMeta, abstractmethod
import numpy as np

# Grid object definiton.
class Grid3d(object):
    """Abstract spatial referencing for unit objects.

    Attributes
    ----------
    num_pixels : ndarray
        Number of discretization of the object.

    limits : ndarray
        Minimum and maximum coordinate values in each dimension.

    pixel_size : ndarray
        Size of the pixels.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def axes_coords(self):
        """ Returns the spatial coordinates of the points
        on the grid in X, Y and Z dimensions.

        Returns
        -------
        coordx : ndarray
            Spatial coordinates in X dimension.

        coordy : ndarray
            Spatial coordinates in Y dimension.

        coordz : ndarray
            Spatial coordinates in Z dimension.
        """
        pass

    @abstractmethod
    def pixel_coords(self):
        """ Returns the complete set of points in the grid.

        Returns
        -------
        out : ndarray
            Point array on the grid.
        """
        pass

    @abstractmethod
    def overlap(self):
        """ Returns a boolean mask array for if the
        points given by ``point`` are inside or outside
        the object boundaries.

        Parameters
        ----------
        point : ndarray
            Input points.

        Returns
        -------
        out : bool ndarray
            A boolean mask array. The coefficients
            are ``True`` corresponding to the points
            inside and ``False`` for points outside
            of the grid object boundaries.
        """
        pass


class Line(Grid3d):
    def __init__(self, num_pixels, limits):
        self.num_pixels = np.array(num_pixels, dtype='int')
        self.limits = np.array(limits, dtype='float')
        self.pixel_size = ([self.limits[1] - self.limits[0]]
                           / self.num_pixels)

    def axes_coords(self):
        return np.linspace(self.limits[0] + self.pixel_size[0] / 2,
                           self.limits[1] - self.pixel_size[0] / 2,
                           self.num_pixels)

    def pixel_coords(self):
        x0 = self.axes_coords()
        y0 = np.zeros(x0.size, dtype=x0.dtype)
        z0 = np.zeros(x0.size, dtype=x0.dtype)
        return np.c_[x0.flatten(1), y0.flatten(1), z0.flatten(1)]

    def overlap(self, point):
        point = np.array(point, dtype='float')
        return np.logical_and(point >= self.limits[0],
                              point <= self.limits[1])


class Plane(Grid3d):
    def __init__(self, num_pixels, limits):
        self.num_pixels = np.array(num_pixels, dtype='int')
        self.limits = np.array(limits, dtype='float')
        self.pixel_size = ([self.limits[1] - self.limits[0],
                            self.limits[3] - self.limits[2]]
                            / self.num_pixels)

    def axes_coords(self):
        coordx = np.linspace(self.limits[0] + self.pixel_size[0] / 2,
                             self.limits[1] - self.pixel_size[0] / 2,
                             self.num_pixels[0])
        coordy = np.linspace(self.limits[2] + self.pixel_size[1] / 2,
                             self.limits[3] - self.pixel_size[1] / 2,
                             self.num_pixels[1])
        return coordx, coordy

    def pixel_coords(self):
        coordx, coordy = self.axes_coords()
        x0, y0 = np.meshgrid(coordx, coordy)
        z0 = np.zeros(x0.size, dtype=x0.dtype)
        return np.c_[x0.flatten(1), y0.flatten(1), z0.flatten(1)]

    def overlap(self, point):
        point = np.array(point, dtype='float')
        if point.size == 2:
            rangex = np.logical_and(point[0] >= self.limits[0],
                                    point[0] <= self.limits[1])
            rangey = np.logical_and(point[1] >= self.limits[2],
                                    point[1] <= self.limits[3])
        else:
            rangex = np.logical_and(point[:, 0] >= self.limits[0],
                                    point[:, 0] <= self.limits[1])
            rangey = np.logical_and(point[:, 1] >= self.limits[2],
                                    point[:, 1] <= self.limits[3])
        return np.logical_and(rangex, rangey)


class Volume(Grid3d):
    def __init__(self, num_pixels, limits):
        self.num_pixels = np.array(num_pixels, dtype='int')
        self.limits = np.array(limits, dtype='float')
        self.pixel_size = ([self.limits[1] - self.limits[0],
                            self.limits[3] - self.limits[2],
                            self.limits[5] - self.limits[4]]
                            / self.num_pixels)

    def axes_coords(self):
        coordx = np.linspace(self.limits[0] + self.pixel_size[0] / 2,
                             self.limits[1] - self.pixel_size[0] / 2,
                             self.num_pixels[0])
        coordy = np.linspace(self.limits[2] + self.pixel_size[1] / 2,
                             self.limits[3] - self.pixel_size[1] / 2,
                             self.num_pixels[1])
        coordz = np.linspace(self.limits[4] + self.pixel_size[2] / 2,
                             self.limits[5] - self.pixel_size[2] / 2,
                             self.num_pixels[2])
        return coordx, coordy, coordz

    def pixel_coords(self):
        coordx, coordy, coordz = self.axes_coords()
        x0, y0, z0 = np.meshgrid(coordx, coordy, coordz)
        return np.c_[x0.flatten(1), y0.flatten(1), z0.flatten(1)]

    def overlap(self, point):
        point = np.array(point, dtype='float')
        if point.size == 3:
            rangex = np.logical_and(point[0] >= self.limits[0],
                                    point[0] <= self.limits[1])
            rangey = np.logical_and(point[1] >= self.limits[2],
                                    point[1] <= self.limits[3])
            rangez = np.logical_and(point[2] >= self.limits[4],
                                    point[2] <= self.limits[5])
        else:
            rangex = np.logical_and(point[:, 0] >= self.limits[0],
                                    point[:, 0] <= self.limits[1])
            rangey = np.logical_and(point[:, 1] >= self.limits[2],
                                    point[:, 1] <= self.limits[3])
            rangez = np.logical_and(point[:, 2] >= self.limits[4],
                                    point[:, 2] <= self.limits[5])
        return reduce(np.logical_and, [rangex, rangey, rangez])
