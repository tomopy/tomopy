# -*- coding: utf-8 -*-
# Filename: grid2d.py
""" Module for spatial referencing of objects in 2-D space.
"""
from abc import ABCMeta, abstractmethod
import numpy as np

# Grid object definiton.
class Grid2d(object):
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
        on the grid in X and Y dimensions.

        Returns
        -------
        coordx : ndarray
            Spatial coordinates in X dimension.

        coordy : ndarray
            Spatial coordinates in Y dimension.
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


class Line(Grid2d):
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
        return self.axes_coords()

    def overlap(self, point):
        point = np.array(point, dtype='float')
        return np.logical_and(point >= self.limits[0],
                              point <= self.limits[1])


class Plane(Grid2d):
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
        return np.c_[x0.flatten(1), y0.flatten(1)]

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
