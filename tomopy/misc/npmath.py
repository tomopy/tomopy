#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy  as np
import logging
import warnings

logger = logging.getLogger(__name__)


__author__ = "Chen Zhang"
__all__    = [ 'gauss',
               'calc_cummulative_dist',
               'calc_affine_transform',
]


def gauss1d(x, *p):
    """
    1D Gaussian function used for curve fitting.

    Parameters
    ----------
    x  :  np.1ndarray
        1D array for curve fitting
    p  :  parameter lis t
        magnitude, center, std = p

    Returns
    -------
    1d Gaussian distrobution evaluted at x with p
    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def calc_cummulative_dist(data, steps=None):
    """
    Calculated the cumulative distribution from a data array without using
    binning.  
    This provides a more robust way to representing the statistics of the 
    incoming data without worrying about the binning/samping effect.

    Parameters
    ----------
    data  :  1D np.array
        1-D numpy array 
    steps :  [ None | int ], optional
        Number of elements in the returning array

    Returns
    -------
    pltX  : 1D np.array
        plt data along x (data direction)
    pltY  : 1D np.array
        plt data long y (density direction)
    """
    x = np.sort(data)

    # check if list empty
    if len(x) < 1:
        return [], []

    # subsamping if steps is speficiied and the number is smaller than the
    # total lenght of x
    if (steps is not None) and len(x) > steps:
        x = x[np.arange(0, len(x), int(np.ceil(len(x)/steps)))]

    # calculate the cumulative density
    xx = np.tile(x, (2, 1)).flatten(order='F')
    y = np.arange(len(x))
    yy = np.vstack((y, y+1)).flatten(order='F')/float(y[-1])

    return xx, yy


def calc_affine_transform(pts_src, pts_tgt):
    """
    Use least square regression to calculate  the 2D affine transformation 
    matrix (3x3, rot&trans) based on given set of (marker) points.
                            pts_src -> pts_tgt

    Parameters
    ----------
    pts_src  :  np.2darray
        source points with dimension of (n, 2) where n is the number of 
        marker points
    pts_tgt  :  np.2darray
        target points where
                F(pts_src) = pts_tgt
    Returns
    -------
    np.2darray
        A 3x3 2D affine transformation matrix
          | r_11  r_12  tx |    | x1 x2 ...xn |   | x1' x2' ...xn' |
          | r_21  r_22  ty | *  | y1 y2 ...yn | = | y1' y2' ...yn' |
          |  0     0     1 |    |  1  1 ... 1 |   |  1   1  ... 1  |
        where r_ij represents the rotation and t_k represents the translation
    """
    # augment data with padding to include translation
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])

    # NOTE:
    #   scipy affine_transform performs as np.dot(m, vec), 
    #   therefore we need to transpose the matrix here 
    #   to get the correct rotation
      
    return np.linalg.lstsq(pad(pts_src), pad(pts_tgt), rcond=-1)[0].T
