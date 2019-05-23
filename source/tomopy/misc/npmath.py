#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2019, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2019. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

import numpy as np
import scipy
import logging

logger = logging.getLogger(__name__)


__author__ = "Chen Zhang"
__all__ = ['gauss1d',
           'discrete_cdf',
           'calc_affine_transform',
           ]


def gauss1d(x, *p):
    """
    1D Gaussian function used for curve fitting.

    Parameters
    ----------
    x  :  np.ndarray
        1D array for curve fitting
    p  :  parameter lis t
        magnitude, center, std = p

    Returns
    -------
    1d Gaussian distribution evaluted at x with p
    """
    A, mu, sigma = p
    return A * np.exp(-(x - mu)**2 / (2. * sigma**2))


def discrete_cdf(data, steps=None):
    """
    Calculate CDF of given data without discrete binning to avoid unnecessary
    skew of distribution.

    The default steps (None) will use the whole data. In other words, it is
    close to considering using bin_size=1 or bins=len(data).

    Parameters
    ----------
    data  :  np.ndarray
        1-D numpy array
    steps :  [ None | int ], optional
        Number of elements in the returning array

    Returns
    -------
    pltX  : np.ndarray
        Data along x (data) direction
    pltY  : np.ndarray
        Data along y (density) direction
    """
    x = np.sort(data)

    # check if list is empty
    if len(x) == 0:
        return [], []

    # subsamping if steps is specified and the number is smaller than the
    # total lenght of x
    if (steps is not None) and len(x) > steps:
        x = x[np.arange(0, len(x), int(np.ceil(len(x) / steps)))]

    # calculate the cumulative density
    xx = np.tile(x, (2, 1)).flatten(order='F')
    y = np.arange(len(x))
    yy = np.vstack((y, y + 1)).flatten(order='F') / float(y[-1])

    return xx, yy


def calc_affine_transform(pts_source, pts_target):
    """
    Use least square regression to calculate the 2D affine transformation
    matrix (3x3, rot&trans) based on given set of (marker) points.
                            pts_source -> pts_target

    Parameters
    ----------
    pts_source  :  np.2darray
        source points with dimension of (n, 2) where n is the number of
        marker points
    pts_target  :  np.2darray
        target points where
                F(pts_source) = pts_target
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
    def pad(x): return np.hstack([x, np.ones((x.shape[0], 1))])

    # NOTE:
    #   scipy affine_transform performs as np.dot(m, vec),
    #   therefore we need to transpose the matrix here
    #   to get the correct rotation

    return scipy.linalg.lstsq(pad(pts_source), pad(pts_target))[0].T
