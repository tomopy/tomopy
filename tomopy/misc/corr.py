#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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

"""
Module for data correction functions.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import ctypes
import os
import tomopy.misc.mproc as mp
from scipy.ndimage import filters
from tomopy.util import *
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['gaussian_filter',
           'median_filter',
           'remove_nan',
           'remove_neg']


def gaussian_filter(arr, sigma=3, order=0, axis=0, ncore=None, nchunk=None):
    """
    Apply Gaussian filter to 3D array along specified axis.

    Parameters
    ----------
    arr : ndarray
        Arbitrary 3D array.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard deviations
        of the Gaussian filter are given for each axis as a sequence, or
        as a single number, in which case it is equal for all axes.
    order : {0, 1, 2, 3} or sequence from same set, optional
        Order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. An order of 1, 2, or 3
        corresponds to convolution with the first, second or third
        derivatives of a Gaussian. Higher order derivatives are not
        implemented
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        3D array of same shape as input.
    """
    arr = as_float32(arr)
    arr = mp.distribute_jobs(
        arr,
        func=_gaussian_filter,
        args=(sigma, order, axis),
        axis=axis,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _gaussian_filter(sigma, order, axis, istart, iend):
    arr = mp.SHARED_ARRAY
    for m in range(istart, iend):
        if axis == 0:
            arr[m, :, :] = filters.gaussian_filter(
                arr[m, :, :], sigma, order)
        elif axis == 1:
            arr[:, m, :] = filters.gaussian_filter(
                arr[:, m, :], sigma, order)
        elif axis == 2:
            arr[:, :, m] = filters.gaussian_filter(
                arr[:, :, m], sigma, order)


def median_filter(arr, size=3, axis=0, ncore=None, nchunk=None):
    """
    Apply median filter to 3D array along specified axis.

    Parameters
    ----------
    arr : ndarray
        Arbitrary 3D array.
    size : int, optional
        The size of the filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """
    arr = as_float32(arr)
    arr = mp.distribute_jobs(
        arr,
        func=_median_filter,
        args=(size, axis),
        axis=axis,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _median_filter(size, axis, istart, iend):
    arr = mp.SHARED_ARRAY
    for m in range(istart, iend):
        if axis == 0:
            arr[m, :, :] = filters.median_filter(
                arr[m, :, :], (size, size))
        elif axis == 1:
            arr[:, m, :] = filters.median_filter(
                arr[:, m, :], (size, size))
        elif axis == 2:
            arr[:, :, m] = filters.median_filter(
                arr[:, :, m], (size, size))


def remove_nan(arr, val=0.):
    """
    Replace NaN values in array with a given value.

    Parameters
    ----------
    arr : ndarray
        Input data.
    val : float, optional
        Values to be replaced with NaN values in array.

    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = as_float32(arr)
    arr[np.isnan(arr)] = val
    return arr


def remove_neg(arr, val=0.):
    """
    Replace negative values in array with a given value.

    Parameters
    ----------
    arr : ndarray
        Input array.
    val : float, optional
        Values to be replaced with negative values in array.

    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = as_float32(arr)
    arr[arr < 0.0] = val
    return arr
