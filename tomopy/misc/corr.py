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
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['median_filter',
           'remove_nan',
           'remove_neg']


def _import_shared_lib(lib_name):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            libpath = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..', '..', 'lib', lib_name + '.pyd'))
            return ctypes.CDLL(libpath)
        else:
            libpath = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..', '..', 'lib', lib_name + '.so'))
            return ctypes.CDLL(libpath)
    except OSError as e:
        logger.warning('OSError: Shared library missing.')


LIB_TOMOPY = _import_shared_lib('libtomopy')


def median_filter(arr, size=3, axis=0, ind=None):
    """
    Apply median filter to a 3D array along a specified axis.

    Parameters
    ----------
    arr : ndarray
        Arbitrary 3D array.
    size : int, optional
        The size of the filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ind : array of int, optional
        Indices at which the filtering is applied.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """
    if type(arr) == str and arr == 'SHARED':
        arr = mp.shared_arr
    else:
        arr = mp.distribute_jobs(
            arr, func=median_filter, axis=axis,
            args=(size, axis))
        return arr

    dx, dy, dz = arr.shape
    if ind is None:
        if axis == 0:
            ind = np.arange(0, dx)
        elif axis == 1:
            ind = np.arange(0, dy)
        elif axis == 2:
            ind = np.arange(0, dz)

    if axis == 0:
        for m in ind:
            arr[m, :, :] = filters.median_filter(
                arr[m, :, :], (size, size))
    elif axis == 1:
        for m in ind:
            arr[:, m, :] = filters.median_filter(
                arr[:, m, :], (size, size))
    elif axis == 2:
        for m in ind:
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
    arr[arr < 0.0] = val
    return arr
