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
Module for data size morphing functions.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tomopy.misc.mproc as mp
import tomopy.extern as ext
from tomopy.util import *
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['pad',
           'downsample',
           'upsample']


LIB_TOMOPY = ext.c_shared_lib('libtomopy')


def pad(arr, axis, npad=None, val=0):
    """
    Pad an array along specified axis.

    Parameters
    ----------
    arr : ndarray
        Input array.
    npad : int, optional
        New dimension after padding.
    axis : int, optional
        Axis along which padding will be performed.
    val : float, optional
        Pad value.

    Returns
    -------
    ndarray
        Padded 3D array.
    """
    if npad is None:
        npad = _get_npad(arr.shape[axis])
    pad_width = _get_pad_sequence(arr.shape, axis, npad)
    cval = _get_pad_sequence(arr.shape, axis, npad)
    return np.pad(arr, pad_width, 'constant', constant_values=cval)


def _get_npad(dim):
    return int(np.ceil(dim * np.sqrt(2)))-dim


def _get_pad_sequence(shape, axis, npad):
    pad_seq = []
    for m in range(len(shape)):
        if m == axis:
            pad_seq.append((npad, npad))
        else:
            pad_seq.append((0, 0))
    return pad_seq


def downsample(arr, level=1, axis=2):
    """
    Downsample along specified axis of a 3D array.

    Parameters
    ----------
    arr : ndarray
        3D input array.
    level : int, optional
        Downsampling level in powers of two.
    axis : int, optional
        Axis along which downsampling will be performed.

    Returns
    -------
    ndarray
        Downsampled 3D array.
    """
    return _sample(arr, level, axis, mode=0)


def upsample(arr, level=1, axis=2):
    """
    Upsample along specified axis of a 3D array.

    Parameters
    ----------
    arr : ndarray
        3D input array.
    level : int, optional
        Downsampling level in powers of two.
    axis : int, optional
        Axis along which upsampling will be performed.

    Returns
    -------
    ndarray
        Upsampled 3D array.
    """
    return _sample(arr, level, axis, mode=1)


def _sample(arr, level, axis, mode):
    arr = as_float32(arr)
    dx, dy, dz = arr.shape

    if mode == 0:
        dim = arr.shape[axis] / np.power(2, level)
    if mode == 1:
        dim = arr.shape[axis] * np.power(2, level)

    out = _init_out(arr, axis, dim)
    return ext.c_sample(mode, arr, dx, dy, dz, level, axis, out)


def _init_out(arr, axis, dim, val=0.):
    if axis > 3:
        logger.warning('Maximum allowable dimension is three.')
    dx, dy, dz = arr.shape
    shape = [dx, dy, dz]
    shape[axis] = dim
    return val * np.ones(shape, dtype='float32')
