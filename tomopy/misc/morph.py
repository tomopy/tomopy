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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tomopy.util.extern as extern
import tomopy.util.dtype as dtype
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['downsample',
           'upsample',
           'pad',
           'sino_360_to_180',
           'sino_360_t0_180',  # For backward compatibility
           'trim_sinogram']


LIB_TOMOPY = extern.c_shared_lib('libtomopy')


def pad(arr, axis, npad=None, mode='constant', **kwargs):
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
    mode : str or function
        One of the following string values or a user supplied function.

        'constant'
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
    constant_values : float, optional
        Used in 'constant'. Pad value

    Returns
    -------
    ndarray
        Padded 3D array.
    """

    allowedkwargs = {'constant': ['constant_values'],
                     'edge': [], }

    kwdefaults = {'constant_values': 0, }

    if isinstance(mode, str):
        for key in kwargs:
            if key not in allowedkwargs[mode]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowedkwargs[mode]))
        for kw in allowedkwargs[mode]:
            kwargs.setdefault(kw, kwdefaults[kw])
    else:
        raise ValueError('mode keyword value must be string, got %s: ' %
                         type(mode))

    if npad is None:
        npad = _get_npad(arr.shape[axis])

    pad_width = _get_pad_sequence(arr.shape, axis, npad)

    return np.pad(arr, pad_width, str(mode), **kwargs)


def _get_npad(dim):
    return int(np.ceil((dim * np.sqrt(2) - dim) / 2))


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
        Downsampled 3D array in float32.
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
        Upsampled 3D array in float32.
    """
    return _sample(arr, level, axis, mode=1)


def _sample(arr, level, axis, mode):
    arr = dtype.as_float32(arr.copy())
    dx, dy, dz = arr.shape

    if mode == 0:
        dim = int(arr.shape[axis] / np.power(2, level))
    if mode == 1:
        dim = int(arr.shape[axis] * np.power(2, level))

    out = _init_out(arr, axis, dim)
    return extern.c_sample(mode, arr, dx, dy, dz, level, axis, out)


def _init_out(arr, axis, dim, val=0.):
    if axis > 3:
        logger.warning('Maximum allowable dimension is three.')
    dx, dy, dz = arr.shape
    shape = [dx, dy, dz]
    shape[axis] = dim
    return val * np.ones(shape, dtype='float32')


def trim_sinogram(data, center, x, y, diameter):
    """
    Provide sinogram corresponding to a circular region of interest
    by trimming the complete sinogram of a compact object.

    Parameters
    ----------
    data : ndarray
        Input 3D data.

    center : float
        Rotation center location.

    x, y : int, int
        x and y coordinates in pixels (image center is (0, 0))

    diameter : float
        Diameter of the circle of the region of interest.

    Returns
    -------
    ndarray
        Output 3D data.

    """
    data = dtype.as_float32(data.copy())
    dx, dy, dz = data.shape

    rad = np.sqrt(x * x + y * y)
    alpha = np.arctan2(x, y)
    l1 = center - diameter / 2
    l2 = center - diameter / 2 + rad

    roidata = np.ones((dx, dy, diameter), dtype='float32')

    delphi = np.pi / dx
    for m in range(dx):

        # Calculate start end coordinates for each row of the sinogram.
        ind1 = np.ceil(np.cos(alpha - m * delphi) * (l2 - l1) + l1)
        ind2 = np.floor(np.cos(alpha - m * delphi) * (l2 - l1) + l1 + diameter)

        # Make sure everythin is inside the frame.
        if ind1 < 0:
            ind1 = 0
        if ind1 > dz:
            ind1 = dz
        if ind2 < 0:
            ind2 = 0
        if ind2 > dz:
            ind2 = dz

        roidata[m, :, 0:(ind2 - ind1)] = data[m:m+1, :, ind1:ind2]
    return roidata

def sino_360_to_180(data, overlap=0, rotation='left'):
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.

    Parameters
    ----------
    data : ndarray
        Input 3D data.

    overlap : scalar, optional
        Overlapping number of pixels.

    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.

    Returns
    -------
    ndarray
        Output 3D data.
    """
    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))

    lo = overlap//2
    ro = overlap - lo
    n = dx//2

    out = np.zeros((n, dy, 2*dz-overlap), dtype=data.dtype)

    if rotation == 'left':
        out[:, :, -(dz-lo):] = data[:n, :, lo:]
        out[:, :, :-(dz-lo)] = data[n:2*n, :, ro:][:, :, ::-1]
    elif rotation == 'right':
        out[:, :, :dz-lo] = data[:n, :, :-lo]
        out[:, :, dz-lo:] = data[n:2*n, :, :-ro][:, :, ::-1]

    return out

#For backward compatibility
sino_360_t0_180 = sino_360_to_180