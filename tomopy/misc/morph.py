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
import pywt
import logging
import os
import ctypes
import tomopy.misc.mproc as mp
from scipy.ndimage import filters


__docformat__ = 'restructuredtext en'
__all__ = ['apply_pad',
           'focus_region',
           'downsample2d',
           'downsample3d',
           'upsample2d',
           'upsample3d',
           'upsample2df']


def import_shared_lib(lname):
    """
    Import the C-shared library.
    """
    try:
        if os.name == 'nt':
            lname = 'lib/' + lname + '.pyd'
            libpath = os.path.join(os.path.dirname(__file__), lname)
            return ctypes.CDLL(os.path.abspath(libpath))
        else:
            lname = 'lib/' + lname + '.so'
            libpath = os.path.join(os.path.dirname(__file__), lname)
            return ctypes.CDLL(os.path.abspath(libpath))
    except OSError as e:
        pass

libtomopy_misc = import_shared_lib('libtomopy_misc')


def apply_pad(data, npad=None, val=0., ind=None):
    """
    Applies padding to each projection data.

    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    npad : scalar, int32
        New dimension of the projections
        after padding.

    val : scalar, float32
        Pad value.

    Returns
    -------
    padded : ndarray
        Padded data.
    """
    dx, dy, dz = data.shape
    if npad is None:
        npad = np.ceil(dz * np.sqrt(2))
    elif npad < dz:
        npad = dz

    npad = np.array(npad, dtype='int32')
    padded = val * np.ones((dx, dy, npad), dtype='float32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_prep.apply_padding.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_prep.apply_padding(
        data.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy),
        ctypes.c_int(dz), ctypes.c_int(npad),
        padded.ctypes.data_as(c_float_p))
    return padded


def focus_region(
        data, xcoord=0, ycoord=0,
        dia=256, center=None, pad=False, corr=True):
    """
    Uses only a portion of the sinogram for reconstructing
    a circular region of interest (ROI).

    Note: Only valid for 0-180 degree span data.

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    xcoord, ycoord : scalar
        X- and Y-coordinates of the center
        location of the circular ROI.

    dia : scalar
        dia of the circular ROI.

    center : scalar
        Center of rotation of the original dataset.

    pad : bool, optional
        True if the original sinogram size is preserved.

    corr : bool, optional
        True if the correct_drift is
        applied after ROI selection.

    Returns
    -------
    roi : ndarray
        Modified ROI data.
    """
    dx, dy, dz = data.shape
    ind = np.arange(0, dx)

    if center is None:
        center = dz / 2.

    rad = np.sqrt(xcoord * xcoord + ycoord * ycoord)
    alpha = np.arctan2(xcoord, ycoord)
    l1 = center - dia / 2
    l2 = center - dia / 2 + rad

    roi = np.ones((dx, dy, dia), dtype='float32')
    if pad:
        roi = np.ones((dx, dy, dz), dtype='float32')

    delphi = PI / dx
    for m in ind:
        ind1 = np.ceil(np.cos(alpha - m * delphi) * (l2 - l1) + l1)
        ind2 = np.floor(np.cos(alpha - m * delphi) * (l2 - l1) + l1 + dia)

        if ind1 < 0:
            ind1 = 0
        if ind2 < 0:
            ind2 = 0
        if ind1 > dz:
            ind1 = dz
        if ind2 > dz:
            ind2 = dz

        arr = np.expand_dims(data[m, :, ind1:ind2], axis=0)
        if pad:
            if corr:
                roi[m, :, ind1:ind2] = correct_air(arr.copy(), air=5)
            else:
                roi[m, :, ind1:ind2] = arr
        else:
            if corr:
                roi[m, :, 0:(ind2 - ind1)] = correct_air(arr, air=5)
            else:
                roi[m, :, 0:(ind2 - ind1)] = arr
        if not pad:
            center = dz / 2.
    return roi, center


def downsample2d(data, level=1):
    """
    Downsample the slices by binning.

    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    level : scalar, int32
        Downsampling level. For example level=2
        means, the sinogram will be downsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    output : ndarray
        Downsampled 3-D tomographic data with dimensions:
        [projections, slices/level^2, pixels]
    """
    dx, dy, dz = data.shape
    level = np.array(level, dtype='int32')

    binsize = np.power(2, level)
    downdat = np.zeros(
        (dx, dy, dz / binsize),
        dtype='float32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_prep.downsample2d.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_prep.downsample2d(
        data.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy),
        ctypes.c_int(dz), ctypes.c_int(level),
        downdat.ctypes.data_as(c_float_p))
    return downdat


def downsample3d(data, level=1):
    """
    Downsample the slices and pixels by binning.

    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    level : scalar, int32
        Downsampling level. For example level=2
        means, the sinogram will be downsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    downdat : ndarray
        Downsampled 3-D tomographic data with dimensions:
        [projections, slices/level^2, pixels/level^2]
    """
    dx, dy, dz = data.shape
    level = np.array(level, dtype='int32')

    if level < 0:
        return data

    binsize = np.power(2, level)
    downdat = np.zeros(
        (dx, dy / binsize, dz / binsize),
        dtype='float32')

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_prep.downsample3d.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_prep.downsample3d(
        data.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy),
        ctypes.c_int(dz), ctypes.c_int(level),
        downdat.ctypes.data_as(c_float_p))
    return downdat


def upsample2d(data, level=1):
    """
    Upsample the slices.

    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]

    level : scalar, int32
        Upsampling level. For example level=2
        means, the sinogram will be upsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    output : ndarray
        Downsampled reconstructed 3-D data with dimensions:
        [slices, pixels*level^2, pixels*level^2]
    """
    dx, dy, dz = data.shape
    level = np.array(level, dtype='int32')

    if level < 0:
        return data

    binsize = np.power(2, level)
    updat = np.zeros((dy, dz * binsize, dz * binsize), dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_prep.upsample2d.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_prep.upsample2d(
        data.ctypes.data_as(c_float_p),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_int(level),
        updat.ctypes.data_as(c_float_p))
    return updat


def upsample3d(data, level=1):
    """
    Upsample the slices and pixels.

    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]

    level : scalar, int32
        Upsampling level. For example level=2
        means, the sinogram will be upsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    output : ndarray
        Downsampled reconstructed 3-D data with dimensions:
        [slices*level^2, pixels*level^2, pixels*level^2]
    """
    dx, dy, dz = data.shape
    level = np.array(level, dtype='int32')

    if level < 0:
        return data

    binsize = np.power(2, level)
    updat = np.zeros((dy, dz * binsize, dz * binsize), dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_prep.upsample3d.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_prep.upsample3d(
        data.ctypes.data_as(c_float_p),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_int(level),
        updat.ctypes.data_as(c_float_p))
    return updat


def upsample2df(data, level):
    """
    Upsample the slices in Fourier domain.

    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]

    level : scalar, int32
        Upsampling level. For example level=2
        means, the sinogram will be upsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    output : ndarray
        Downsampled reconstructed 3-D data with dimensions:
        [slices, pixels*level^2, pixels*level^2]

    """
    dx, dy, dz = data.shape
    level = np.array(level, dtype='int32')

    if level < 0:
        return data

    binsize = np.power(2, level)
    fftw2data = np.zeros((dz * binsize, dz * binsize), dtype='complex')
    updat = np.zeros((dy, dz * binsize, dz * binsize), dtype='float32')

    ind = slice(dz * (binsize - 1) / 2, dz * (binsize - 1) / 2 + dz, 1)
    for m in range(dy):
        fftw2data[ind, ind] = np.fft.fftshift(fftw2(data[m, :, :]))
        updat[m, :, :] = np.real(ifftw2(np.fft.ifftshift(fftw2data)))

    return updat
