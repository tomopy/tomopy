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
import os
import ctypes
import tomopy.misc.mproc as mp
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['apply_pad',
           'downsample',
           'upsample']


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


def apply_pad(arr, npad=None, axis=2, val=0.):
    """
    Extend size of 3D array along specified axis.

    Parameters
    ----------
    arr : ndarray
        3D input array.
    npad : int, optional
        New dimensions after padding.
    axis : int, optional
        Axis along which padding will be performed.
    val : float, optional
        Pad value.

    Returns
    -------
    ndarray
        Padded 3D array.
    """
    dx, dy, dz = arr.shape
    if axis == 0:
        if npad is None:
            npad = int(np.ceil(dx * np.sqrt(2)))
        elif npad < dx:
            npad = dx
        out = val * np.ones((npad, dy, dz), dtype='float32')
    if axis == 1:
        if npad is None:
            npad = int(np.ceil(dy * np.sqrt(2)))
        elif npad < dy:
            npad = dy
        out = val * np.ones((dx, npad, dz), dtype='float32')
    if axis == 2:
        if npad is None:
            npad = int(np.ceil(dz * np.sqrt(2)))
        elif npad < dz:
            npad = dz
        out = val * np.ones((dx, dy, npad), dtype='float32')

    # Make sure that input datatype is correct.
    if not isinstance(arr, np.float32):
        arr = np.array(arr, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.apply_pad.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.apply_pad(
        arr.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy), ctypes.c_int(dz),
        ctypes.c_int(axis), ctypes.c_int(npad),
        out.ctypes.data_as(c_float_p))
    return out


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
    dx, dy, dz = arr.shape
    if axis == 0:
        out = np.zeros((dx / np.power(2, level), dy, dz), dtype='float32')
    if axis == 1:
        out = np.zeros((dx, dy / np.power(2, level), dz), dtype='float32')
    if axis == 2:
        out = np.zeros((dx, dy, dz / np.power(2, level)), dtype='float32')

    # Make sure that input datatype is correct.
    if not isinstance(arr, np.float32):
        arr = np.array(arr, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.downsample.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.downsample(
        arr.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy), ctypes.c_int(dz),
        ctypes.c_int(level), ctypes.c_int(axis),
        out.ctypes.data_as(c_float_p))
    return out


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
    dx, dy, dz = arr.shape
    if axis == 0:
        out = np.zeros((dx * np.power(2, level), dy, dz), dtype='float32')
    if axis == 1:
        out = np.zeros((dx, dy * np.power(2, level), dz), dtype='float32')
    if axis == 2:
        out = np.zeros((dx, dy, dz * np.power(2, level)), dtype='float32')

    # Make sure that input datatype is correct.
    if not isinstance(arr, np.float32):
        arr = np.array(arr, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.upsample.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.upsample(
        arr.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy), ctypes.c_int(dz),
        ctypes.c_int(level), ctypes.c_int(axis),
        out.ctypes.data_as(c_float_p))
    return out
