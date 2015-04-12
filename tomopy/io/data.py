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
Module for data I/O.
"""

from __future__ import absolute_import, division, print_function

from skimage import io as sio
import numpy as np
import multiprocessing as mp
import ctypes
import os
import h5py
import logging
import warnings


__docformat__ = 'restructuredtext en'
__all__ = ['as_shared_array',
           'as_float32',
           'as_uint8',
           'as_uint16',
           'remove_neg',
           'remove_nan',
           'read_hdf5',
           'write_hdf5',
           'read_tiff_stack',
           'write_tiff_stack']


def as_shared_array(arr):
    """
    Converts array to a shared memory array.

    Parameters
    ----------
    arr : ndarray
        Input array

    Returns
    -------
    out : ndarray
        Output array
    """
    sarr = mp.Array(ctypes.c_float, arr.size)
    sarr = np.frombuffer(sarr.get_obj(), dtype='float32')
    sarr = np.reshape(sarr, arr.shape)
    sarr[:] = arr
    return sarr


def as_float32(arr):
    """
    Convert a numpy array to ``float32``.

    Parameters
    ----------
    arr : ndarray
        Input array

    Returns
    -------
    out : ndarray
        Output array
    """
    if not isinstance(arr, np.float32):
        arr = np.array(arr, dtype='float32')
    return arr


def as_uint8(arr, dmin=None, dmax=None):
    """
    Convert a numpy array to ``uint8``.

    Parameters
    ----------
    arr : ndarray
        Input array

    dmin : scalar
        Mininum value to adjust float-to-int conversion dynamic range

    dmax : scalar
        Maximum value to adjust float-to-int conversion dynamic range

    Returns
    -------
    out : ndarray
        Output array
    """
    if not isinstance(arr, np.int8):
        arr = arr.copy()
        if dmax is None:
            dmax = np.max(arr)
        if dmin is None:
            dmin = np.min(arr)
        if dmax < np.max(arr):
            arr[arr > dmax] = dmax
        if dmin > np.min(arr):
            arr[arr < dmin] = dmin
        if dmax == dmin:
            arr = arr.astype('uint8')
        else:
            arr = ((arr * 1.0 - dmin) / (dmax - dmin) * 255).astype('uint8')
    return arr


def as_uint16(arr, dmin=None, dmax=None):
    """
    Convert a numpy array to ``uint16``.

    Parameters
    ----------
    arr : ndarray
        Input array

    dmin : scalar
        Mininum value to adjust float-to-int conversion dynamic range

    dmax : scalar
        Maximum value to adjust float-to-int conversion dynamic range

    Returns
    -------
    out : ndarray
        Output array
    """
    if not isinstance(arr, np.int16):
        arr = arr.copy()
        if dmax is None:
            dmax = np.max(arr)
        if dmin is None:
            dmin = np.min(arr)
        if dmax < np.max(arr):
            arr[arr > dmax] = dmax
        if dmin > np.min(arr):
            arr[arr < dmin] = dmin
        if dmax == dmin:
            arr = arr.astype('uint16')
        else:
            arr = ((arr * 1.0 - dmin) / (dmax - dmin) * 65535).astype('uint16')
    return arr


def remove_neg(dat, val=0.):
    """
    Replace negative values in data with ``val``.

    Parameters
    ----------
    dat : ndarray
        Input data.

    val : scalar
        Values to be replaced with negative values in data.

    Returns
    -------
    out : ndarray
       Corrected data.
    """
    dat = as_float32(dat)
    dat[dat < 0.0] = val
    return dat


def remove_nan(dat, val=0.):
    """
    Replace NaN values in data with ``val``.

    Parameters
    ----------
    dat : ndarray
        Input data.

    val : scalar
        Values to be replaced with NaN values in data.

    Returns
    -------
    out : ndarray
       Corrected data.
    """
    dat = as_float32(dat)
    dat[np.isnan(dat)] = val
    return dat


def _add_index_to_string(string, ind, digit):
    """
    Add index to a string, usually for image stacking purposes.

    For example if strng is "mydata", ind is 134 and digit is 5,
    the output string is "mydata-00134".

    Parameters
    ----------
    string : string
        Given string.

    ind : scalar
        The index to be added at the end of string.

    digit : scalar
        Number of digit for the string indexing.

    Returns
    -------
    out : string
        Indexed string.
    """
    # Index for stacking.
    string_index = ["" for x in range(digit)]
    for m in range(digit):
        string_index[m] = '0' * (digit - m - 1)

    # This is to keep the digit size for various numbers.
    # E.g. 00123 includes 2 zeros and a 3 digit number
    for n in range(digit):
        if ind < np.power(10, n + 1):
            string += '_' + string_index[n] + str(ind)
            return string


def _suggest_new_fname(fname):
    """
    Suggests a new file name with an attached (or increased) index
    at the end of the file name.

    For example if "myfile.tiff" exist, it will return "myfile-1.tiff".

    Parameters
    ----------
    string : string
        Given string.

    Returns
    -------
    out : string
        Indexed new file.
    """
    ext = '.' + fname.split(".")[-1]
    fname = fname.split(".")[-2]
    indq = 1
    _flag = False
    while not _flag:
        _fname = fname + '-' + str(indq)
        if not os.path.isfile(_fname + ext):
            _flag = True
            fname = _fname
        else:
            indq += 1
    fname += ext
    return fname


def read_hdf5(fname, gname="/exchange/data", dtype='float32'):
    """
    Read data from a hdf5 file from a specific group.

    Parameters
    ----------
    fname : string
        Path to hdf5 file.

    gname : string
        Path to group where data is.

    Returns
    -------
    out : ndarray (probably)
        Returned data.
    """
    fname = os.path.abspath(fname)
    f = h5py.File(fname, "r")
    data = f[gname][:].astype(dtype)
    f.close()
    return data


def write_hdf5(data, fname, gname="exchange", overwrite=False):
    """
    Write data to a hdf5 file in a specific group.

    Parameters
    ----------
    data : ndarray (probably)
        Input data.

    fname : string
        Path to hdf5 file without extension.

    gname : string
        Path to group where data is.

    overwrite: bool, optional
        if True the existing files in the reconstruction folder will be
        overwritten with the new ones.

    Returns
    -------
    out : ndarray (probably)
        Returned data.
    """
    fname += '.h5'
    if not overwrite:
        if os.path.isfile(fname):
            fname = _suggest_new_fname(fname)

    f = h5py.File(fname, 'w')
    ds = f.create_dataset('implements', data="exchange")
    exchangeGrp = f.create_group(gname)
    ds = exchangeGrp.create_dataset('data', data=data)
    f.close()


def read_tiff_stack(fname, span, digit, ext='tiff'):
    """
    Read data from a tiff stack in a folder.

    Parameters
    ----------
    fname : string
        Path to hdf5 file.

    span : list
        (start, end) indices of the files to read.

    digit : scalar
        Determines the number of digits in indexing tiff images.

    ext : string
        Specifies the extension of tiff files (e.g., tiff or tif). 

    Returns
    -------
    out : ndarray
        Returned data.
    """
    d = ['0' * (digit - x - 1) for x in range(digit)]
    ind = range(span[0], span[1] + 1)
    for m in ind:
        for n in range(digit):
            if m < np.power(10, n + 1):
                img = fname + d[n] + str(m) + '.' + ext
                break

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = skimage_io.imread(img, plugin='tifffile')

            if a == 0:
                sx = ind.size
                sy, sz = out.shape
                data = np.zeros((sx, sy, sz), dtype='float32')

            data[m] = out
    return data


def write_tiff_stack(
        data, fname, axis=0, id=0,
        digit=5, overwrite=False,
        dtype='uint8', dmin=None, dmax=None):
    """
    Write 3-D data to a stack of tiff files.

    Parameters
    -----------
    data : ndarray
        3-D input data.

    fname : str, optional
        Path of the output file without any extensions.

    id : scalar, optional
        First index of the data on first dimension of the array.

    digit : scalar, optional
        Number of digit used for file indexing.
        For example if 4: test_XXXX.tiff

    axis : scalar, optional
        Images are read along that axis.

    overwrite: bool, optional
        if True the existing files in the reconstruction folder will be
        overwritten with the new ones.

    dtype : bool, optional
        Export data type precision.

    dmin, dmax : scalar, optional
        User defined minimum and maximum values in the data that will be
        used to scale the dataset when saving.
    """
    ext = '.tiff'
    fname = os.path.abspath(fname)
    dir_path = os.path.dirname(fname)

    # Find max min of data for scaling before int conversion
    if dmax is None:
        dmax = np.max(data)
    if dmin is None:
        dmin = np.min(data)
    if dmax < np.max(data):
        data[data > dmax] = dmax
    if dmin > np.min(data):
        data[data < dmin] = dmin

    # Create new folders.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Select desired x from whole data.
    nx, ny, nz = data.shape
    if axis == 0:
        end_id = id + nx
    elif axis == 1:
        end_id = id + ny
    elif axis == 2:
        end_id = id + nz

    # Range of indices for stacking tiffs
    ind = range(id, end_id)
    for m in range(len(ind)):
        string = _add_index_to_string(fname, ind[m], digit)
        new_fname = string + ext
        if not overwrite:
            if os.path.isfile(new_fname):
                new_fname = _suggest_new_fname(new_fname)

        if axis == 0:
            arr = data[m, :, :]
        elif axis == 1:
            arr = data[:, m, :]
        elif axis == 2:
            arr = data[:, :, m]

        if dtype is 'uint8':
            arr = as_uint8(arr, dmin, dmax)
        elif dtype is 'uint16':
            arr = as_uint16(arr, dmin, dmax)
        elif dtype is 'float32':
            arr = as_float32(arr)

        # Save as tiff
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sio.imsave(new_fname, arr, plugin='tifffile')
