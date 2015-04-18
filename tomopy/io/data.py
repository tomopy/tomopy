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
import spefile
import netCDF4
import EdfFile
# import DM3lib # TODO: build fails
# import tifffile # TODO: rewrite conda recipe
import logging
import warnings


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['as_shared_array',
           'as_float32',
           'as_uint8',
           'as_uint16',
           'remove_neg',
           'remove_nan',
           'read_hdf5',
           'read_edf',
           # 'read_dm3',
           'read_spe',
           'read_netcdf4',
           'read_stack',
           'write_hdf5',
           'write_tiff_stack']


def as_shared_array(arr):
    """
    Converts array to a shared memory array.

    Parameters
    ----------
    arr : ndarray
        Input array.

    Returns
    -------
    ndarray
        Output array.
    """
    sarr = mp.Array(ctypes.c_float, arr.size)
    sarr = np.frombuffer(sarr.get_obj(), dtype='float32')
    sarr = np.reshape(sarr, arr.shape)
    sarr[:] = arr
    return sarr


def as_float32(arr):
    """
    Convert a numpy array to float32.

    Parameters
    ----------
    arr : ndarray
        Input array.

    Returns
    -------
    ndarray
        Output array.
    """
    if not isinstance(arr, np.float32):
        arr = np.array(arr, dtype='float32')
    return arr


def as_uint8(arr, dmin=None, dmax=None):
    """
    Convert a numpy array to uint8.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dmin : float, optional
        Mininum value to adjust float-to-int conversion range.
    dmax : float, optional
        Maximum value to adjust float-to-int conversion range.

    Returns
    -------
    ndarray
        Output array.
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
    Convert a numpy array to uint16.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dmin : float, optional
        Mininum value to adjust float-to-int conversion range.
    dmax : float, optional
        Maximum value to adjust float-to-int conversion range.

    Returns
    -------
    ndarray
        Output array.
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
    Replace negative values in data with a given value.

    Parameters
    ----------
    dat : ndarray
        Input data.
    val : float, optional
        Values to be replaced with negative values in data.

    Returns
    -------
    ndarray
       Corrected data.
    """
    dat = as_float32(dat)
    dat[dat < 0.0] = val
    return dat


def remove_nan(dat, val=0.):
    """
    Replace NaN values in data with a given value.

    Parameters
    ----------
    dat : ndarray
        Input data.
    val : float, optional
        Values to be replaced with NaN values in data.

    Returns
    -------
    ndarray
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
    string : str
        Given string (typically a file name).
    ind : int
        A value index to be added at the end of string.
    digit : int
        Number of digits in indexing tiff images.

    Returns
    -------
    str
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
    Suggest new string with an attached (or increased) value indexing
    at the end of a given string.

    For example if "myfile.tiff" exist, it will return "myfile-1.tiff".

    Parameters
    ----------
    fname : str
        Given string.

    Returns
    -------
    str
        Indexed new string.
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


def read_stack(bfname, ind, digit, format, ext=None):
    """
    Read data from a 2D image stack in a folder.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.
    ind : list of int
        Indices of the files to read.
    digit : int
        Number of digits used in indexing images.
    format : str, optional
        Data format. 'tif', 'tifc'
    ext : str, optional
        Extension of the files. 'tif'

    Returns
    -------
    ndarray
        Data.
    """
    if ext is None:
        ext = format
    d = ['0' * (digit - x - 1) for x in range(digit)]
    a = 0
    for m in ind:
        for n in range(digit):
            if m < np.power(10, n + 1):
                fname = bfname + d[n] + str(m) + '.' + ext
                if format is 'tiff' or format is 'tif':
                    _arr = _Format(fname).tiff()
                if format is 'tiffc' or format is 'tifc':
                    _arr = _Format(fname).tiffc()
                if a == 0:
                    dx = len(ind)
                    dy, dz = _arr.shape
                    arr = np.zeros((dx, dy, dz))
                arr[a] = _arr
                a += 1
                break
    return arr


def read_edf(fname, dim1=None, dim2=None, dim3=None):
    """
    Read data from a edf file.

    Parameters
    ----------
    fname : str
        Path to edf file.
    dim1, dim2, dim3 : slice, optional
        Slice object representing the set of indices along the
        1st, 2nd and 3rd dimensions respectively.

    Returns
    -------
    ndarray
        Data.
    """
    return _Format(fname).edf(dim1, dim2, dim3)


def read_hdf5(fname, gname, dim1=None, dim2=None, dim3=None):
    """
    Read data from hdf5 file from a specific group.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.
    gname : str
        Path to the group inside hdf5 file where data is located.
    dim1, dim2, dim3 : slice, optional
        Slice object representing the set of indices along the
        1st, 2nd and 3rd dimensions respectively.

    Returns
    -------
    ndarray
        Data.
    """
    return _Format(fname).hdf5(gname, dim1, dim2, dim3)


# def read_dm3(fname, gname, dim1=None, dim2=None, dim3=None):
#     """
#     Read data from GATAN DM3 (DigitalMicrograph) file.

#     Parameters
#     ----------
#     fname : str
#         Path to hdf5 file.
#     dim1, dim2, dim3 : slice, optional
#         Slice object representing the set of indices along the
#         1st, 2nd and 3rd dimensions respectively.

#     Returns
#     -------
#     ndarray
#         Data.
#     """
#     return _Format(fname).dm3(dim1, dim2, dim3)


def read_spe(fname, dim1=None, dim2=None, dim3=None):
    """
    Read data from a spe file.

    Parameters
    ----------
    fname : str
        Path to spe file.
    dim1, dim2, dim3 : slice, optional
        Slice object representing the set of indices along the
        1st, 2nd and 3rd dimensions respectively.

    Returns
    -------
    ndarray
        Data.
    """
    return _Format(fname).spe(dim1, dim2, dim3)


def read_netcdf4(fname, dim1=None, dim2=None, dim3=None):
    """
    Read data from a netcdf file.

    Parameters
    ----------
    fname : str
        Path to spe file.
    dim1, dim2, dim3 : slice, optional
        Slice object representing the set of indices along the
        1st, 2nd and 3rd dimensions respectively.

    Returns
    -------
    ndarray
        Data.
    """
    return _Format(fname).netcdf4(dim1, dim2, dim3)


def write_hdf5(data, fname, gname="exchange", overwrite=False):
    """
    Write data to a hdf5 file in a specific group.

    Parameters
    ----------
    data : ndarray
        Input data.
    fname : str
        Path to hdf5 file without extension.
    gname : str, optional
        Path to the group inside hdf5 file where data is located.
    overwrite: bool, optional
        if True, the existing files in the reconstruction folder will be
        overwritten with the new ones.
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


def write_tiff_stack(
        data, fname, axis=0, id=0,
        digit=5, overwrite=False,
        dtype='uint8', dmin=None, dmax=None):
    """
    Write 3D data to a stack of tiff files.

    Parameters
    ----------
    data : ndarray
        Input data as 3D array.
    fname : str
        Path of output file without extension.
    axis : int, optional
        Axis along which saving is performed.
    id : int, optional
        First index of file for saving.
    digit : int, optional
        Number of digits in indexing tiff images.
    overwrite: bool, optional
        if True, the existing files in the reconstruction folder will be
        overwritten with the new ones.
    dtype : str, optional
        The desired data-type for saved images.
    dmin : float, optional
        Minimum value in data for scaling before saving.
    dmax : float, optional
        Maximum value in data for scaling before saving.
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


class _Format():

    """
    Helper class for reading data from various data formats.

    Attributes
    ----------
    fname : str
        String defining the path or file name.
    """

    def __init__(self, fname):
        fname = os.path.abspath(fname)
        self.fname = fname

    def hdf5(self, gname, dim1=None, dim2=None, dim3=None):
        """
        Read data from hdf5 file from a specific group.

        Parameters
        ----------
        gname : str
            Path to the group inside hdf5 file where data is located.
        dim1, dim2, dim3 : slice, optional
            Slice object representing the set of indices along the
            1st, 2nd and 3rd dimensions respectively.

        Returns
        -------
        ndarray
            Data.
        """
        f = h5py.File(self.fname, "r")
        arr = f[gname]
        arr = self._slice_array(arr, dim1, dim2, dim3)
        f.close()
        return arr

    def tiff(self):
        """
        Read 2D tiff image.

        Returns
        -------
        ndarray
            Output 2D image.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = sio.imread(self.fname, plugin='tifffile')
        return arr

#     def tiffc(self):
#         """
#         Read 2D compressed tiff image.

#         Returns
#         -------
#         ndarray
#             Output 2D image.
#         """
#         print(self.fname)
#         f = tifffile.TiffFile(self.fname)
#         arr = f[0].asarray()
#         f.close()
#         return arr

    def spe(self, dim1=None, dim2=None, dim3=None):
        """
        Read data from a spe file.

        Parameters
        ----------
        dim1, dim2, dim3 : slice, optional
            Slice object representing the set of indices along the
            1st, 2nd and 3rd dimensions respectively.

        Returns
        -------
        ndarray
            Data.
        """
        f = spefile.PrincetonSPEFile(self.fname)
        arr = f.getData()
        arr = self._slice_array(arr, dim1, dim2, dim3)
        return arr

    # def dm3(self, dim1=None, dim2=None, dim3=None):
    #     """
    #     Read data from a dm3 file.

    #     Parameters
    #     ----------
    #     dim1, dim2, dim3 : slice, optional
    #         Slice object representing the set of indices along the
    #         1st, 2nd and 3rd dimensions respectively.

    #     Returns
    #     -------
    #     ndarray
    #         Data.
    #     """
    #     f = DM3lib.DM3(self.fname, dims=3)
    #     arr = f.imagedata
    #     arr = self._slice_array(arr, dim1, dim2, dim3)
    #     return arr

    def netcdf4(self, var, dim1=None, dim2=None, dim3=None):
        """
        Read data from a netcdf4 file.

        Parameters
        ----------
        var : str
            Variable name where data is stored.
        dim1, dim2, dim3 : slice, optional
            Slice object representing the set of indices along the
            1st, 2nd and 3rd dimensions respectively.

        Returns
        -------
        ndarray
            Data.
        """
        f = netCDF4.Dataset(self.fname, 'r')
        arr = f.variables[var]
        arr = self._slice_array(arr, dim1, dim2, dim3)
        f.close()
        return arr

    def edf(self, dim1=None, dim2=None, dim3=None):
        """
        Read data from a edf file.

        Parameters
        ----------
        var : str
            Variable name where data is stored.
        dim1, dim2, dim3 : slice, optional
            Slice object representing the set of indices along the
            1st, 2nd and 3rd dimensions respectively.

        Returns
        -------
        ndarray
            Data.
        """
        f = EdfFile.EdfFile(self.fname, access='r')
        d = f.GetStaticHeader(0)
        arr = np.empty((f.NumImages, int(d['Dim_2']), int(d['Dim_1'])))
        for (i, ar) in enumerate(arr):
            arr[i::] = f.GetData(i)
        arr = self._slice_array(arr, dim1, dim2, dim3)
        return arr

    def _slice_array(self, arr, dim1=None, dim2=None, dim3=None):
        """
        Perform slicing on ndarray.

        Parameters
        ----------

        dim1, dim2, dim3 : slice, optional
            Slice object representing the set of indices along the
            1st, 2nd and 3rd dimensions respectively.

        Returns
        -------
        ndarray
            Sliced array.
        """
        if len(arr.shape) == 1:
            if dim1 is None:
                dim1 = slice(0, arr.shape[0])
            arr = arr[dim1]
        elif len(arr.shape) == 2:
            if dim1 is None:
                dim1 = slice(0, arr.shape[0])
            if dim2 is None:
                dim2 = slice(0, arr.shape[1])
            arr = arr[dim1, dim2]
        elif len(arr.shape) == 3:
            if dim1 is None:
                dim1 = slice(0, arr.shape[0])
            if dim2 is None:
                dim2 = slice(0, arr.shape[1])
            if dim3 is None:
                dim3 = slice(0, arr.shape[2])
            arr = arr[dim1, dim2, dim3]
        else:
            arr = arr[:]
        return arr
