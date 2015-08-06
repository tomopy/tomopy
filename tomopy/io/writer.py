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

from tomopy.misc.corr import adjust_range
import tomopy.util.dtype as dt
from skimage import io as sio
import warnings
import numpy as np
import os
import h5py
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__credits__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['write_hdf5',
           'write_npy',
           'write_tiff',
           'write_tiff_stack']


def get_body(fname, digit=None):
    """
    Get file name after extension removed.
    """
    body = os.path.splitext(fname)[0]
    if digit is not None:
        body = ''.join(body[:-digit])
    return body


def get_extension(fname):
    """
    Get file extension.
    """
    return '.' + fname.split(".")[-1]


def _init_dirs(fname):
    """
    Initialize directories for saving output files.

    Parameters
    ----------
    fname : str
        Output file name.
    """
    dname = os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(dname):
        os.makedirs(dname)


def _suggest_new_fname(fname, digit):
    """
    Suggest new string with an attached (or increased) value indexing
    at the end of a given string.

    For example if "myfile.tiff" exist, it will return "myfile-1.tiff".

    Parameters
    ----------
    fname : str
        Output file name.
    digit : int, optional
        Number of digits in indexing stacked files.

    Returns
    -------
    str
        Indexed new string.
    """
    if os.path.isfile(fname):
        body = get_body(fname)
        ext = get_extension(fname)
        indq = 1
        file_exist = False
        while not file_exist:
            fname = body + '-' + '{0:0={1}d}'.format(indq, digit) + ext
            if not os.path.isfile(fname):
                file_exist = True
            else:
                indq += 1
    return fname


def _init_write(arr, fname, ext, dtype, overwrite):
    if not isinstance(fname, basestring):
        fname = 'tmp/data' + ext
    else:
        if not fname.endswith(ext):
            fname = fname + ext
    fname = os.path.abspath(fname)
    if not overwrite:
        fname = _suggest_new_fname(fname, digit=1)
    _init_dirs(fname)
    if dtype is not None:
        arr = dt.as_dtype(arr, dtype)
    return fname, arr


def write_hdf5(
        data, fname='tmp/data.tiff', gname='exchange',
        dtype=None, overwrite=False):
    """
    Write data to hdf5 file in a specific group.

    Parameters
    ----------
    data : ndarray
        Array data to be saved.
    fname : str
        File name to which the data is saved. ``.h5`` extension
        will be appended if it does not already have one.
    gname : str, optional
        Path to the group inside hdf5 file where data will be written.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    overwrite: bool, optional
        if True, overwrites the existing file if the file exists.
    """
    fname, data = _init_write(data, fname, '.h5', dtype, overwrite)
    f = h5py.File(fname, 'w')
    ds = f.create_dataset('implements', data=gname)
    exchangeGrp = f.create_group(gname)
    ds = exchangeGrp.create_dataset('data', data=data)
    f.close()


def write_npy(
        data, fname='tmp/data.npy', dtype=None, overwrite=False):
    """
    Write data to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    data : ndarray
        Array data to be saved.
    fname : str
        File name to which the data is saved. ``.npy`` extension
        will be appended if it does not already have one.
    """
    fname, data = _init_write(data, fname, '.npy', dtype, overwrite)
    np.save(fname, data)


def write_tiff(
        data, fname='tmp/data.tiff', dtype=None, overwrite=False):
    """
    Write image data to a tiff file.

    Parameters
    ----------
    data : ndarray
        Array data to be saved.
    fname : str
        File name to which the data is saved. ``.tiff`` extension
        will be appended if it does not already have one.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    overwrite: bool, optional
        if True, overwrites the existing file if the file exists.
    """
    fname, data = _init_write(data, fname, '.tiff', dtype, overwrite)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sio.imsave(fname, data, plugin='tifffile')


def write_tiff_stack(
        data, fname='tmp/data.tiff', dtype=None, axis=0, digit=5,
        start=0, overwrite=False):
    """
    Write data to stack of tiff file.

    Parameters
    ----------
    data : ndarray
        Array data to be saved.
    fname : str
        Base file name to which the data is saved. ``.tiff`` extension
        will be appended if it does not already have one.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    axis : int, optional
        Axis along which stacking is performed.
    start : int, optional
        First index of file in stack for saving.
    digit : int, optional
        Number of digits in indexing stacked files.
    overwrite: bool, optional
        if True, overwrites the existing file if the file exists.
    """
    fname, data = _init_write(data, fname, '.tiff', dtype, True)
    body = get_body(fname)
    ext = get_extension(fname)
    _data = np.swapaxes(data, 0, axis)
    for m in range(start, start + data.shape[axis]):
        _fname = body + '_' + '{0:0={1}d}'.format(m, digit) + ext
        if not overwrite:
            _fname = _suggest_new_fname(_fname, digit=1)
        write_tiff(_data[m - start], _fname, overwrite=overwrite)
