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
Module for describing beamline/experiment specific data recipes.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import re
import h5py
import tomopy.io.data as iod
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['read_als832',
           'read_aps2bm',
           'read_aps7bm',
           'read_aps13id',
           'read_aps32id']


def read_als832(fname):
    """
    Read ALS 8.3.2 standard data format.

    Parameters
    ----------
    fname : str
        Path to file name without indices and extension.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3D white field data.

    ndarray
        3D dark field data.

    array
        Projection angles in radian.
    """
    fname = os.path.abspath(fname)
    tname = fname + '_0000_'
    wname = fname + 'bak_'
    dname = fname + 'drk_'
    lname = fname + '.sct'

    # Read ALS log file data
    lfile = open(lname, 'r')
    for line in lfile:
        if '-nangles' in line:
            nang = int(re.findall(r'\d+', line)[0])
        if '-arange' in line:
            rang = int(re.findall(r'\d+', line)[0])
        if '-num_bright_field' in line:
            nflt = int(re.findall(r'\d+', line)[0])
        if '-num_dark_fields' in line:
            ndrk = int(re.findall(r'\d+', line)[0])
    lfile.close()

    data = iod.read_tiff_stack(tname, span=(0, nang-1), digit=4, ext='tif')
    white = iod.read_tiff_stack(wname, span=(0, nflt-1), digit=4, ext='tif')
    dark = iod.read_tiff_stack(dname, span=(0, ndrk-1), digit=4, ext='tif')
    theta = np.linspace(0, rang*np.pi/180., nang, dtype='float32')

    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
    if not isinstance(white, np.float32):
        white = np.array(white, dtype='float32')
    if not isinstance(dark, np.float32):
        dark = np.array(dark, dtype='float32')

    return data, white, dark, theta


def read_aps2bm(fname, proj=None, sino=None):
    """
    Read APS 2-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read.

    sino : slice, optional
        Specifies the sinograms to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3D white field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    f = h5py.File(fname, "r")
    data = f['/exchange/data']
    white = f['/exchange/data_white']
    dark = f['/exchange/data_dark']

    # Slice projection and sinogram axes.
    if proj is None:
        proj = slice(0, data.shape[0])
    if sino is None:
        sino = slice(0, data.shape[1])
    data = data[proj, sino, :].astype('float32')
    white = white[:, sino, :].astype('float32')
    dark = dark[:, sino, :].astype('float32')
    f.close()

    return data, white, dark


def read_aps7bm(fname, proj=None, sino=None):
    """
    Read APS 7-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read.

    sino : slice, optional
        Specifies the sinograms to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    array
        Projection angles in radian.
    """
    fname = os.path.abspath(fname)
    f = h5py.File(fname, "r")
    data = f['/exchange/data']
    theta = f['/exchange/theta']

    # Slice projection and sinogram axes.
    if proj is None:
        proj = slice(0, data.shape[0])
    if sino is None:
        sino = slice(0, data.shape[1])
    data = data[proj, sino, :].astype('float32')
    theta = theta[proj].astype('float32')
    f.close()

    return data, theta


def read_aps13id(fname, proj=None, sino=None):
    """
    Read APS 13-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read.

    sino : slice, optional
        Specifies the sinograms to read.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    fname = os.path.abspath(fname)
    f = h5py.File(fname, "r")
    data = f['/xrfmap/roimap/sum_cor']

    # Slice projection and sinogram axes.
    if proj is None:
        proj = slice(0, data.shape[1])
    if sino is None:
        sino = slice(0, data.shape[2])
    data = data[:, proj, sino].astype('float32')
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2).copy()
    f.close()

    return data


def read_aps32id(fname, proj=None, sino=None):
    """
    Read APS 32-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read.

    sino : slice, optional
        Specifies the sinograms to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3D white field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    f = h5py.File(fname, "r")
    data = f['/exchange/data']
    white = f['/exchange/data_white']
    dark = f['/exchange/data_dark']

    # Slice projection and sinogram axes.
    if proj is None:
        proj = slice(0, data.shape[0])
    if sino is None:
        sino = slice(0, data.shape[1])
    data = data[proj, sino, :].astype('float32')
    white = white[:, sino, :].astype('float32')
    dark = dark[:, sino, :].astype('float32')
    f.close()

    return data, white, dark
