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
__all__ = ['read_als_832',
           'read_anka_tomotopo',
           'read_aps_2bm',
           'read_aps_7bm',
           'read_aps_13bm',
           'read_aps_13id',
           'read_aps_32id',
           'read_aus_microct',
           'read_diamond_l12',
           'read_petra3_p05',
           'read_sls_tomcat']


def read_als_832(fname, ind=None):
    """
    Read ALS 8.3.2 standard data format.

    Parameters
    ----------
    fname : str
        Path to file name without indices and extension.

    ind : list of int, optional
        Indices of the projection files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    # File definitions.
    fname = os.path.abspath(fname)
    tomo_name = fname + '_0000_'
    flat_name = fname + 'bak_'
    dark_name = fname + 'drk_'
    log_file = fname + '.sct'

    # Read metadata from ALS log file.
    contents = open(log_file, 'r')
    for line in contents:
        if '-nangles' in line:
            nproj = int(re.findall(r'\d+', line)[0])
        if '-num_bright_field' in line:
            nflat = int(re.findall(r'\d+', line)[0])
        if '-num_dark_fields' in line:
            ndark = int(re.findall(r'\d+', line)[0])
    contents.close()

    if ind is None:
        ind = range(0, nproj)

    tomo = iod.read_stack(tomo_name, ind, digit=4, format='tif')
    white = iod.read_stack(flat_name, range(0, nflat), digit=4, format='tif')
    dark = iod.read_stack(dark_name, range(0, ndark), digit=4, format='tif')
    return tomo, white, dark


def read_anka_tomotopo(fname, ind_tomo, ind_flat, ind_dark):
    """
    Read ANKA TOMO-TOMO standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder name without indices and extension.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    ind_flat : list of int, optional
        Indices of the flat field files to read.

    ind_dark : list of int, optional
        Indices of the dark field files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    tomo_name = fname + '/radios/image_'
    flat_name = fname + '/flats/image_'
    dark_name = fname + '/darks/image_'
    tomo = iod.read_stack(tomo_name, ind_tomo, digit=5, format='tif')
    flat = iod.read_stack(flat_name, ind_flat, digit=5, format='tif')
    dark = iod.read_stack(dark_name, ind_dark, digit=5, format='tif')
    return tomo, flat, dark


def read_aps_2bm(fname, proj=None, sino=None):
    """
    Read APS 2-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read from a slice object.

    sino : slice, optional
        Specifies the sinograms to read from a slice object.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    tomo = iod.read_hdf5(fname, '/exchange/data', proj, sino)
    flat = iod.read_hdf5(fname, '/exchange/data_white', proj, sino)
    dark = iod.read_hdf5(fname, '/exchange/data_dark', proj, sino)
    return tomo, flat, dark


def read_aps_7bm(fname, proj=None, sino=None):
    """
    Read APS 7-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read from a slice object.

    sino : slice, optional
        Specifies the sinograms to read from a slice object.

    Returns
    -------
    ndarray
        3D tomographic data.

    array
        Projection angles in radian.
    """
    fname = os.path.abspath(fname)
    f = h5py.File(fname, "r")
    tomo = iod.read_hdf5(fname, '/exchange/data', proj, sino)
    theta = iod.read_hdf5(fname, '/exchange/theta', proj)
    return tomo, theta


def read_aps_13bm(fname, format, proj=None, sino=None):
    """
    Read APS 13-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    format : str
        Data format. 'spe' or 'netcdf4'

    proj : slice, optional
        Specifies the projections to read from a slice object.

    sino : slice, optional
        Specifies the sinograms to read from a slice object.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    fname = os.path.abspath(fname)
    if format is 'spe':
        tomo = iod.read_spe(fname, proj, sino)
    elif format is 'netcdf4':
        tomo = iod.read_netcdf4(fname, 'array_data', proj, sino)
    return tomo


def read_aps_13id(fname, gname='/xrfmap/roimap/sum_cor', proj=None, sino=None):
    """
    Read APS 13-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    gname : str, optional
        Path to the group inside hdf5 file where data is located.

    proj : slice, optional
        Specifies the projections to read from a slice object.

    sino : slice, optional
        Specifies the sinograms to read from a slice object.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    fname = os.path.abspath(fname)
    tomo = iod.read_hdf5(fname, gname, None, proj, sino)
    tomo = np.swapaxes(tomo, 0, 1)
    tomo = np.swapaxes(tomo, 1, 2).copy()
    return tomo


def read_aps_32id(fname, proj=None, sino=None):
    """
    Read APS 32-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : slice, optional
        Specifies the projections to read from a slice object.

    sino : slice, optional
        Specifies the sinograms to read from a slice object.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    tomo = iod.read_hdf5(fname, '/exchange/data', proj, sino)
    flat = iod.read_hdf5(fname, '/exchange/data_white', proj, sino)
    dark = iod.read_hdf5(fname, '/exchange/data_dark', proj, sino)
    return tomo, flat, dark


def read_aus_microct(fname, ind_tomo, ind_flat, ind_dark):
    """
    Read Australian Synchrotron Micro Computed Tomography standard
    data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    ind_flat : list of int, optional
        Indices of the flat field files to read.

    ind_dark : list of int, optional
        Indices of the dark field files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    tomo_name = fname + '/SAMPLE_T_'
    flat_name = fname + '/BG__BEFORE_'
    dark_name = fname + '/DF__BEFORE_'
    tomo = iod.read_stack(tomo_name, ind_tomo, digit=4, format='tif')
    flat = iod.read_stack(flat_name, ind_flat, digit=2, format='tif')
    dark = iod.read_stack(dark_name, ind_dark, digit=2, format='tif')
    return tomo, flat, dark


def read_diamond_l12(fname, ind):
    """
    Read Diamond Light Source L12 (JEEP) standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind : list of int, optional
        Indices of the projection files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.
    """
    fname = os.path.abspath(fname)
    tomo_name = fname + '/im_'
    flat_name = fname + '/flat_'
    tomo = iod.read_stack(tomo_name, ind, digit=6, format='tif')
    flat = iod.read_stack(flat_name, range(0, 1), digit=6, format='tif')
    return tomo, flat


def read_petra3_p05(fname, ind_tomo, ind_flat, ind_dark):
    """
    Read Petra-III P05 standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    ind_flat : list of int, optional
        Indices of the flat field files to read.

    ind_dark : list of int, optional
        Indices of the dark field files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    fname = os.path.abspath(fname)
    tomo_name = fname + '/scan_0002/ccd/pco01/ccd_'
    flat_name = fname + '/scan_0001/ccd/pco01/ccd_'
    dark_name = fname + '/scan_0003/ccd/pco01/ccd_'
    tomo = iod.read_stack(tomo_name, ind_tomo, digit=4, format='tif')
    flat = iod.read_stack(flat_name, ind_flat, digit=4, format='tif')
    dark = iod.read_stack(dark_name, ind_dark, digit=4, format='tif')
    return tomo, flat, dark


def read_sls_tomcat(fname, ind=None):
    """
    Read SLS TOMCAT standard data format.

    Parameters
    ----------
    fname : str
        Path to file name without indices and extension.

    ind : list of int, optional
        Indices of the projection files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    # File definitions.
    fname = os.path.abspath(fname)
    log_file = fname + '.log'

    # Read metadata from ALS log file.
    contents = open(log_file, 'r')
    for line in contents:
        ls = line.split()
        if len(ls) > 1:
            if (ls[0] == 'Number' and ls[2] == 'projections'):
                nproj = int(ls[4])
            elif (ls[0] == 'Number' and ls[2] == 'flats'):
                nflat = int(ls[4])
            elif (ls[0] == 'Number' and ls[2] == 'darks'):
                ndark = int(ls[4])
    contents.close()

    if ind is None:
        ind = range(ndark + nflat + 1, ndark + nflat + nproj)
    find = range(ndark + 1, ndark + nflat)
    dind = range(1, ndark)

    tomo = iod.read_stack(fname, ind, digit=4, format='tif')
    flat = iod.read_stack(fname, find, digit=4, format='tif')
    dark = iod.read_stack(fname, dind, digit=4, format='tif')
    return tomo, flat, dark
