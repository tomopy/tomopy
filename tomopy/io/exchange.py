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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os.path
import re
import tomopy.io.reader as tio
from tomopy.util.misc import deprecated
import logging
import warnings

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Luis Barroso-Luque"
__credits__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['read_als_832',
           'read_als_832h5',
           'read_anka_topotomo',
           'read_aps_1id',
           'read_aps_2bm',
           'read_aps_7bm',
           'read_aps_13bm',
           'read_aps_13id',
           'read_aps_32id',
           'read_aus_microct',
           'read_diamond_l12',
           'read_elettra_syrmep',
           'read_esrf_id19',
           'read_lnls_imx',
           'read_petraIII_p05',
           'read_sls_tomcat']


@deprecated
def read_als_832(fname, ind_tomo=None, normalized=False):
    """
    Read ALS 8.3.2 standard data format.

    Parameters
    ----------
    fname : str
        Path to file name without indices and extension.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    normalized : boolean
        If False, darks and flats will not be read. This should
        only be used for cases where tomo is already normalized.
        8.3.2 has a plugin that normalization is preferred to be
        done with prior to tomopy reconstruction.

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

    if not normalized:
        fname = fname.split(
            'output')[0] + fname.split('/')[len(fname.split('/')) - 1]
        tomo_name = fname + '_0000_0000.tif'
        flat_name = fname + 'bak_0000.tif'
        dark_name = fname + 'drk_0000.tif'
        log_file = fname + '.sct'
    else:
        if "output" not in fname:
            raise Exception(
                'Please provide the normalized output directory as input')
        tomo_name = fname + '_0.tif'
        fname = fname.split(
            'output')[0] + fname.split('/')[len(fname.split('/')) - 1]
        log_file = fname + '.sct'

    # Read metadata from ALS log file.
    contents = open(log_file, 'r')
    for line in contents:
        if '-nangles' in line:
            nproj = int(re.findall(r'\d+', line)[0])
        if '-num_bright_field' in line:
            nflat = int(re.findall(r'\d+', line)[0])
        if '-i0cycle' in line:
            inter_bright = int(re.findall(r'\d+', line)[1])
        if '-num_dark_fields' in line:
            ndark = int(re.findall(r'\d+', line)[0])
    contents.close()
    if ind_tomo is None:
        ind_tomo = list(range(0, nproj))
    if not normalized:
        ind_flat = list(range(0, nflat))
        if inter_bright > 0:
            ind_flat = list(range(0, nproj, inter_bright))
            flat_name = fname + 'bak_0000_0000.tif'
        ind_dark = list(range(0, ndark))

    # Read image data from tiff stack.
    tomo = tio.read_tiff_stack(tomo_name, ind=ind_tomo, digit=4)

    if not normalized:

        # Adheres to 8.3.2 flat/dark naming conventions:
        # ----Flats----
        # root_namebak_xxxx_yyyy
        # For datasets that take flat at the start and end of its scan,
        # xxxx is in incrementals of one, and yyyy is either 0000 or the
        # last projection. For datasets that take flat while they scan
        # (when the beam fluctuates during scans),
        # xxxx is always 0000, and yyyy is in intervals given by log file.

        if inter_bright == 0:
            a = [0, nproj - 1]
            list_flat = tio._list_file_stack(flat_name, ind_flat, digit=4)
            for x in ind_flat:
                body = os.path.splitext(list_flat[x])[0] + "_"
                ext = os.path.splitext(list_flat[x])[1]
                for y, z in enumerate(a):
                    y = body + '{0:0={1}d}'.format(z, 4) + ext
                    if z == 0:
                        list_flat[x] = y
                    else:
                        list_flat.append(y)
            list_flat = sorted(list_flat)
            for m, image in enumerate(list_flat):
                _arr = tio.read_tiff(image)
                if m == 0:
                    dx = len(ind_flat * 2)
                    dy, dz = _arr.shape
                    flat = np.zeros((dx, dy, dz))
                flat[m] = _arr
            flat = tio._slice_array(flat, None)
        else:
            flat = tio.read_tiff_stack(flat_name, ind=ind_flat, digit=4)

        # Adheres to 8.3.2 flat/dark naming conventions:
        # ----Darks----
        # root_namedrk_xxxx_yyyy
        # All datasets thus far that take darks at the start and end of
        # its scan, so xxxx is in incrementals of one, and yyyy is either
        # 0000 or the last projection.

        list_dark = tio._list_file_stack(dark_name, ind_dark, digit=4)
        for x in ind_dark:
            body = os.path.splitext(list_dark[x])[0] + '_'
            ext = os.path.splitext(list_dark[x])[1]
            body = body + '{0:0={1}d}'.format(nproj - 1, 4) + ext
            list_dark[x] = body
        list_dark = sorted(list_dark)
        for m, image in enumerate(list_dark):
            _arr = tio.read_tiff(image)
            if m == 0:
                dx = len(ind_dark)
                dy, dz = _arr.shape
                dark = np.zeros((dx, dy, dz))
            dark[m] = _arr
        dark = tio._slice_array(dark, None)
    else:
        flat = np.ones(1)
        dark = np.zeros(1)
    return tomo, flat, dark


@deprecated
def read_als_832h5(fname, ind_tomo=None, ind_flat=None, ind_dark=None,
                   proj=None, sino=None):
    """
    Read ALS 8.3.2 hdf5 file with stacked datasets.

    Parameters
    ----------

    fname : str
        Path to hdf5 file.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    ind_flat : list of int, optional
        Indices of the flat field files to read.

    ind_dark : list of int, optional
        Indices of the dark field files to read.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3D flat field data.

    ndarray
        3D dark field data.

    list of int
        Indices of flat field data within tomography projection list
    """

    dgroup = tio._find_dataset_group(fname)
    dname = dgroup.name.split('/')[-1]

    tomo_name = dname + '_0000_0000.tif'
    flat_name = dname + 'bak_0000.tif'
    dark_name = dname + 'drk_0000.tif'

    # Read metadata from dataset group attributes
    keys = list(dgroup.attrs.keys())
    if 'nangles' in keys:
        nproj = int(dgroup.attrs['nangles'])
    if 'i0cycle' in keys:
        inter_bright = int(dgroup.attrs['i0cycle'])
    if 'num_bright_field' in keys:
        nflat = int(dgroup.attrs['num_bright_field'])
    else:
        nflat = tio._count_proj(dgroup, flat_name, nproj,
                                inter_bright=inter_bright)
    if 'num_dark_fields' in keys:
        ndark = int(dgroup.attrs['num_dark_fields'])
    else:
        ndark = tio._count_proj(dgroup, dark_name, nproj)

    # Create arrays of indices to read projections, flats and darks
    if ind_tomo is None:
        ind_tomo = list(range(0, nproj))
    ind_dark = list(range(0, ndark))
    group_dark = [nproj-1]
    ind_flat = list(range(0, nflat))

    if inter_bright > 0:
        group_flat = list(range(0, nproj, inter_bright))
        if group_flat[-1] != nproj-1:
            group_flat.append(nproj-1)
    elif inter_bright == 0:
        group_flat = [0, nproj-1]
    else:
        group_flat = None

    tomo = tio.read_hdf5_stack(dgroup, tomo_name, ind_tomo, slc=(proj, sino))

    flat = tio.read_hdf5_stack(dgroup, flat_name, ind_flat, slc=(None, sino),
                               out_ind=group_flat)

    dark = tio.read_hdf5_stack(dgroup, dark_name, ind_dark, slc=(None, sino),
                               out_ind=group_dark)

    group_flat = tio._map_loc(ind_tomo, group_flat)

    return tomo, flat, dark, group_flat


@deprecated
def read_anka_topotomo(
        fname, ind_tomo, ind_flat, ind_dark, proj=None, sino=None):
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

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

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
    tomo_name = os.path.join(fname, 'radios', 'image_00000.tif')
    flat_name = os.path.join(fname, 'flats', 'image_00000.tif')
    dark_name = os.path.join(fname, 'darks', 'image_00000.tif')
    tomo = tio.read_tiff_stack(
        tomo_name, ind=ind_tomo, digit=5, slc=(sino, proj))
    flat = tio.read_tiff_stack(
        flat_name, ind=ind_flat, digit=5, slc=(sino, None))
    dark = tio.read_tiff_stack(
        dark_name, ind=ind_dark, digit=5, slc=(sino, None))
    return tomo, flat, dark


@deprecated
def read_aps_1id(fname, ind_tomo=None, proj=None, sino=None):
    """
    Read APS 1-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to file name without indices and extension.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

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
    _fname = fname + '000001.tif'
    log_file = os.path.dirname(fname) + os.path.sep + 'TomoStillScan.dat'

    # Read APS 1-ID log file data
    contents = open(log_file, 'r')
    for line in contents:
        ls = line.split()
        if len(ls) > 1:
            if ls[0] == "Tomography" and ls[1] == "scan":
                prj_start = int(ls[6])
            elif ls[0] == "Number" and ls[2] == "scan":
                nprj = int(ls[4])
            elif ls[0] == "Dark" and ls[1] == "field":
                dark_start = int(ls[6])
            elif ls[0] == "Number" and ls[2] == "dark":
                ndark = int(ls[5])
            elif ls[0] == "White" and ls[1] == "field":
                flat_start = int(ls[6])
            elif ls[0] == "Number" and ls[2] == "white":
                nflat = int(ls[5])
    contents.close()

    if ind_tomo is None:
        ind_tomo = list(range(prj_start, prj_start + nprj))
    ind_flat = list(range(flat_start, flat_start + nflat))
    ind_dark = list(range(dark_start, dark_start + ndark))
    tomo = tio.read_tiff_stack(_fname, ind=ind_tomo, digit=6, slc=(sino, proj))
    flat = tio.read_tiff_stack(_fname, ind=ind_flat, digit=6, slc=(sino, None))
    dark = tio.read_tiff_stack(_fname, ind=ind_dark, digit=6, slc=(sino, None))
    return tomo, flat, dark


@deprecated
def read_aps_2bm(fname, proj=None, sino=None):
    """
    Read APS 2-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    return read_aps_32id(fname, proj=proj, sino=sino)


@deprecated
def read_aps_7bm(fname, proj=None, sino=None):
    """
    Read APS 7-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.

    array
        Projection angles in radian.
    """
    tomo_grp = '/'.join(['exchange', 'data'])
    theta_grp = '/'.join(['exchange', 'theta'])
    tomo = tio.read_hdf5(fname, tomo_grp, slc=(proj, sino))
    theta = tio.read_hdf5(fname, theta_grp, slc=(proj, ))
    return tomo, theta


@deprecated
def read_aps_13bm(fname, format, proj=None, sino=None):
    """
    Read APS 13-BM standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    format : str
        Data format. 'spe' or 'netcdf4'

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    if format is 'spe':
        tomo = tio.read_spe(fname, slc=(None, sino))
    elif format is 'netcdf4':
        tomo = tio.read_netcdf4(fname, 'array_data', slc=(proj, sino))
    return tomo


@deprecated
def read_aps_13id(
        fname, group='/xrfmap/roimap/sum_cor', proj=None, sino=None):
    """
    Read APS 13-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    group : str, optional
        Path to the group inside hdf5 file where data is located.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    tomo = tio.read_hdf5(fname, group, slc=(None, proj, sino))
    tomo = np.swapaxes(tomo, 0, 1)
    tomo = np.swapaxes(tomo, 1, 2).copy()
    return tomo


@deprecated
def read_aps_32id(fname, exchange_rank=0, proj=None, sino=None, 
                  dtype=None, shared=True):
    """
    Read APS 32-ID standard data format.

    Parameters
    ----------
    fname : str
        Path to hdf5 file.

    exchange_rank : int, optional
        exchange_rank is added to "exchange" to point tomopy to the data
        to recontruct. if rank is not set then the data are raw from the
        detector and are located under exchange = "exchange/...", to process
        data that are the result of some intemedite processing step then
        exchange_rank = 1, 2, ... will direct tomopy to process
        "exchange1/...",

    proj : {sequence, int} or np.slice, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int} or np.slice, optional
        Specify sinograms to read. (start, end, step)

    dtype : numpy datatype, optional
        Convert data to this datatype on read if specified.

    shared : bool, optional
        If True, read proj data into shared memory location.  Defaults to False.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    if exchange_rank > 0:
        exchange_base = 'exchange{:d}'.format(int(exchange_rank))
    else:
        exchange_base = "exchange"
    
    tomo_grp = '/'.join([exchange_base, 'data'])
    flat_grp = '/'.join([exchange_base, 'data_white'])
    dark_grp = '/'.join([exchange_base, 'data_dark'])
    tomo = tio.read_hdf5(fname, tomo_grp, (proj, sino), dtype, shared)
    flat = tio.read_hdf5(fname, flat_grp, (None, sino), dtype)
    dark = tio.read_hdf5(fname, dark_grp, (None, sino), dtype)
    return tomo, flat, dark


@deprecated
def read_aus_microct(fname, ind_tomo, ind_flat, ind_dark):
    """
    Read Australian Synchrotron micro-CT standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind_tomo : list of int
        Indices of the projection files to read.

    ind_flat : list of int
        Indices of the flat field files to read.

    ind_dark : list of int
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
    tomo_name = os.path.join(fname, 'SAMPLE_T_0000.tif')
    flat_name = os.path.join(fname, 'BG__BEFORE_00.tif')
    dark_name = os.path.join(fname, 'DF__BEFORE_00.tif')
    tomo = tio.read_tiff_stack(tomo_name, ind=ind_tomo, digit=4)
    flat = tio.read_tiff_stack(flat_name, ind=ind_flat, digit=2)
    dark = tio.read_tiff_stack(dark_name, ind=ind_dark, digit=2)
    return tomo, flat, dark


@deprecated
def read_esrf_id19(fname, proj=None, sino=None):
    """
    Read ESRF ID-19 standard data format.

    Parameters
    ----------
    fname : str
        Path to edf file.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

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
    tomo_name = os.path.join(fname, 'tomo.edf')
    flat_name = os.path.join(fname, 'flat.edf')
    dark_name = os.path.join(fname, 'dark.edf')
    tomo = tio.read_edf(tomo_name, slc=(proj, sino))
    flat = tio.read_edf(flat_name, slc=(None, sino))
    dark = tio.read_edf(dark_name, slc=(None, sino))
    return tomo, flat, dark


@deprecated
def read_diamond_l12(fname, ind_tomo):
    """
    Read Diamond Light Source L12 (JEEP) standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind_tomo : list of int
        Indices of the projection files to read.

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.
    """
    fname = os.path.abspath(fname)
    tomo_name = os.path.join(fname, 'im_001000.tif')
    flat_name = os.path.join(fname, 'flat_000000.tif')
    ind_flat = list(range(0, 1))
    tomo = tio.read_tiff_stack(tomo_name, ind=ind_tomo, digit=6)
    flat = tio.read_tiff_stack(flat_name, ind=ind_flat, digit=6)
    return tomo, flat


@deprecated
def read_elettra_syrmep(
        fname, ind_tomo, ind_flat, ind_dark, proj=None, sino=None):
    """
    Read Elettra SYRMEP standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind_tomo : list of int
        Indices of the projection files to read.

    ind_flat : list of int
        Indices of the flat field files to read.

    ind_dark : list of int
        Indices of the dark field files to read.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

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
    tomo_name = os.path.join(fname, 'tomo_0001.tif')
    flat_name = os.path.join(fname, 'flat_1.tif')
    dark_name = os.path.join(fname, 'dark_1.tif')
    tomo = tio.read_tiff_stack(
        tomo_name, ind=ind_tomo, digit=4, slc=(sino, proj))
    flat = tio.read_tiff_stack(
        flat_name, ind=ind_flat, digit=1, slc=(sino, None))
    dark = tio.read_tiff_stack(
        dark_name, ind=ind_dark, digit=1, slc=(sino, None))
    return tomo, flat, dark


@deprecated
def read_lnls_imx(folder, proj=None, sino=None):
    """
    Read LNLS IMX standard data format.

    Parameters
    ----------
    folder : str
        Path to sample folder (containing tomo.h5, flat.h5, dark.h5)

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

    Returns
    -------
    ndarray
        3D tomographic data.

    ndarray
        3d flat field data.

    ndarray
        3D dark field data.
    """
    folder = os.path.abspath(folder)
    tomo_name = os.path.join(folder, 'tomo.h5')
    flat_name = os.path.join(folder, 'tomo_flat_before.h5')
    dark_name = os.path.join(folder, 'tomo_dark_before.h5')
    tomo = tio.read_hdf5(tomo_name, 'images', slc=(proj, sino))
    flat = tio.read_hdf5(flat_name, 'flats', slc=(None, sino))
    dark = tio.read_hdf5(dark_name, 'darks', slc=(None, sino))
    return tomo, flat, dark


@deprecated
def read_petraIII_p05(
        fname, ind_tomo, ind_flat, ind_dark, proj=None, sino=None):
    """
    Read Petra-III P05 standard data format.

    Parameters
    ----------
    fname : str
        Path to data folder.

    ind_tomo : list of int
        Indices of the projection files to read.

    ind_flat : list of int
        Indices of the flat field files to read.

    ind_dark : list of int
        Indices of the dark field files to read.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

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
    tomo_name = os.path.join(
        fname, 'scan_0002', 'ccd', 'pco01', 'ccd_0000.tif')
    flat_name = os.path.join(
        fname, 'scan_0001', 'ccd', 'pco01', 'ccd_0000.tif')
    dark_name = os.path.join(
        fname, 'scan_0000', 'ccd', 'pco01', 'ccd_0000.tif')
    tomo = tio.read_tiff_stack(
        tomo_name, ind=ind_tomo, digit=4, slc=(sino, proj))
    flat = tio.read_tiff_stack(
        flat_name, ind=ind_flat, digit=4, slc=(sino, None))
    dark = tio.read_tiff_stack(
        dark_name, ind=ind_dark, digit=4, slc=(sino, None))
    return tomo, flat, dark


@deprecated
def read_sls_tomcat(fname, ind_tomo=None, proj=None, sino=None):
    """
    Read SLS TOMCAT standard data format.

    Parameters
    ----------
    fname : str
        Path to file name without indices and extension.

    ind_tomo : list of int, optional
        Indices of the projection files to read.

    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)

    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)

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
    _fname = fname + '0001.tif'
    log_file = fname + '.log'

    # Read metadata from SLS log file.
    contents = open(log_file, 'r')
    for line in contents:
        ls = line.split()
        if len(ls) > 1:
            if ls[0] == 'Number' and ls[2] == 'projections':
                nproj = int(ls[4])
            elif ls[0] == 'Number' and ls[2] == 'flats':
                nflat = int(ls[4])
            elif ls[0] == 'Number' and ls[2] == 'darks':
                ndark = int(ls[4])
    contents.close()

    dark_start = 1
    dark_end = ndark + 1
    flat_start = dark_end
    flat_end = flat_start + nflat
    proj_start = flat_end
    proj_end = proj_start + nproj

    if ind_tomo is None:
        ind_tomo = list(range(proj_start, proj_end))
    ind_flat = list(range(flat_start, flat_end))
    ind_dark = list(range(dark_start, dark_end))
    tomo = tio.read_tiff_stack(_fname, ind=ind_tomo, digit=4, slc=(sino, proj))
    flat = tio.read_tiff_stack(_fname, ind=ind_flat, digit=4, slc=(sino, None))
    dark = tio.read_tiff_stack(_fname, ind=ind_dark, digit=4, slc=(sino, None))

    return tomo, flat, dark
