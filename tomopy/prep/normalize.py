#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2015-2019. UChicago Argonne, LLC. This software was produced  #
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
Module for data normalization.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tomopy.util.mproc as mproc
import tomopy.util.extern as extern
import tomopy.util.dtype as dtype
import logging
import numexpr as ne

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Luis Barroso-Luque"
__credits__ = "Mark Rivers"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['minus_log',
           'normalize',
           'normalize_bg',
           'normalize_roi',
           'normalize_nf']


def minus_log(arr, ncore=None, out=None):
    """
    Computation of the minus log of a given array.

    Parameters
    ----------
    arr : ndarray
        3D stack of projections.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result. If same as arr, process
        will be done in-place.

    Returns
    -------
    ndarray
        Minus-log of the input data.
    """
    arr = dtype.as_float32(arr)

    with mproc.set_numexpr_threads(ncore):
        out = ne.evaluate('-log(arr)', out=out)
    return out


def normalize(arr, flat, dark, cutoff=None, ncore=None, out=None):
    """
    Normalize raw projection data using the flat and dark field projections.

    Parameters
    ----------
    arr : ndarray
        3D stack of projections.
    flat : ndarray
        3D flat field data.
    dark : ndarray
        3D dark field data.
    cutoff : float, optional
        Permitted maximum vaue for the normalized data.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result. If same as arr,
        process will be done in-place.

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    arr = dtype.as_float32(arr)
    l = np.float32(1e-6)
    flat = np.mean(flat, axis=0, dtype=np.float32)
    dark = np.mean(dark, axis=0, dtype=np.float32)

    with mproc.set_numexpr_threads(ncore):
        denom = ne.evaluate('flat-dark')
        ne.evaluate('where(denom<l,l,denom)', out=denom)
        out = ne.evaluate('arr-dark', out=out)
        ne.evaluate('out/denom', out=out, truediv=True)
        if cutoff is not None:
            cutoff = np.float32(cutoff)
            ne.evaluate('where(out>cutoff,cutoff,out)', out=out)
    return out


# TODO: replace roi indexes with slc object
def normalize_roi(arr, roi=[0, 0, 10, 10], ncore=None):
    """
    Normalize raw projection data using an average of a selected window
    on projection images.

    Parameters
    ----------
    arr : ndarray
        3D tomographic data.
    roi: list of int, optional
        [top-left, top-right, bottom-left, bottom-right] pixel coordinates.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    arr = dtype.as_float32(arr)

    arr = mproc.distribute_jobs(
        arr,
        func=_normalize_roi,
        args=(roi, ),
        axis=0,
        ncore=ncore,
        nchunk=0)
    return arr


def _normalize_roi(proj, roi):
    bg = proj[roi[0]:roi[2], roi[1]:roi[3]].mean()
    # avoid divide by zero
    if bg != 0:
        np.true_divide(proj, bg, proj)


def normalize_bg(tomo, air=1, ncore=None, nchunk=None):
    """
    Normalize 3D tomgraphy data based on background intensity.

    Weight sinogram such that the left and right image boundaries
    (i.e., typically the air region around the object) are set to one
    and all intermediate values are scaled linearly.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    air : int, optional
        Number of pixels at each boundary to calculate the scaling factor.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    tomo = dtype.as_float32(tomo)
    air = dtype.as_int32(air)

    arr = mproc.distribute_jobs(
        tomo,
        func=extern.c_normalize_bg,
        args=(air,),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def normalize_nf(tomo, flats, dark, flat_loc,
                 cutoff=None, ncore=None, out=None):
    """
    Normalize raw 3D projection data with flats taken more than once during
    tomography. Normalization for each projection is done with the mean of the
    nearest set of flat fields (nearest flat fields).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flats : ndarray
        3D flat field data.
    dark : ndarray
        3D dark field data.
    flat_loc : list of int
        Indices of flat field data within tomography
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result. If same as arr, process
        will be done in-place.

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """

    tomo = dtype.as_float32(tomo)
    flats = dtype.as_float32(flats)
    dark = dtype.as_float32(dark)
    l = np.float32(1e-6)
    if cutoff is not None:
        cutoff = np.float32(cutoff)
    if out is None:
        out = np.empty_like(tomo)

    dark = np.median(dark, axis=0)
    denom = np.empty_like(dark)

    num_flats = len(flat_loc)
    total_flats = flats.shape[0]
    total_tomo = tomo.shape[0]

    num_per_flat = total_flats//num_flats
    tend = 0

    for m, loc in enumerate(flat_loc):
        fstart = m*num_per_flat
        fend = (m + 1)*num_per_flat
        flat = np.median(flats[fstart:fend], axis=0)

        # Normalization can be parallelized much more efficiently outside this
        # for loop accounting for the nested parallelism arising from
        # chunking the total normalization and each chunked normalization
        tstart = 0 if m == 0 else tend
        tend = total_tomo if m >= num_flats-1 \
            else int(np.round((flat_loc[m+1]-loc)/2)) + loc
        tomo_l = tomo[tstart:tend]
        out_l = out[tstart:tend]
        with mproc.set_numexpr_threads(ncore):
            ne.evaluate('flat-dark', out=denom)
            ne.evaluate('where(denom<l,l,denom)', out=denom)
            ne.evaluate('(tomo_l-dark)/denom', out=out_l, truediv=True)
            if cutoff is not None:
                ne.evaluate('where(out_l>cutoff,cutoff,out_l)', out=out_l)

    return out
