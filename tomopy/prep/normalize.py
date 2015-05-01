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
Module for data normalization.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tomopy.misc.mproc as mp
import tomopy.extern as ext
from tomopy.util import *
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['normalize',
           'normalize_bg']


def normalize(tomo, flat, dark, cutoff=None, ncore=None, nchunk=None):
    """
    Normalize raw projection data using the flat and dark field projections.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flat : ndarray
        3D flat field data.
    dark : ndarray
        3D dark field data.
    cutoff : float, optional
        Permitted maximum vaue for the normalized data.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    tomo = as_float32(tomo)
    flat = as_float32(flat)
    dark = as_float32(dark)

    flat = flat.mean(axis=0)
    dark = dark.mean(axis=0)

    arr = mp.distribute_jobs(
        tomo,
        func=_normalize,
        args=(flat, dark, cutoff),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _normalize(flat, dark, cutoff, istart, iend):
    tomo = mp.SHARED_ARRAY

    # Avoid zero division in normalization
    denom = flat - dark
    denom[denom == 0] = 1e-6

    for m in range(istart, iend):
        proj = np.true_divide(tomo[m, :, :] - dark, denom)
        if cutoff is not None:
            proj[proj > cutoff] = cutoff
        tomo[m, :, :] = proj


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
    tomo = as_float32(tomo)
    air = as_int32(air)
    dx, dy, dz = tomo.shape

    arr = mp.distribute_jobs(
        tomo,
        func=ext.c_normalize_bg,
        args=(dx, dy, dz, air),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr
