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
Module for phase retrieval.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from tomopy.util import *
import tomopy.misc.mproc as mp
import logging

logger = logging.getLogger(__name__)

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['retrieve_phase']


BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def retrieve_phase(
        tomo, psize=1e-4, dist=50, energy=20,
        alpha=1e-3, pad=True, ncore=None, nchunk=None):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    psize : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter.
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Approximated 3D tomographic phase data.
    """
    # Compute the filter.
    H, xshift, yshift, prj = _paganin_filter(
        tomo, psize, dist, energy, alpha, pad)

    arr = mp.distribute_jobs(
        tomo,
        func=_retrieve_phase,
        args=(H, xshift, yshift, prj, pad),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _retrieve_phase(H, xshift, yshift, prj, pad, istart, iend):
    tomo = mp.SHARED_ARRAY
    dx, dy, dz = tomo.shape
    for m in range(istart, iend):
        proj = tomo[m, :, :]
        if pad:
            prj[xshift:dy + xshift, yshift:dz + yshift] = proj
            fproj = np.fft.fft2(prj)
            filtproj = np.multiply(H, fproj)
            tmp = np.real(np.fft.ifft2(filtproj)) / np.max(H)
            proj = tmp[xshift:dy + xshift, yshift:dz + yshift]
        elif not pad:
            fproj = np.fft.fft2(proj)
            filtproj = np.multiply(H, fproj)
            proj = np.real(np.fft.ifft2(filtproj)) / np.max(H)
        tomo[m, :, :] = proj


def _paganin_filter(tomo, psize, dist, energy, alpha, pad):
    """
    Calculate Paganin-type 2D filter to be used for phase retrieval.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    psize : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    alpha : float
        Regularization parameter.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    ndarray
        2D Paganin filter.
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    ndarray
        Padded 2D projection image.
    """
    dx, dy, dz = tomo.shape
    wavelen = 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy

    if pad:
        # Find pad values.
        val = np.mean((tomo[:, :, 0] + tomo[:, :, dz - 1]) / 2)

        # Fourier pad in powers of 2.
        padpix = np.ceil(PI * wavelen * dist / psize ** 2)

        nx = pow(2, np.ceil(np.log2(dy + padpix)))
        ny = pow(2, np.ceil(np.log2(dz + padpix)))
        xshift = int((nx - dy) / 2.)
        yshift = int((ny - dz) / 2.)

        # Template pad image.
        prj = val * np.ones((nx, ny), dtype='float32')

    elif not pad:
        nx, ny = dy, dz
        xshift, yshift, prj = None, None, None
        prj = np.ones((dy, dz), dtype='float32')

    # Sampling in reciprocal space.
    indx = (1 / ((nx - 1) * psize)) * np.arange(-(nx - 1) * 0.5, nx * 0.5)
    indy = (1 / ((ny - 1) * psize)) * np.arange(-(ny - 1) * 0.5, ny * 0.5)
    du, dv = np.meshgrid(indy, indx)
    w2 = np.square(du) + np.square(dv)

    # Filter in Fourier space.
    H = 1 / (wavelen * dist * w2 / (4 * PI) + alpha)
    H = np.fft.fftshift(H)
    return H, xshift, yshift, prj
