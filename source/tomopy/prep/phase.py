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
Module for phase retrieval.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from tomopy.util.misc import (fft2, ifft2)


import tomopy.util.mproc as mproc
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__credits__ = "Mark Rivers, Xianghui Xiao"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['retrieve_phase']


BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def retrieve_phase(
        tomo, pixel_size=1e-4, dist=50, energy=20,
        alpha=1e-3, pad=True, ncore=None, nchunk=None):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float, optional
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
    # New dimensions and pad value after padding.
    py, pz, val = _calc_pad(tomo, pixel_size, dist, energy, pad)

    # Compute the reciprocal grid.
    dx, dy, dz = tomo.shape
    w2 = _reciprocal_grid(pixel_size, dy + 2 * py, dz + 2 * pz)

    # Filter in Fourier space.
    phase_filter = np.fft.fftshift(
        _paganin_filter_factor(energy, dist, alpha, w2))

    prj = np.full((dy + 2 * py, dz + 2 * pz), val, dtype='float32')
    arr = mproc.distribute_jobs(
        tomo,
        func=_retrieve_phase,
        args=(phase_filter, py, pz, prj, pad),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _retrieve_phase(tomo, phase_filter, px, py, prj, pad):
    dx, dy, dz = tomo.shape
    num_jobs = tomo.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()
    for m in range(num_jobs):
        prj[px:dy + px, py:dz + py] = tomo[m]
        prj[:px] = prj[px]
        prj[-px:] = prj[-px-1]
        prj[:, :py] = prj[:, py][:, np.newaxis]
        prj[:, -py:] = prj[:, -py-1][:, np.newaxis]
        fproj = fft2(prj, extra_info=num_jobs)
        fproj *= normalized_phase_filter
        proj = np.real(ifft2(fproj, extra_info=num_jobs, overwrite_input=True))
        if pad:
            proj = proj[px:dy + px, py:dz + py]
        tomo[m] = proj



def _calc_pad(tomo, pixel_size, dist, energy, pad):
    """
    Calculate new dimensions and pad value after padding.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    float
        Pad value.
    """
    dx, dy, dz = tomo.shape
    wavelength = _wavelength(energy)
    py, pz, val = 0, 0, 0
    if pad:
        val = _calc_pad_val(tomo)
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)
    return py, pz, val


def _paganin_filter_factor(energy, dist, alpha, w2):
    return 1 / (_wavelength(energy) * dist * w2 / (4 * PI) + alpha)


def _calc_pad_width(dim, pixel_size, wavelength, dist):
    pad_pix = np.ceil(PI * wavelength * dist / pixel_size ** 2)
    return int((pow(2, np.ceil(np.log2(dim + pad_pix))) - dim) * 0.5)


def _calc_pad_val(tomo):
    return np.mean((tomo[..., 0] + tomo[..., -1]) * 0.5)


def _reciprocal_grid(pixel_size, nx, ny):
    """
    Calculate reciprocal grid.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, nx)
    indy = _reciprocal_coord(pixel_size, ny)
    np.square(indx, out=indx)
    np.square(indy, out=indy)
    return np.add.outer(indx, indy)


def _reciprocal_coord(pixel_size, num_grid):
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype = np.float32)
    rc *= 0.5 / (n * pixel_size)
    return  rc
