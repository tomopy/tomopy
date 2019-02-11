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
Module for simulation of x-rays.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['calc_intensity',
           'propagate_tie',
           'probe_gauss']


def propagate_tie(mu, delta, pixel_size, dist):
    """
    Propagate emitting x-ray wave based on Transport of Intensity.

    Parameters
    ----------
    mu : ndarray, optional
        3D tomographic data for attenuation.
    delta : ndarray
        3D tomographic data for refractive index.
    pixel_size : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.

    Returns
    -------
    ndarray
        3D propagated tomographic intensity.
    """
    i1 = np.exp(-mu)
    i2 = np.zeros(delta.shape)
    for m in range(delta.shape[0]):
        dx, dy = np.gradient(delta[m], pixel_size)
        d2x, _ = np.gradient(i1[m] * dx, pixel_size)
        _, d2y = np.gradient(i1[m] * dy, pixel_size)
        i2[m] = i1[m] + dist * (d2x + d2y)
    return i2


def probe_gauss(nx, ny, fwhm=None, center=None, max_int=1):
    """
    Simulate incident x-ray beam (probe) as a square Gaussian kernel.

    Parameters
    ----------
    nx, ny : int
        Grid size along x and y axes.
    fwhm : float, optional
        Effective radius of the source.
    center : array_like, optional
        x and y coordinates of the center of the gaussian function.
    max_int : int
        Maximum x-ray intensity.

    Returns
    -------
    ndarray
        2D source intensity distribution.
    """
    if fwhm is None:
        fwhm = max(nx, ny) // 2
    if center is None:
        x0, y0 = nx // 2, ny // 2
    else:
        x0, y0 = np.array(center)
    x, y = np.mgrid[0:nx, 0:ny]
    return max_int * np.exp(-4 * np.log(2) * (
        (x - x0 + 0.5) ** 2 +
        (y - y0 + 0.5) ** 2) / fwhm ** 2)


def _rect_scan_coords(probe_grid, proj_grid, shift_x, shift_y):
    """
    Calculate upper-left scan coordinates of a rectangular kernel given
    a projection image.

    Parameters
    ----------
    proj_grid : (int, int)
        Grid size of the projection image along x and y axes.
    shift_x, shift_y : int
        Relative shift distace of the source along x and y axes.

    Returns
    -------
    array
        x coordinates of upper-left scan coordinates
    array
        y coordinates of upper-left scan coordinates
    """
    x = np.arange(0, proj_grid[0], shift_x)
    y = np.arange(0, proj_grid[1], shift_y)
    while x.size * shift_x > proj_grid[0] - probe_grid[0] + shift_x:
        x = x[:-1]
    while y.size * shift_y > proj_grid[1] - probe_grid[1] + shift_y:
        y = y[:-1]
    return x, y


def _rect_scan_probe(probe, proj, shift_x=None, shift_y=None):
    """
    Calculate individual raster scanned images for a given rectangular
    x-ray probe and an object plane intensity.

    Parameters
    ----------
    probe : ndarray
        Rectangular x-ray source kernel.
    proj : ndarray
        Object plane intensity image.
    shift_x, shift_y : int, optional
        Shift amount of probe along x and y axes.

    Returns
    -------
    ndarray
        Individual raster scanned images as 3D array.
    """
    sx, sy = probe.shape
    px, py = proj.shape

    # Assume half overlap.
    if shift_x is None:
        shift_x = sx // 2
    if shift_y is None:
        shift_y = sy // 2

    # Calculate upper-left scan coordinates along x and y axes.
    x, y = _rect_scan_coords(probe.shape, proj.shape, shift_x, shift_y)

    # Convert to image stack.
    arr = [probe * proj[i:i + sx, j:j + sy] for i in x for j in y]
    return np.array(arr)


def calc_intensity(probe, proj, shift_x=None, shift_y=None, mode='near'):
    """
    Calculate far field intensity.

    Parameters
    ----------
    probe : ndarray
        Rectangular x-ray source kernel.
    proj : ndarray
        Object plane intensity image.
    shift_x, shift_y : int, optional
        Shift amount of probe along x and y axes.
    mode : str, optional
        Specify the regime. 'near' or 'far'

    Returns
    -------
    ndarray
        Individual raster scanned far field images as 3D array.
    """
    psi = _rect_scan_probe(probe, proj, shift_x, shift_y)
    if mode == 'near':
        intensity = abs(psi) ** 2
    elif mode == 'far':
        intensity = abs(np.fft.fftshift(np.fft.fft2(psi))) ** 2
    return intensity
