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
Module for simulation of x-rays.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tomopy.util.dtype as dtype
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['propagate_tie',
           'source']


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


def source(fwhm, nx, ny, center=None):
    """
    Simulate x-ray beam as a square Gaussian kernel.

    Parameters
    ----------
    fwhm : float
        Effective radius of the source.
    nx, ny : int
        Grid size along x and y axes.
    center : array_like, 
        x and y coordinates of the center of the gaussian function.

    Returns
    -------
    ndarray
        2D source intensity distribution.
    """
    if center is None:
        x0, y0 = nx // 2, ny // 2
    else:
        x0, y0 = np.array(center)
    x, y = np.mgrid[0:nx, 0:ny]
    g = np.exp(-4*np.log(2) * ((x-x0+0.5)**2 + (y-y0+0.5)**2) / fwhm**2)
    return g / g.sum()
