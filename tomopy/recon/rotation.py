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
Module for functions related to finding axis of rotation.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import minimize
from tomopy.io.writer import Writer
from tomopy.misc.mask import circ_mask
from tomopy.recon.algorithm import Recon
from tomopy.util import *
import os.path
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['find_center',
           'write_center']


def find_center(
        tomo, theta, ind=None, emission=True, init=None,
        tol=0.5, mask=True, ratio=1.):
    """
    Find rotation axis location.

    The function exploits systematic artifacts in reconstructed images
    due to shifts in the rotation center. It uses image entropy
    as the error metric and ''Nelder-Mead'' routine (of the scipy
    optimization module) as the optimizer :cite:`Donath:06`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    init : float
        Initial guess for the center.
    tol : scalar
        Desired sub-pixel accuracy.
    mask : bool, optional
        If ``True``, apply a circular mask to the reconstructed image to
        limit the analysis into a circular region.
    ratio : float, optional
        The ratio of the radius of the circular mask to the edge of the
        reconstructed image.

    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = as_float32(tomo)
    theta = as_float32(theta)

    dx, dy, dz = tomo.shape
    if ind is None:
        ind = dy / 2
    if init is None:
        init = dz / 2

    # Make an initial reconstruction to adjust histogram limits.
    rec = gridrec(tomo[:, ind:ind + 1, :], theta, emission=emission)

    # Apply circular mask.
    if mask is True:
        rec = circ_mask(rec, axis=0)

    # Adjust histogram boundaries according to reconstruction.
    hmin = np.min(rec)
    if hmin < 0:
        hmin = 2 * hmin
    elif hmin >= 0:
        hmin = 0.5 * hmin
    hmax = np.max(rec)
    if hmax < 0:
        hmax = 0.5 * hmax
    elif hmax >= 0:
        hmax = 2 * hmax

    # Magic is ready to happen...
    res = minimize(
        _find_center_cost, init,
        args=(tomo, rec, theta, ind, hmin, hmax, mask, ratio, emission),
        method='Nelder-Mead',
        tol=tol)


def _find_center_cost(
        center, tomo, rec, theta, ind, hmin, hmax, mask, ratio, emission):
    """
    Cost function used for the ``find_center`` routine.
    """
    print('Trying center: ', center)
    center = np.array(center, dtype='float32')
    rec = gridrec(tomo[:, ind:ind + 1, :], theta, center, emission)

    if mask is True:
        rec = circ_mask(rec, axis=0)

    hist, e = np.histogram(rec, bins=64, range=[hmin, hmax])
    hist = hist.astype('float32') / rec.size + 1e-12
    return -np.dot(hist, np.log2(hist))


def write_center(
        tomo, theta, dpath='tmp/center', cen_range=None, ind=None,
        emission=True, mask=False, ratio=1.):
    """
    Save images reconstructed with a range of rotation centers.

    Helps finding the rotation center manually by visual inspection of
    images reconstructed with a set of different centers.The output
    images are put into a specified folder and are named by the
    center position corresponding to the image.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    dpath : str, optional
        Folder name to save output images.
    cen_range : list, optional
        [start, end, step] Range of center values.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    mask : bool, optional
        If ``True``, apply a circular mask to the reconstructed image to
        limit the analysis into a circular region.
    ratio : float, optional
        The ratio of the radius of the circular mask to the edge of the
        reconstructed image.
    """
    tomo = as_float32(tomo)
    theta = as_float32(theta)

    dx, dy, dz = tomo.shape
    if ind is None:
        ind = dy / 2
    if cen_range is None:
        center = np.arange(dz / 2 - 5, dz / 2 + 5, 0.5)
    else:
        center = np.arange(cen_range[0], cen_range[1], cen_range[2] / 2.)

    stack = np.zeros((dx, len(center), dz))
    for m in range(center.size):
        stack[:, m, :] = tomo[:, ind, :]

    # Reconstruct the same slice with a range of centers.
    rec = Recon(stack, theta, center=center, emission=emission).gridrec()

    # Apply circular mask.
    if mask is True:
        rec = circ_mask(rec, axis=0)

    # Save images to a temporary folder.
    for m in range(len(center)):
        if m % 2 == 0:  # 2 slices same bec of gridrec.
            fname = os.path.join(
                dpath, str('{:.2f}'.format(center[m]) + 'tiff'))
            Writer(rec[m:m + 1], fname=fname, overwrite=True).tiff(stack=False)
