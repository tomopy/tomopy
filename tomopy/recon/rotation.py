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
from scipy import ndimage
from scipy.optimize import minimize
from tomopy.io.writer import write_tiff
from tomopy.misc.mask import circ_mask
from tomopy.recon.algorithm import recon
import tomopy.util.dtype as dtype
import os.path
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['find_center',
           'find_center_vo',
           'write_center']


PI = 3.14159265359


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
    tomo = dtype.as_float32(tomo)
    theta = dtype.as_float32(theta)

    if ind is None:
        ind = tomo.shape[1] // 2
    if init is None:
        init = tomo.shape[2] // 2

    hmin, hmax = _adjust_hist_limits(
        tomo[:, ind:ind + 1, :], theta, ind, mask, emission)

    # Magic is ready to happen...
    res = minimize(
        _find_center_cost, init,
        args=(tomo, theta, ind, hmin, hmax, mask, ratio, emission),
        method='Nelder-Mead',
        tol=tol)
    return res.x


def _adjust_hist_limits(tomo, theta, ind, mask, emission):
    # Make an initial reconstruction to adjust histogram limits.
    rec = recon(tomo, theta, emission=emission, algorithm='gridrec')

    # Apply circular mask.
    if mask is True:
        rec = circ_mask(rec, axis=0)

    # Adjust histogram boundaries according to reconstruction.
    return _adjust_hist_min(rec.min()), _adjust_hist_max(rec.max())


def _adjust_hist_min(val):
    if val < 0:
        val = 2 * val
    elif val >= 0:
        val = 0.5 * val
    return val


def _adjust_hist_max(val):
    if val < 0:
        val = 0.5 * val
    elif val >= 0:
        val = 2 * val
    return val


def _find_center_cost(
        center, tomo, theta, ind, hmin, hmax, mask, ratio, emission):
    """
    Cost function used for the ``find_center`` routine.
    """
    logger.info('trying center: %s', center)
    center = np.array(center, dtype='float32')
    rec = recon(
        tomo[:, ind:ind + 1, :], theta, center,
        emission=emission, algorithm='gridrec')

    if mask is True:
        rec = circ_mask(rec, axis=0)

    hist, e = np.histogram(rec, bins=64, range=[hmin, hmax])
    hist = hist.astype('float32') / rec.size + 1e-12
    return -np.dot(hist, np.log2(hist))


def find_center_vo(tomo, ind=None, smin=-50, smax=50, srad=3,
                   tol=0.5, ratio=2., drop=20):
    """
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Reference to the horizontal center of the sinogram.
    srad : float, optional
        Initial guess for the center.
    tol : scalar, optional
        Desired sub-pixel accuracy.
    ratio : float, optional
        The ratio between the size of object and FOV of the camera.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.

    Returns
    -------
    float
        Rotation axis location.

    Warning
    -------
    Not tested yet.
    """
    tomo = dtype.as_float32(tomo)

    if ind is None:
        ind = tomo.shape[1] // 2
    if init is None:
        init = tomo.shape[2] // 2

    # Reduce noise by smooth filtering.
    tomo = ndimage.filters.gaussian_filter(tomo, sigma=(3, 1))

    # Coarse search for finiding the roataion center.
    init_cen = _search_coarse(tomo, smin, smax, ratio, drop)

    # Fine search for finiding the roataion center.
    return _search_fine(tomo, srad, step, init_cen, ratio, drop)


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finiding the roataion center.
    """
    (Nrow, Ncol) = sino.shape
    centerfliplr = (Ncol - 1.0) / 2.0

    # Copy the sinogram and flip left right, the purpose is to make a full
    # [0;2Pi] sinogram
    _copy_sino = np.fliplr(sino[1:])

    # This image is used for compensating the shift of sinogram 2
    temp_img = np.zeros((Nrow - 1, Ncol), dtype='float32')
    temp_img[:] = sino[-1]

    # Start coarse search in which the shift step is 1
    listshift = np.arange(smin, smax + 1)
    listmetric = np.zeros(len(listshift), dtype='float32')
    mask = _create_mask(2 * Nrow - 1, Ncol, 0.5 * ratio * Ncol, drop)
    for i in listshift:
        _sino = np.roll(_copy_sino, i, axis=1)
        if i >= 0:
            _sino[:, 0:i] = temp_img[:, 0:i]
        else:
            _sino[:, i:] = temp_img[:, i:]
        listmetric[
            i - smin] = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(np.vstack((sino, _sino))))) * mask)
    minpos = np.argmin(listmetric)
    return centerfliplr + listshift[minpos] / 2.0


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finiding the roataion center.
    """
    Nrow, Ncol = sino.shape
    centerfliplr = (Ncol + 1.0) / 2.0 - 1.0

    # Use to shift the sinogram 2 to the raw CoR.
    shiftsino = np.int16(2 * (init_cen - centerfliplr))
    _copy_sino = np.roll(np.fliplr(sino[1:]), shiftsino, axis=1)
    lefttake = 0
    righttake = Ncol - 1
    if init_cen <= centerfliplr:
        lefttake = np.ceil(srad + 1)
        righttake = np.floor(2 * init_cen - srad - 1)
    else:
        lefttake = np.ceil(
            init_cen - (Ncol - 1 - init_cen) + srad + 1)
        righttake = np.floor(Ncol - 1 - srad - 1)
    Ncol1 = righttake - lefttake + 1
    mask = _create_mask(2 * Nrow - 1, Ncol1, 0.5 * ratio * Ncol, drop)
    numshift = np.int16((2 * srad + 1.0) / step)
    listshift = np.linspace(-srad, srad, num=numshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    num1 = 0
    for i in listshift:
        _sino = ndimage.ndi.interpolation.shift(
            _copy_sino, (0, i), prefilter=False)
        sinojoin = np.vstack((sino, _sino))
        listmetric[num1] = np.sum(
            np.abs(np.fft.fftshift(np.fft.fft2(sinojoin[:, lefttake:righttake + 1]))) * mask)
        num1 = num1 + 1
    minpos = np.argmin(listmetric)
    return init_cen + listshift[minpos] / 2.0


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * PI)
    centerrow = np.ceil(nrow / 2) - 1
    centercol = np.ceil(ncol / 2) - 1
    mask = np.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        num1 = np.round(((i - centerrow) * dv / radius) / du)
        (p1, p2) = np.clip(np.sort((-num1 + centercol, num1 + centercol)), 0, ncol - 1)
        mask[i, p1:p2 + 1] = np.ones(p2 - p1 + 1, dtype='float32')
    if drop < centerrow:
        mask[centerrow - drop:centerrow + drop + 1,
             :] = np.zeros((2 * drop + 1, ncol), dtype='float32')
    return mask


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
    tomo = dtype.as_float32(tomo)
    theta = dtype.as_float32(theta)

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
    rec = recon(
        stack, theta, center=center, emission=emission, algorithm='gridrec')

    # Apply circular mask.
    if mask is True:
        rec = circ_mask(rec, axis=0)

    # Save images to a temporary folder.
    for m in range(len(center)):
        if m % 2 == 0:  # 2 slices same bec of gridrec.
            fname = os.path.join(
                dpath, str('{:.2f}'.format(center[m]) + '.tiff'))
            write_tiff(rec[m:m + 1], fname=fname, overwrite=True)
