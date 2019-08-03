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
Module for functions related to finding axis of rotation.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import ndimage
from tomopy.util.misc import fft2, write_tiff
from scipy.optimize import minimize
from skimage.feature import register_translation
from tomopy.misc.corr import circ_mask
from tomopy.misc.morph import downsample
from tomopy.recon.algorithm import recon
import tomopy.util.dtype as dtype
import os.path
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Luis Barroso-Luque, Nghia Vo"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['find_center',
           'find_center_vo',
           'find_center_pc',
           'write_center',
           'mask_empty_slice',
           ]


def find_center(
        tomo, theta, ind=None, init=None,
        tol=0.5, mask=True, ratio=1., sinogram_order=False):
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
    sinogram_order: bool, optional
        Determins whether data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).

    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = dtype.as_float32(tomo)
    theta = dtype.as_float32(theta)

    if sinogram_order:
        dy, dt, dx = tomo.shape
    else:
        dt, dy, dx = tomo.shape

    if ind is None:
        ind = dy // 2
    if init is None:
        init = dx // 2

    # extract slice we are using to find center
    if sinogram_order:
        tomo_ind = tomo[ind:ind + 1]
    else:
        tomo_ind = tomo[:, ind:ind + 1, :]

    hmin, hmax = _adjust_hist_limits(
        tomo_ind, theta, mask, sinogram_order)

    # Magic is ready to happen...
    res = minimize(
        _find_center_cost, init,
        args=(tomo_ind, theta, hmin, hmax, mask, ratio, sinogram_order),
        method='Nelder-Mead',
        tol=tol)
    return res.x


def _adjust_hist_limits(tomo_ind, theta, mask, sinogram_order):
    # Make an initial reconstruction to adjust histogram limits.
    rec = recon(tomo_ind,
                theta,
                sinogram_order=sinogram_order,
                algorithm='gridrec')

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
        center, tomo_ind, theta, hmin, hmax, mask, ratio,
        sinogram_order=False):
    """
    Cost function used for the ``find_center`` routine.
    """
    logger.info('Trying rotation center: %s', center)
    center = np.array(center, dtype='float32')
    rec = recon(
        tomo_ind, theta, center,
        sinogram_order=sinogram_order,
        algorithm='gridrec')

    if mask is True:
        rec = circ_mask(rec, axis=0)

    hist, e = np.histogram(rec, bins=64, range=[hmin, hmax])
    hist = hist.astype('float32') / rec.size + 1e-12
    val = -np.dot(hist, np.log2(hist))
    logger.info("Function value = %f" % val)
    return val


def find_center_vo(tomo, ind=None, smin=-50, smax=50, srad=6, step=0.25,
                   ratio=0.5, drop=20):
    """
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Coarse search radius. Reference to the horizontal center of the sinogram.
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.

    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = dtype.as_float32(tomo)
    (depth, height, width) = tomo.shape
    if ind is None:
        ind = height // 2
        if height > 10:
            # Averaging sinograms to improve SNR
            _tomo = np.mean(tomo[:, ind - 5:ind + 5, :], axis=1)
        else:
            _tomo = tomo[:, ind, :]
    else:
        _tomo = tomo[:, ind, :]

    # Denoising
    # There's a critical reason to use different window sizes
    # between coarse and fine search.
    _tomo_cs = ndimage.filters.gaussian_filter(_tomo, (3, 1))
    _tomo_fs = ndimage.filters.gaussian_filter(_tomo, (2, 2))

    # Coarse and fine searches for finding the rotation center.
    if _tomo.shape[0] * _tomo.shape[1] > 4e6:  # If data is large (>2kx2k)
        _tomo_coarse = downsample(
            np.expand_dims(_tomo_cs, 1), level=2)[:, 0, :]
        init_cen = _search_coarse(
            _tomo_coarse, smin / 4.0, smax / 4.0, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step,
                                init_cen * 4, ratio, drop)
    else:
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen, ratio, drop)

    logger.debug('Rotation center search finished: %i', fine_cen)
    return fine_cen


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    smin = np.int16(np.clip(smin + cen_fliplr, 0, ncol - 1) - cen_fliplr)
    smax = np.int16(np.clip(smax + cen_fliplr, 0, ncol - 1) - cen_fliplr)
    # Flip left-right the [0:Pi ] sinogram to make a full [0;2Pi] sinogram
    flip_sino = np.fliplr(sino)
    # Below image is used for compensating the shift of the [Pi;2Pi] sinogram
    # It helps to avoid local minima.
    comp_sino = np.flipud(sino)
    list_shift = np.arange(smin, smax + 1)
    list_metric = np.zeros(len(list_shift), dtype='float32')
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    sino_sino = np.vstack((sino, flip_sino))
    abs_fft2_sino = np.empty_like(sino_sino)
    for i in list_shift:
        _sino = sino_sino[nrow:]
        _sino[...] = np.roll(flip_sino, i, axis=1)
        if i >= 0:
            _sino[:, 0:i] = comp_sino[:, 0:i]
        else:
            _sino[:, i:] = comp_sino[:, i:]
        fft2sino = np.fft.fftshift(fft2(sino_sino))
        np.abs(fft2sino, out=abs_fft2_sino)
        abs_fft2_sino *= mask
        list_metric[i - smin] = abs_fft2_sino.mean()
    minpos = np.argmin(list_metric)
    if minpos == 0:
        logger.debug('WARNING!!!Global minimum is out of searching range')
        logger.debug('Please extend smin: %i', smin)
    if minpos == len(list_metric) - 1:
        logger.debug('WARNING!!!Global minimum is out of searching range')
        logger.debug('Please extend smax: %i', smax)
    init_cen = cen_fliplr + list_shift[minpos] / 2.0
    return init_cen


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    srad = np.clip(np.abs(srad), 1.0, ncol / 4.0)
    step = np.clip(np.abs(step), 0.1, srad)
    init_cen = np.clip(init_cen, srad, ncol - srad - 1)
    list_cor = init_cen + np.arange(-srad, srad + step, step)
    flip_sino = np.fliplr(sino)
    comp_sino = np.flipud(sino)  # Used to avoid local minima
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    for i, cor in enumerate(list_cor):
        shift = 2.0 * (cor - cen_fliplr)
        sino_shift = ndimage.interpolation.shift(
            flip_sino, (0, shift), order=3, prefilter=True)
        if shift >= 0:
            shift_int = np.int16(np.ceil(shift))
            sino_shift[:, :shift_int] = comp_sino[:, :shift_int]
        else:
            shift_int = np.int16(np.floor(shift))
            sino_shift[:, shift_int:] = comp_sino[:, shift_int:]
        sinojoin = np.vstack((sino, sino_shift))
        list_metric[i] = np.mean(np.abs(
            np.fft.fftshift(fft2(sinojoin))) * mask)
    cor = list_cor[np.argmin(list_metric)]
    return cor


def _create_mask(nrow, ncol, radius, drop):
    """
    Make a binary mask to select coefficients outside the double-wedge region.
    Eq.(3) in https://doi.org/10.1364/OE.22.019078

    Parameters
    ----------
    nrow : int
        Image height.
    ncol : int
        Image width.
    radius: int
        Radius of an object, in pixel unit.
    drop : int
        Drop lines around vertical center of the mask.

    Returns
    -------
        2D binary mask.
    """
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = np.int16(np.ceil(nrow / 2.0) - 1)
    cen_col = np.int16(np.ceil(ncol / 2.0) - 1)
    drop = min(drop, np.int16(np.ceil(0.1 * nrow)))
    mask = np.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        pos = np.int16(np.round(((i - cen_row) * dv / radius) / du))
        (pos1, pos2) = np.clip(np.sort(
            (-pos + cen_col, pos + cen_col)), 0, ncol - 1)
        mask[i, pos1:pos2 + 1] = 1.0
    mask[cen_row - drop:cen_row + drop + 1, :] = 0.0
    mask[:, cen_col - 1:cen_col + 2] = 0.0
    return mask


def find_center_pc(proj1, proj2, tol=0.5, rotc_guess=None):
    """
    Find rotation axis location by finding the offset between the first
    projection and a mirrored projection 180 degrees apart using
    phase correlation in Fourier space.
    The ``register_translation`` function uses cross-correlation in Fourier
    space, optionally employing an upsampled matrix-multiplication DFT to
    achieve arbitrary subpixel precision. :cite:`Guizar:08`.

    Parameters
    ----------
    proj1 : ndarray
        2D projection data.

    proj2 : ndarray
        2D projection data.

    tol : scalar, optional
        Subpixel accuracy

    rotc_guess : float, optional
        Initual guess value for the rotation center

    Returns
    -------
    float
        Rotation axis location.
    """
    imgshift = 0.0 if rotc_guess is None else rotc_guess - \
        (proj1.shape[1] - 1.0) / 2.0

    proj1 = ndimage.shift(proj1, [0, -imgshift], mode='constant', cval=0)
    proj2 = ndimage.shift(proj2, [0, -imgshift], mode='constant', cval=0)

    # create reflection of second projection
    proj2 = np.fliplr(proj2)

    # Determine shift between images using scikit-image pcm
    shift = register_translation(proj1, proj2, upsample_factor=1.0 / tol)

    # Compute center of rotation as the center of first image and the
    # registered translation with the second image
    center = (proj1.shape[1] + shift[0][1] - 1.0) / 2.0

    return center + imgshift


def write_center(
        tomo, theta, dpath='tmp/center', cen_range=None, ind=None,
        mask=False, ratio=1., sinogram_order=False, algorithm='gridrec', filter_name='parzen'):
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
    mask : bool, optional
        If ``True``, apply a circular mask to the reconstructed image to
        limit the analysis into a circular region.
    ratio : float, optional
        The ratio of the radius of the circular mask to the edge of the
        reconstructed image.
    sinogram_order: bool, optional
        Determins whether data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).
    algorithm : {str, function}
        One of the following string values.

        'art'
            Algebraic reconstruction technique :cite:`Kak:98`.
        'bart'
            Block algebraic reconstruction technique.
        'fbp'
            Filtered back-projection algorithm.
        'gridrec'
            Fourier grid reconstruction algorithm :cite:`Dowd:99`,
            :cite:`Rivers:06`.
        'mlem'
            Maximum-likelihood expectation maximization algorithm
            :cite:`Dempster:77`.
        'osem'
            Ordered-subset expectation maximization algorithm
            :cite:`Hudson:94`.
        'ospml_hybrid'
            Ordered-subset penalized maximum likelihood algorithm with
            weighted linear and quadratic penalties.
        'ospml_quad'
            Ordered-subset penalized maximum likelihood algorithm with
            quadratic penalties.
        'pml_hybrid'
            Penalized maximum likelihood algorithm with weighted linear
            and quadratic penalties :cite:`Chang:04`.
        'pml_quad'
            Penalized maximum likelihood algorithm with quadratic penalty.
        'sirt'
            Simultaneous algebraic reconstruction technique.
        'tv'
            Total Variation reconstruction technique
            :cite:`Chambolle:11`.
        'grad'
            Gradient descent method with a constant step size
        'tikh'
            Tikhonov regularization with identity Tikhonov matrix.            
            

    filter_name : str, optional
        Name of the filter for analytic reconstruction.

        'none'
            No filter.
        'shepp'
            Shepp-Logan filter (default).
        'cosine'
            Cosine filter.
        'hann'
            Cosine filter.
        'hamming'
            Hamming filter.
        'ramlak'
            Ram-Lak filter.
        'parzen'
            Parzen filter.
        'butterworth'
            Butterworth filter.
        'custom'
            A numpy array of size `next_power_of_2(num_detector_columns)/2`
            specifying a custom filter in Fourier domain. The first element
            of the filter should be the zero-frequency component.
        'custom2d'
            A numpy array of size `num_projections*next_power_of_2(num_detector_columns)/2`
            specifying a custom angle-dependent filter in Fourier domain. The first element
            of each filter should be the zero-frequency component.
    """
    tomo = dtype.as_float32(tomo)
    theta = dtype.as_float32(theta)

    if sinogram_order:
        dy, dt, dx = tomo.shape
    else:
        dt, dy, dx = tomo.shape
    if ind is None:
        ind = dy // 2
    if cen_range is None:
        center = np.arange(dx / 2 - 5, dx / 2 + 5, 0.5)
    else:
        center = np.arange(*cen_range)

    stack = dtype.empty_shared_array((len(center), dt, dx))

    for m in range(center.size):
        if sinogram_order:
            stack[m] = tomo[ind]
        else:
            stack[m] = tomo[:, ind, :]

    # Reconstruct the same slice with a range of centers.
    rec = recon(stack,
                theta,
                center=center,
                sinogram_order=True,
                algorithm=algorithm,
                filter_name=filter_name,
                nchunk=1)

    # Apply circular mask.
    if mask is True:
        rec = circ_mask(rec, axis=0)

    # Save images to a temporary folder.
    dpath = os.path.abspath(dpath)
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    for m in range(len(center)):
        write_tiff(data=rec[m], fname=dpath, digit='{0:.2f}'.format(center[m]))


def mask_empty_slice(tomo, threshold=0.25):
    """
    Generate a mask to indicate whether current slice contains sample

    At APS 1ID, some of the projection images contains large empty area above
    the sample, resulting in empty layers.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    threshold : float, optional
        determine whether a layer is considered to be empty

    Returns
    -------
    nparray:
        a mask indicate the emptyness of each layer
    """
    projs_sum = np.sum(tomo, axis=0)
    projs_sum /= projs_sum.max()

    projs_sumsum = np.sum(projs_sum, axis=0)
    projs_sumsum /= projs_sumsum.max()

    stds = np.array([np.std(projs_sum[i, :])
                     for i in range(projs_sum.shape[0])])
    std_ref = np.std(projs_sumsum) * threshold

    return np.array([std < std_ref for std in stds])
