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
Module for pre-processing tasks.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pywt
import tomopy.util.extern as extern
import tomopy.util.mproc as mproc
import tomopy.util.dtype as dtype
from tomopy.util.misc import (fft, ifft, fft2, ifft2)
from scipy.ndimage import median_filter
from scipy import signal
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import uniform_filter1d
from scipy import interpolate
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Eduardo X. Miqueles, Nghia Vo"
__credits__ = "Juan V. Bermudez, Hugo H. Slepicka"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['remove_stripe_fw',
           'remove_stripe_ti',
           'remove_stripe_sf',
           'remove_stripe_based_sorting',
           'remove_stripe_based_filtering',
           'remove_stripe_based_fitting',
           'remove_large_stripe',
           'remove_dead_stripe',
           'remove_all_stripe']


def remove_stripe_fw(
        tomo, level=None, wname='db5', sigma=2,
        pad=True, ncore=None, nchunk=None):
    """
    Remove horizontal stripes from sinogram using the Fourier-Wavelet (FW)
    based method :cite:`Munch:09`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    level : int, optional
        Number of discrete wavelet transform levels.
    wname : str, optional
        Type of the wavelet filter. 'haar', 'db5', sym5', etc.
    sigma : float, optional
        Damping parameter in Fourier space.
    pad : bool, optional
        If True, extend the size of the sinogram by padding with zeros.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    if level is None:
        size = np.max(tomo.shape)
        level = int(np.ceil(np.log2(size)))

    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_stripe_fw,
        args=(level, wname, sigma, pad),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _remove_stripe_fw(tomo, level, wname, sigma, pad):
    dx, dy, dz = tomo.shape
    nx = dx
    if pad:
        nx = dx + dx // 8
    xshift = int((nx - dx) // 2)

    num_jobs = tomo.shape[1]

    for m in range(num_jobs):
        sli = np.zeros((nx, dz), dtype='float32')
        sli[xshift:dx + xshift] = tomo[:, m, :]

        # Wavelet decomposition.
        cH = []
        cV = []
        cD = []
        for n in range(level):
            sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
            cH.append(cHt)
            cV.append(cVt)
            cD.append(cDt)

        # FFT transform of horizontal frequency bands.
        for n in range(level):
            # FFT
            fcV = np.fft.fftshift(fft(cV[n], axis=0, extra_info=num_jobs))
            my, mx = fcV.shape

            # Damping of ring artifact information.
            y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = -np.expm1(-np.square(y_hat) / (2 * np.square(sigma)))
            fcV *= np.transpose(np.tile(damp, (mx, 1)))

            # Inverse FFT.
            cV[n] = np.real(ifft(np.fft.ifftshift(
                fcV), axis=0, extra_info=num_jobs))

        # Wavelet reconstruction.
        for n in range(level)[::-1]:
            sli = sli[0:cH[n].shape[0], 0:cH[n].shape[1]]
            sli = pywt.idwt2((sli, (cH[n], cV[n], cD[n])), wname)
        tomo[:, m, :] = sli[xshift:dx + xshift, 0:dz]


def remove_stripe_ti(tomo, nblock=0, alpha=1.5, ncore=None, nchunk=None):
    """
    Remove horizontal stripes from sinogram using Titarenko's
    approach :cite:`Miqueles:14`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    nblock : int, optional
        Number of blocks.
    alpha : int, optional
        Damping factor.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_stripe_ti,
        args=(nblock, alpha),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _remove_stripe_ti(tomo, nblock, alpha):
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        if nblock == 0:
            d1 = _ring(sino, 1, 1)
            d2 = _ring(sino, 2, 1)
            p = d1 * d2
            d = np.sqrt(p + alpha * np.abs(p.min()))
        else:
            size = int(sino.shape[0] / nblock)
            d1 = _ringb(sino, 1, 1, size)
            d2 = _ringb(sino, 2, 1, size)
            p = d1 * d2
            d = np.sqrt(p + alpha * np.fabs(p.min()))
        tomo[:, m, :] = d


def _kernel(m, n):
    v = [[np.array([1, -1]),
          np.array([-3 / 2, 2, -1 / 2]),
          np.array([-11 / 6, 3, -3 / 2, 1 / 3])],
         [np.array([-1, 2, -1]),
          np.array([2, -5, 4, -1])],
         [np.array([-1, 3, -3, 1])]]
    return v[m - 1][n - 1]


def _ringMatXvec(h, x):
    s = np.convolve(x, np.flipud(h))
    u = s[np.size(h) - 1:np.size(x)]
    y = np.convolve(u, h)
    return y


def _ringCGM(h, alpha, f):
    x0 = np.zeros(np.size(f))
    r = f - (_ringMatXvec(h, x0) + alpha * x0)
    w = -r
    z = _ringMatXvec(h, w) + alpha * w
    a = np.dot(r, w) / np.dot(w, z)
    x = x0 + np.dot(a, w)
    B = 0
    for i in range(1000000):
        r = r - np.dot(a, z)
        if np.linalg.norm(r) < 0.0000001:
            break
        B = np.dot(r, z) / np.dot(w, z)
        w = -r + np.dot(B, w)
        z = _ringMatXvec(h, w) + alpha * w
        a = np.dot(r, w) / np.dot(w, z)
        x = x + np.dot(a, w)
    return x


def _get_parameter(x):
    return 1 / (2 * (x.sum(0).max() - x.sum(0).min()))


def _ring(sino, m, n):
    mysino = np.transpose(sino)
    R = np.size(mysino, 0)
    N = np.size(mysino, 1)

    # Remove NaN.
    pos = np.where(np.isnan(mysino) is True)
    mysino[pos] = 0

    # Parameter.
    alpha = _get_parameter(mysino)

    # Mathematical correction.
    pp = mysino.mean(1)
    h = _kernel(m, n)
    f = -_ringMatXvec(h, pp)
    q = _ringCGM(h, alpha, f)

    # Update sinogram.
    q.shape = (R, 1)
    K = np.kron(q, np.ones((1, N)))
    new = np.add(mysino, K)
    newsino = new.astype(np.float32)
    return np.transpose(newsino)


def _ringb(sino, m, n, step):
    mysino = np.transpose(sino)
    R = np.size(mysino, 0)
    N = np.size(mysino, 1)

    # Remove NaN.
    pos = np.where(np.isnan(mysino) is True)
    mysino[pos] = 0

    # Kernel & regularization parameter.
    h = _kernel(m, n)

    # Mathematical correction by blocks.
    nblock = int(N / step)
    new = np.ones((R, N))
    for k in range(0, nblock):
        sino_block = mysino[:, k * step:(k + 1) * step]
        alpha = _get_parameter(sino_block)
        pp = sino_block.mean(1)

        f = -_ringMatXvec(h, pp)
        q = _ringCGM(h, alpha, f)

        # Update sinogram.
        q.shape = (R, 1)
        K = np.kron(q, np.ones((1, step)))
        new[:, k * step:(k + 1) * step] = np.add(sino_block, K)
    newsino = new.astype(np.float32)
    return np.transpose(newsino)


def remove_stripe_sf(tomo, size=5, ncore=None, nchunk=None):
    """
    Normalize raw projection data using a smoothing filter approach.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    size : int, optional
        Size of the smoothing filter.
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
    arr = mproc.distribute_jobs(
        tomo,
        func=extern.c_remove_stripe_sf,
        args=(size,),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def remove_stripe_based_sorting(tomo, size=None, ncore=None, nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18`
    Algorithm 3 in the paper. Remove stripes using the sorting technique.
    Work particularly well for removing partial stripes.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    size : int
        Window size of the median filter.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_stripe_based_sorting,
        args=(size,),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for sorting technique
    """
    listindex = np.arange(0.0, ncol, 1.0)
    matindex = np.tile(listindex, (nrow, 1))
    return matindex


def _rs_sort(sinogram, size, matindex):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = np.transpose(sinogram)
    matcomb = np.asarray(np.dstack((matindex, sinogram)))
    matsort = np.asarray(
        [row[row[:, 1].argsort()] for row in matcomb])
    matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    matsortback = np.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = matsortback[:, :, 1]
    return np.transpose(sino_corrected)


def _remove_stripe_based_sorting(tomo, size):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    if size is None:
        if tomo.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * tomo.shape[2]))
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_sort(sino, size, matindex)


def remove_stripe_based_filtering(
        tomo, sigma=3, size=None, ncore=None, nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18`
    Algorithm 2 in the paper. Remove stripes using the filtering technique.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    sigma : float
        Sigma of the Gaussian window which is used to separate 
        the low-pass and high-pass components of the intensity
        profiles of each column.
    size : int
        Window size of the median filter.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_stripe_based_filtering,
        args=(sigma, size),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _create_1d_window(ncol, sigma, pad):
    window = signal.gaussian(ncol + 2 * pad, std=sigma)
    return window


def _create_listsign(ncol, pad):
    listsign = np.power(-1.0, np.arange(ncol + 2 * pad))
    return listsign


def _rs_filter(sinogram, window, listsign, sigma, size, pad):
    """
    Remove stripes using the filtering technique.
    """
    sinogram = np.transpose(sinogram)
    padded_sino = np.pad(sinogram, ((0, 0), (pad, pad)), mode='reflect')
    (_, ncol) = padded_sino.shape
    sinosmooth = np.zeros_like(sinogram)
    for i, sinolist in enumerate(padded_sino):
        sinosmooth[i] = np.real(
            ifft(fft(sinolist * listsign) * window) * listsign)[pad:ncol - pad]
    sinosharp = sinogram - sinosmooth
    sinosmooth_cor = median_filter(sinosmooth, (size, 1))
    return np.transpose(sinosmooth_cor + sinosharp)


def _remove_stripe_based_filtering(tomo, sigma, size):
    pad = 150  # To reduce artifacts caused by FFT
    window = _create_1d_window(tomo.shape[0], sigma, pad)
    listsign = _create_listsign(tomo.shape[0], pad)
    if size is None:
        if tomo.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * tomo.shape[2]))
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_filter(
            sino, window, listsign, sigma, size, pad)


def remove_stripe_based_fitting(
        tomo, order=3, sigma=(5, 20), ncore=None, nchunk=None):
    """
    Remove horizontal stripes from sinogram using Nghia Vo's
    approach :cite:`Vo:18`
    Algorithm 1 in the paper. Remove stripes using the fitting technique.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    order : int
        Polynomial fit order.
    sigma : tuple of 2 floats
        Sigmas of a 2D Gaussian window in x and y direction.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_stripe_based_fitting,
        args=(order, sigma),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _create_2d_window(nrow, ncol, sigma, pad):
    """
    Create a 2D Gaussian window.

    Parameters
    ----------
    nrow : int
        Height of the window.
    ncol: int 
        Width of the window.
    sigma: tuple of 2 floats
        Sigmas of the window.
    pad : int
        Padding.

    Returns
    -------
        2D array.
    """
    (sigmax, sigmay) = sigma
    height = nrow + 2 * pad
    width = ncol + 2 * pad
    centerx = (width - 1.0) / 2.0
    centery = (height - 1.0) / 2.0
    y, x = np.ogrid[-centery:height - centery, -centerx:width - centerx]
    numx = 2.0 * sigmax * sigmax
    numy = 2.0 * sigmay * sigmay
    win2d = np.exp(-(x * x / numx + y * y / numy))
    return win2d


def _create_matsign(nrow, ncol, pad):
    listx = 1.0 * np.arange(0, ncol + 2 * pad)
    listy = 1.0 * np.arange(0, nrow + 2 * pad)
    x, y = np.meshgrid(listx, listy)
    matsign = np.power(-1.0, x + y)
    return matsign


def _2d_filter(mat, win2d, matsign, pad):
    """
    Filtering an image using a 2D window.

    Parameters
    ----------
    mat : 2D array of floats
    nrow : int
        Height of the window.
    ncol: int 
        Width of the window.
    sigma: tuple of 2 floats
        Sigmas of the window.
    pad : int
        Padding.

    Returns
    -------    
        Filtered image.
    """
    matpad = np.pad(mat, ((0, 0), (pad, pad)), mode='edge')
    matpad = np.pad(matpad, ((pad, pad), (0, 0)), mode='mean')
    (nrow, ncol) = matpad.shape
    matfilter = np.real(ifft2(fft2(matpad * matsign) * win2d) * matsign)
    return matfilter[pad:nrow - pad, pad:ncol - pad]


def _rs_fit(sinogram, order, win2d, matsign, pad):
    """
    Remove stripes using the fitting technique.
    """
    (nrow, _) = sinogram.shape
    nrow1 = nrow
    if nrow1 % 2 == 0:
        nrow1 = nrow1 - 1
    if order >= nrow1:
        order = nrow1 - 1
    sinofit = savgol_filter(sinogram, nrow1, order, axis=0, mode='mirror')
    sinofitsmooth = _2d_filter(sinofit, win2d, matsign, pad)
    num1 = np.mean(sinofit)
    num2 = np.mean(sinofitsmooth)
    sinofitsmooth = num1 * sinofitsmooth / num2
    return sinogram / sinofit * sinofitsmooth


def _remove_stripe_based_fitting(tomo, order, sigma):
    pad = 50  # To reduce artifacts caused by FFT
    nrow = tomo.shape[0]
    ncol = tomo.shape[2]
    win2d = _create_2d_window(nrow, ncol, sigma, pad)
    matsign = _create_matsign(nrow, ncol, pad)
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_fit(sino, order, win2d, matsign, pad)


def remove_large_stripe(tomo, snr=3, size=51, ncore=None, nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18`
    Algorithm 5 in the paper. Remove large stripes.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to locate of large stripes.
        Greater is less sensitive. 
    size : int
        Window size of the median filter.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_large_stripe,
        args=(snr, size),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _detect_stripe(listdata, snr):
    """
    Algorithm 4 in the paper. Used to locate stripes.
    """
    numdata = len(listdata)
    listsorted = np.sort(listdata)[::-1]
    xlist = np.arange(0, numdata, 1.0)
    ndrop = np.int16(0.25 * numdata)
    (_slope, _intercept) = np.polyfit(
        xlist[ndrop:-ndrop - 1], listsorted[ndrop:-ndrop - 1], 1)
    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = np.abs(numt1 - _intercept)
    val1 = np.abs(listsorted[0] - _intercept) / noiselevel
    val2 = np.abs(listsorted[-1] - numt1) / noiselevel
    listmask = np.zeros_like(listdata)
    if (val1 >= snr):
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask[listdata > upper_thresh] = 1.0
    if (val2 >= snr):
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask[listdata <= lower_thresh] = 1.0
    return listmask


def _rs_large(sinogram, snr, size, matindex):
    """
    Remove large stripes by: locating stripes, normalizing to remove
    full stripes, using the sorting technique to remove partial stripes.    
    """
    badpixelratio = 0.05
    (nrow, ncol) = sinogram.shape
    ndrop = np.int16(badpixelratio * nrow)
    sinosorted = np.sort(sinogram, axis=0)
    sinosmoothed = median_filter(sinosorted, (1, size))
    list1 = np.mean(sinosorted[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sinosmoothed[ndrop:nrow - ndrop], axis=0)
    listfact = list1 / list2
    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    matfact = np.tile(listfact, (nrow, 1))
    # Normalize
    sinogram = sinogram / matfact
    sinogram1 = np.transpose(sinogram)
    matcombine = np.asarray(np.dstack((matindex, sinogram1)))
    matsort = np.asarray(
        [row[row[:, 1].argsort()] for row in matcombine])
    matsort[:, :, 1] = np.transpose(sinosmoothed)
    matsortback = np.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = np.transpose(matsortback[:, :, 1])
    listxmiss = np.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram


def _remove_large_stripe(tomo, snr, size):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_large(sino, snr, size, matindex)


def remove_dead_stripe(tomo, snr=3, size=51, ncore=None, nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18`
    Algorithm 6 in the paper. Remove unresponsive and fluctuating stripes.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to detect locations of large stripes.
        Greater is less sensitive. 
    size : int
        Window size of the median filter.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_dead_stripe,
        args=(snr, size),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _rs_dead(sinogram, snr, size, matindex):
    """
    Remove unresponsive and fluctuating stripes.
    """
    sinogram = np.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    sinosmoothed = np.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    listdiff = np.sum(np.abs(sinogram - sinosmoothed), axis=0)
    nmean = np.mean(listdiff)
    listdiffbck = median_filter(listdiff, size)
    listdiffbck[listdiffbck == 0.0] = nmean
    listfact = listdiff / listdiffbck
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = np.where(listmask < 1.0)[0]
    listy = np.arange(nrow)
    matz = sinogram[:, listx]
    finter = interpolate.interp2d(listx, listy, matz, kind='linear')
    listxmiss = np.where(listmask > 0.0)[0]
    if len(listxmiss) > 0:
        matzmiss = finter(listxmiss, listy)
        sinogram[:, listxmiss] = matzmiss
    # Use algorithm 5 to remove residual stripes
    sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


def _remove_dead_stripe(tomo, snr, size):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_dead(sino, snr, size, matindex)


def remove_all_stripe(tomo, snr=3, la_size=61, sm_size=21, ncore=None, nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18`
    Combine algorithms 6,5,4,3 to remove all types of stripes.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to locate large stripes.
        Greater is less sensitive. 
    la_size : int
        Window size of the median filter to remove large stripes.
    sm_size : int
        Window size of the median filter to remove small-to-medium stripes.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_remove_all_stripe,
        args=(snr, la_size, sm_size),
        axis=1,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _remove_all_stripe(tomo, snr, la_size, sm_size):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        sino = _rs_dead(sino, snr, la_size, matindex)
        sino = _rs_sort(sino, sm_size, matindex)
        tomo[:, m, :] = sino
