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
__all__ = [
    'remove_stripe_fw',
    'remove_stripe_ti',
    'remove_stripe_sf',
    'remove_stripe_based_sorting',
    'remove_stripe_based_filtering',
    'remove_stripe_based_fitting',
    'remove_large_stripe',
    'remove_dead_stripe',
    'remove_all_stripe',
    'remove_stripe_based_interpolation',
    'stripes_detect3d',
    'stripes_mask3d',
]


def remove_stripe_fw(tomo,
                     level=None,
                     wname='db5',
                     sigma=2,
                     pad=True,
                     ncore=None,
                     nchunk=None):
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

    arr = mproc.distribute_jobs(tomo,
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
            cV[n] = np.real(
                ifft(np.fft.ifftshift(fcV), axis=0, extra_info=num_jobs))

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
    arr = mproc.distribute_jobs(tomo,
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
    v = [
        [
            np.array([1, -1]),
            np.array([-3 / 2, 2, -1 / 2]),
            np.array([-11 / 6, 3, -3 / 2, 1 / 3]),
        ],
        [
            np.array([-1, 2, -1]),
            np.array([2, -5, 4, -1]),
        ],
        [
            np.array([-1, 3, -3, 1]),
        ],
    ]
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
    mysino[np.isnan(mysino)] = 0

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
    arr = mproc.distribute_jobs(tomo,
                                func=extern.c_remove_stripe_sf,
                                args=(size,),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def remove_stripe_based_sorting(tomo,
                                size=None,
                                dim=1,
                                ncore=None,
                                nchunk=None):
    """
    Remove full and partial stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 3).
    Suitable for removing partial stripes.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
                                func=_remove_stripe_based_sorting,
                                args=(size, dim),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = np.arange(0.0, ncol, 1.0)
    matindex = np.tile(listindex, (nrow, 1))
    return matindex


def _rs_sort(sinogram, size, matindex, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = np.transpose(sinogram)
    matcomb = np.asarray(np.dstack((matindex, sinogram)))
    matsort = np.asarray([row[row[:, 1].argsort()] for row in matcomb])
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))
    matsortback = np.asarray([row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = matsortback[:, :, 1]
    return np.transpose(sino_corrected)


def _remove_stripe_based_sorting(tomo, size, dim):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    if size is None:
        if tomo.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * tomo.shape[2]))
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_sort(sino, size, matindex, dim)


def remove_stripe_based_filtering(tomo,
                                  sigma=3,
                                  size=None,
                                  dim=1,
                                  ncore=None,
                                  nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 2).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    sigma : float
        Sigma of the Gaussian window which is used to separate
        the low-pass and high-pass components of the intensity
        profiles of each column. Recommended values: 3->10.
    size : int
        Window size of the median filter.
    dim : {1, 2}, optional
        Dimension of the window.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
                                func=_remove_stripe_based_filtering,
                                args=(sigma, size, dim),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def _create_1d_window(ncol, sigma, pad):
    window = signal.windows.gaussian(ncol + 2 * pad, std=sigma)
    return window


def _create_listsign(ncol, pad):
    listsign = np.power(-1.0, np.arange(ncol + 2 * pad))
    return listsign


def _rs_filter(sinogram, window, listsign, size, dim, pad):
    """
    Remove stripes using the filtering technique.
    """
    sinogram = np.transpose(sinogram)
    padded_sino = np.pad(sinogram, ((0, 0), (pad, pad)), mode='reflect')
    (nrow, ncol) = padded_sino.shape
    sinosmooth = np.zeros_like(sinogram)
    for i, sinolist in enumerate(padded_sino):
        sinosmooth[i] = np.real(
            ifft(fft(sinolist * listsign) * window) * listsign)[pad:ncol - pad]
    sinosharp = sinogram - sinosmooth
    matindex = _create_matindex(nrow, ncol - 2 * pad)
    sinosmooth_cor = np.transpose(
        _rs_sort(np.transpose(sinosmooth), size, matindex, dim))
    return np.transpose(sinosmooth_cor + sinosharp)


def _remove_stripe_based_filtering(tomo, sigma, size, dim):
    pad = min(150, int(0.1 * tomo.shape[0]))
    window = _create_1d_window(tomo.shape[0], sigma, pad)
    listsign = _create_listsign(tomo.shape[0], pad)
    if size is None:
        if tomo.shape[2] > 2000:
            size = 21
        else:
            size = max(5, int(0.01 * tomo.shape[2]))
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_filter(sino, window, listsign, size, dim, pad)


def remove_stripe_based_fitting(tomo,
                                order=3,
                                sigma=(5, 20),
                                ncore=None,
                                nchunk=None):
    """
    Remove stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 1).
    Suitable for removing low-pass stripes.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    order : int
        Polynomial fit order. Recommended values: 1-> 5
    sigma : tuple of 2 floats
        Sigmas of a 2D Gaussian window in x and y direction.
        Recommended values (3, 20) -> (10, 60).
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
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
    nrow = tomo.shape[0]
    ncol = tomo.shape[2]
    pad = min(150, int(0.1 * nrow))
    win2d = _create_2d_window(nrow, ncol, sigma, pad)
    matsign = _create_matsign(nrow, ncol, pad)
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_fit(sino, order, win2d, matsign, pad)


def remove_large_stripe(tomo,
                        snr=3,
                        size=51,
                        drop_ratio=0.1,
                        norm=True,
                        ncore=None,
                        nchunk=None):
    """
    Remove large stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (algorithm 5).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to locate of large stripes.
        Greater is less sensitive.
    size : int
        Window size of the median filter.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to reduce the false
        detection of stripes.
    norm : bool, optional
        Apply normalization if True.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
                                func=_remove_large_stripe,
                                args=(snr, size, drop_ratio, norm),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def _detect_stripe(listdata, snr):
    """
    Algorithm 4 in :cite:`Vo:18`. Used to locate stripes.
    """
    numdata = len(listdata)
    listsorted = np.sort(listdata)[::-1]
    xlist = np.arange(0, numdata, 1.0)
    ndrop = np.int16(0.25 * numdata)
    (_slope, _intercept) = np.polyfit(xlist[ndrop:-ndrop - 1],
                                      listsorted[ndrop:-ndrop - 1], 1)
    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = np.abs(numt1 - _intercept)
    noiselevel = np.clip(noiselevel, 1e-6, None)
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


def _rs_large(sinogram, snr, size, matindex, drop_ratio=0.1, norm=True):
    """
    Remove large stripes.
    """
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sinosort = np.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, size))
    list1 = np.mean(sinosort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sinosmooth[ndrop:nrow - ndrop], axis=0)
    listfact = np.divide(list1,
                         list2,
                         out=np.ones_like(list1),
                         where=list2 != 0)
    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    matfact = np.tile(listfact, (nrow, 1))
    # Normalize
    if norm is True:
        sinogram = sinogram / matfact
    sinogram1 = np.transpose(sinogram)
    matcombine = np.asarray(np.dstack((matindex, sinogram1)))
    matsort = np.asarray([row[row[:, 1].argsort()] for row in matcombine])
    matsort[:, :, 1] = np.transpose(sinosmooth)
    matsortback = np.asarray([row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = np.transpose(matsortback[:, :, 1])
    listxmiss = np.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram


def _remove_large_stripe(tomo, snr, size, drop_ratio, norm):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_large(sino, snr, size, matindex, drop_ratio, norm)


def remove_dead_stripe(tomo,
                       snr=3,
                       size=51,
                       norm=True,
                       ncore=None,
                       nchunk=None):
    """
    Remove unresponsive and fluctuating stripe artifacts from sinogram using
    Nghia Vo's approach :cite:`Vo:18` (algorithm 6).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to detect locations of large stripes.
        Greater is less sensitive.
    size : int
        Window size of the median filter.
    norm : bool, optional
        Remove residual stripes if True.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
                                func=_remove_dead_stripe,
                                args=(snr, size, norm),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def _rs_dead(sinogram, snr, size, matindex, norm=True):
    """
    Remove unresponsive and fluctuating stripes.
    """
    sinogram = np.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    sinosmooth = np.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    listdiff = np.sum(np.abs(sinogram - sinosmooth), axis=0)
    listdiffbck = median_filter(listdiff, size)
    listfact = np.divide(listdiff,
                         listdiffbck,
                         out=np.ones_like(listdiff),
                         where=listdiffbck != 0)
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = np.where(listmask < 1.0)[0]
    listy = np.arange(nrow)
    matz = sinogram[:, listx]
    finter = interpolate.RectBivariateSpline(listy, listx, matz,
                                             kx=1, ky=1)
    listxmiss = np.where(listmask > 0.0)[0]
    if len(listxmiss) > 0:
        matxmiss, maty = np.meshgrid(listxmiss, listy)
        output = finter.ev(np.ndarray.flatten(maty),
                           np.ndarray.flatten(matxmiss))
        sinogram[:, listxmiss] = output.reshape(matxmiss.shape)
    # Remove residual stripes
    if norm is True:
        sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


def _remove_dead_stripe(tomo, snr, size, norm):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        tomo[:, m, :] = _rs_dead(sino, snr, size, matindex, norm)


def remove_all_stripe(tomo,
                      snr=3,
                      la_size=61,
                      sm_size=21,
                      dim=1,
                      ncore=None,
                      nchunk=None):
    """
    Remove all types of stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (combination of algorithm 3,4,5, and 6).

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
    dim : {1, 2}, optional
        Dimension of the window.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
                                func=_remove_all_stripe,
                                args=(snr, la_size, sm_size, dim),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def _remove_all_stripe(tomo, snr, la_size, sm_size, dim):
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        sino = _rs_dead(sino, snr, la_size, matindex)
        sino = _rs_sort(sino, sm_size, matindex, dim)
        tomo[:, m, :] = sino


def remove_stripe_based_interpolation(tomo,
                                      snr=3,
                                      size=31,
                                      drop_ratio=0.1,
                                      norm=True,
                                      ncore=None,
                                      nchunk=None):
    """
    Remove most types of stripe artifacts from sinograms based on
    interpolation. Derived from algorithm 4, 5, and 6 in :cite:`Vo:18`.

    .. versionadded:: 1.9

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr : float
        Ratio used to segment between useful information and noise.
    size : int
        Window size of the median filter used to detect stripes.
    drop_ratio : float, optional
        Ratio of pixels to be dropped, which is used to to reduce
        the possibility of the false detection of stripes.
    norm : bool, optional
        Apply normalization if True.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(tomo,
                                func=_remove_stripe_based_interpolation,
                                args=(snr, size, drop_ratio, norm),
                                axis=1,
                                ncore=ncore,
                                nchunk=nchunk)
    return arr


def _rs_interpolation(sinogram, snr, size, drop_ratio=0.1, norm=True):
    """
    Remove stripe artifacts based on interpolation.
    """
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    sinogram = np.copy(sinogram)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sinosort = np.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, size))
    list1 = np.mean(sinosort[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sinosmooth[ndrop:nrow - ndrop], axis=0)
    listfact = np.divide(list1,
                         list2,
                         out=np.ones_like(list1),
                         where=list2 != 0)
    listmask = _detect_stripe(listfact, snr)
    listmask = np.float32(binary_dilation(listmask, iterations=1))
    matfact = np.tile(listfact, (nrow, 1))
    if norm is True:
        sinogram = sinogram / matfact
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = np.where(listmask < 1.0)[0]
    listy = np.arange(nrow)
    matz = sinogram[:, listx]
    finter = interpolate.RectBivariateSpline(listy, listx, matz,
                                             kx=1, ky=1)
    listxmiss = np.where(listmask > 0.0)[0]
    if len(listxmiss) > 0:
        matxmiss, maty = np.meshgrid(listxmiss, listy)
        output = finter.ev(np.ndarray.flatten(maty),
                           np.ndarray.flatten(matxmiss))
        sinogram[:, listxmiss] = output.reshape(matxmiss.shape)
    return sinogram


def _remove_stripe_based_interpolation(tomo, snr, size, drop_ratio, norm):
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        sino = _rs_interpolation(sino, snr, size, drop_ratio, norm)
        tomo[:, m, :] = sino


def stripes_detect3d(tomo, size=10, radius=3, ncore=None):
    """
    Detect stripes in a 3D array. Usually applied to normalized projection
    data. The algorithm is presented in the paper :cite:`Kazantsev:23`.

    The method works with full and partial stripes of constant ot varying
    intensity.

    .. versionadded:: 1.14

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data of float32 data type, preferably in the [0, 1]
        range, although reasonable deviations accepted (e.g. the result of the
        normalization and the negative log taken of the raw data). The
        projection data should be given with [angle, detY(depth),
        detX(horizontal)] axis orientation. With this orientation, the stripes
        are features along the angle axis.
    size : int, optional
        The pixel size of the 1D median filter orthogonal to stripes
        orientation to minimise false detections. Increase it if you have
        longer or full stripes in the data.
    radius : int, optional
        The pixel size of the 3D stencil to calculate the mean ratio between
        the angular and detX orientations of the detX gradient. The larger
        values can affect the width of the detected stripe, use 1,2,3 values.
    ncore : int, optional
        Number of cores that will be assigned to jobs. All cores will be used
        if unspecified.

    Returns
    -------
    ndarray
        Weights in the range of [0, 1] of float32 data type where stripe's
        edges are highlighted with the smaller (e.g. < 0.5) values. The weights
        can be manually thresholded or passed to stripes_mask3d function for
        further processing and a binary mask generation.

    Raises
    ------
    ValueError
        If the `tomo` is not three dimensional.

        If the `size` is invalid.
    """
    if ncore is None:
        ncore = mproc.mp.cpu_count()

    input_type = tomo.dtype
    if (input_type != 'float32'):
        tomo = dtype.as_float32(tomo)  # silent convertion to float32 data type
    out = np.empty_like(tomo, order='C')

    if tomo.ndim == 3:
        dz, dy, dx = tomo.shape
        if (dz == 0) or (dy == 0) or (dx == 0):
            msg = "The length of one of dimensions is equal to zero"
            raise ValueError(msg)
    else:
        msg = "The input array must be a 3D array"
        raise ValueError(msg)

    if size <= 0 or size > dz // 2:
        msg = ("The size of the filter should be larger than zero "
               "and smaller than the half of the vertical dimension")
        raise ValueError(msg)

    # perform stripes detection
    extern.c_stripes_detect3d(np.ascontiguousarray(tomo), out, size, radius,
                              ncore, dx, dy, dz)
    return out


def stripes_mask3d(weights,
                   threshold=0.6,
                   min_stripe_length=20,
                   min_stripe_depth=10,
                   min_stripe_width=5,
                   sensitivity_perc=85.0,
                   ncore=None):
    """
    Takes the result of the stripes_detect3d module as an input and generates a
    binary 3D mask with ones where stripes present.

    The method tries to eliminate non-stripe features in data by checking the
    consistency of weights in three directions. The algorithm is
    presented in the paper :cite:`Kazantsev:23`.

    .. versionadded:: 1.14

    Parameters
    ----------
    weights : ndarray
        3D weights array, a result of stripes_detect3d module given in [angles,
        detY(depth), detX] axis orientation.
    threshold : float, optional
        Threshold for the given weights. This parameter defines what weights
        will be considered as potential candidates for stripes. The lower value
        (< 0.5) will result in only the most prominent stripes in the data.
        Increase the threshold cautiously because increasing the threshold
        increases the probability of false detections. The good range to try is
        between 0.5 and 0.7.
    min_stripe_length : int, optional
        Minimum length of a stripe in pixels with respect to the "angles" axis.
        If there are full stripes in the data, then this could be >50% of the
        size of the the "angles" axis.
    min_stripe_depth : int, optional
        Minimum depth of a stripe in pixels with respect to the "detY" axis.
        The stripes do not extend very deep normally in the data. By setting
        this parameter to the approximate depth of the stripe more false alarms
        can be removed.
    min_stripe_width : int, optional
        Minimum width of a stripe in pixels with respect to the "detX" axis.
        The stripes that close to each other can be merged together with this
        parameter.
    sensitivity_perc : float, optional
        The value in the range [0, 100] that controls the strictness of the
        minimum length, depth and width parameters of a stripe. 0 is
        less-strict. 100 is more-strict.
    ncore : int, optional
        Number of cores that will be assigned to jobs. All cores will be used
        if unspecified.

    Returns
    -------
    ndarray
        A binary mask of bool data type with stripes highlighted as True
        values.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.

        If a min_stripe_length parameter is negative, zero or longer than its
        corresponding dimension ("angle")

        If a min_stripe_depth parameter is negative or longer than its
        corresponding dimension ("detY")

        If a min_stripe_width parameter is negative, zero or longer than its
        corresponding dimension ("detX")

        If a sensitivity_perc parameter doesn't lie in the (0,100] range

    """
    if ncore is None:
        ncore = mproc.mp.cpu_count()

    input_type = weights.dtype
    if (input_type != 'float32'):
        weights = dtype.as_float32(
            weights)  # silent convertion to float32 data type
    out = np.zeros(np.shape(weights), dtype=bool, order='C')

    if weights.ndim == 3:
        dz, dy, dx = weights.shape
        if (dz == 0) or (dy == 0) or (dx == 0):
            msg = "The length of one of dimensions is equal to zero"
            raise ValueError(msg)
    else:
        msg = "The input array must be a 3D array"
        raise ValueError(msg)

    if min_stripe_length <= 0 or min_stripe_length >= dz:
        msg = ("The minimum length of a stripe cannot be zero "
               "or exceed the size of the angular dimension")
        raise ValueError(msg)

    if min_stripe_depth < 0 or min_stripe_depth >= dy:
        msg = ("The minimum depth of a stripe cannot exceed "
               "the size of the depth dimension")
        raise ValueError(msg)

    if min_stripe_width <= 0 or min_stripe_width >= dx:
        msg = ("The minimum width of a stripe cannot be zero "
               "or exceed the size of the horizontal dimension")
        raise ValueError(msg)

    if 0.0 < sensitivity_perc <= 100.0:
        pass
    else:
        msg = "sensitivity_perc value must be in (0, 100] percentage range"
        raise ValueError(msg)

    # perform mask creation based on the input provided by stripes_detect3d module
    extern.c_stripesmask3d(
        np.ascontiguousarray(weights),
        out,
        threshold,
        min_stripe_length,
        min_stripe_depth,
        min_stripe_width,
        sensitivity_perc,
        ncore,
        dx,
        dy,
        dz,
    )
    return out
