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
Module for pre-processing tasks.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pywt
import tomopy.util.mproc as mproc
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Eduardo X. Miqueles"
__credits__ = "Juan V. Bermudez, Hugo H. Slepicka"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['remove_stripe_fw',
           'remove_stripe_ti']


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


def _remove_stripe_fw(level, wname, sigma, pad, istart, iend):
    tomo = mproc.SHARED_ARRAY
    dx, dy, dz = tomo.shape
    nx = dx
    if pad:
        nx = dx + dx / 8
    xshift = int((nx - dx) / 2.)

    for m in range(istart, iend):
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
            fcV = np.fft.fftshift(np.fft.fft(cV[n], axis=0))
            my, mx = fcV.shape

            # Damping of ring artifact information.
            y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
            fcV = np.multiply(fcV, np.transpose(np.tile(damp, (mx, 1))))

            # Inverse FFT.
            cV[n] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

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


def _remove_stripe_ti(nblock, alpha, istart, iend):
    tomo = mproc.SHARED_ARRAY
    for m in range(istart, iend):
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
