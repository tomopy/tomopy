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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pywt
import pyfftw
import tomopy.prep.phase as phase
import tomopy.util.extern as extern
import tomopy.util.mproc as mproc
import tomopy.util.dtype as dtype
import logging
import concurrent.futures as cf

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Eduardo X. Miqueles"
__credits__ = "Juan V. Bermudez, Hugo H. Slepicka"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['remove_stripe_fw',
           'remove_stripe_ti',
           'remove_stripe_sf']

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
    
    tomo = dtype.as_float32(tomo)
    
    wname = pywt.Wavelet(wname)
    m = pywt.Modes.from_object('symmetric')
    def dwt(x, wname, m):
        a, d = pywt._extensions._dwt.dwt_axis(x, wname, m, 0)
        aa, ad = pywt._extensions._dwt.dwt_axis(a, wname, m, 1)
        da, dd = pywt._extensions._dwt.dwt_axis(d, wname, m, 1)
        return aa, da, ad, dd
    
    def idwt(aa, da, ad, dd, wname, m, out):
        a = pywt._extensions._dwt.idwt_axis(aa, ad, wname, m, 1)
        d = pywt._extensions._dwt.idwt_axis(da, dd, wname, m, 1)
        return pywt._extensions._dwt.idwt_axis(a, d, wname, m, 0, output=out)
    
    def runlevel(cH, cV, cD, sli, oshp, wname):
        for k in range(sli.shape[1]):
            sli[:oshp[0], k, :oshp[1]], cH[k], cV[k], cD[k] = dwt(sli[:, k, :], wname, m)
        
    
    dx, dy, dz = tomo.shape
    nx = dx
    if pad:
        nx = dx + dx // 8
    xshift = int((nx - dx) // 2)
    
    out = np.zeros((nx, dy, dz), dtype=np.float32)
    out[xshift:dx+xshift] = tomo
    
    axis_size = out.shape[1]
    ncore, nchunk = mproc.get_ncore_nchunk(axis_size, ncore, nchunk)
    chnks = np.round(np.linspace(0, axis_size, ncore+1)).astype(np.int)
    e = cf.ThreadPoolExecutor(ncore)
    
    cH = []
    cV = []
    cD = []
    sli = out[:, 0, :]
    slishp = np.zeros((level+1, 2), dtype=np.int)
    slishp[0] = sli.shape
    py, px = slishp[0]
    for n in range(level):
        my = pywt.dwt_coeff_len(py, wname.dec_len, m)
        mx = pywt.dwt_coeff_len(px, wname.dec_len, m)
        nm = np.array([my, mx])
        nm[nm%2==1]+=1
        slishp[n+1] = nm
        chn = np.zeros((dy, my, mx), dtype=np.float32)
        cvn = np.zeros((dy, my, mx), dtype=np.complex64)
        cdn = np.zeros((dy, my, mx), dtype=np.float32)
        y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
        damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
        damp = np.fft.ifftshift(damp)
        thrds = [e.submit(runlevel, 
                chn[chnks[i]:chnks[i+1]], cvn[chnks[i]:chnks[i+1]], 
                cdn[chnks[i]:chnks[i+1]], out[:py,chnks[i]:chnks[i+1],:px],
                [my, mx], wname) for i in range(ncore)]
        for t in thrds:
            t.result()
        plan = pyfftw.FFTW(cvn, cvn, axes=(1,), flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'), threads=ncore)
        plan.execute()
        cvn *= damp[np.newaxis, :, np.newaxis]
        plan = pyfftw.FFTW(cvn, cvn, axes=(1,), flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'), direction='FFTW_BACKWARD', threads=ncore)
        plan()
        cH.append(chn)
        cV.append(np.real(cvn))
        cD.append(cdn)
        py, px = my, mx
    
    def runleveli(cH, cV, cD, sli, oshp, wname):
        for k in range(sli.shape[1]):
             sli[:, k] = idwt(sli[:oshp[0], k, :oshp[1]], cH[k], cV[k], cD[k], wname, m)

    for n in range(level)[::-1]:
        chn = cH[n]
        cvn = np.real(cV[n])
        cdn = cD[n]
        csh = chn[0].shape
        my, mx = slishp[n]
        thrds = [e.submit(runleveli, 
                    chn[chnks[i]:chnks[i+1]], cvn[chnks[i]:chnks[i+1]],
                    cdn[chnks[i]:chnks[i+1]], out[:my,chnks[i]:chnks[i+1],:mx],
                    csh, wname) for i in range(ncore)]
        for t in thrds:
            t.result()

    return out[xshift:dx+xshift]

def _remove_stripe_fw(tomo, level, wname, sigma, pad):
    dx, dy, dz = tomo.shape
    nx = dx
    if pad:
        nx = dx + dx // 8
    xshift = int((nx - dx) // 2)

    num_jobs = tomo.shape[1]
    cH = []
    cV = []
    cD = []
    damps = []
    for m in range(num_jobs):
        sli = np.zeros((nx, dz), dtype='float32')
        sli[xshift:dx + xshift] = tomo[:, m, :]

        # Wavelet decomposition.

        for n in range(level):
            sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
            if m == 0:
                cH.append(np.zeros((num_jobs, cHt.shape[0], cHt.shape[1]), dtype=cHt.dtype))
                cV.append(np.zeros((num_jobs, cVt.shape[0], cVt.shape[1]), dtype=cVt.dtype))
                cD.append(np.zeros((num_jobs, cDt.shape[0], cDt.shape[1]), dtype=cDt.dtype))
                my, mx = cVt.shape
                y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
                damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
                damps.append(np.fft.ifftshift(damp))
            k = cH[n]
            k[m] = cHt
            k = cV[n]
            k[m] = cVt
            k = cD[n]
            k[m] = cDt
        if m == 0:
            slis = np.zeros((num_jobs, sli.shape[0], sli.shape[1]), dtype=sli.dtype)
        slis[m] = sli
    for n in range(level):
        out = pyfftw.interfaces.numpy_fft.fft(
                cV[n], axis=1, planner_effort='FFTW_ESTIMATE')
        out *= damps[n][np.newaxis, :, np.newaxis]
        cV[n] = np.real(pyfftw.interfaces.numpy_fft.ifft(
                out, axis=1,
                planner_effort='FFTW_ESTIMATE'))
    for m in range(num_jobs):
        # Wavelet reconstruction.
        sli = slis[m]
        for n in range(level)[::-1]:
            sli = sli[0:cH[n][m].shape[0], 0:cH[n][m].shape[1]]
            sli = pywt.idwt2((sli, (cH[n][m], cV[n][m], cD[n][m])), wname)
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


def remove_stripe_sf(tomo, size=5, ncore=None, nchunk=None, out=None):
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
    out : ndarray, optional
        Output array for result.  If same as tomo, process will be done in-place.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    tomo = dtype.as_float32(tomo)
    axis_size = tomo.shape[1]
    ncore, nchunk = mproc.get_ncore_nchunk(axis_size, ncore, nchunk)

    chnks = np.round(np.linspace(0, axis_size, ncore+1)).astype(np.int)
    c_cont = [tomo[:, chnks[i]:chnks[i+1]].copy('C')
                for i in range(ncore)] # Suboptimal
    mulargs = []
    for i in range(ncore):
        mulargs.append(extern.c_remove_stripe_sf(c_cont[i], size))
    e = cf.ThreadPoolExecutor(ncore)
    thrds = [e.submit(args[0], *args[1:]) for args in mulargs]
    for t in thrds:
        t.result()

    if out is None:
        out = np.empty_like(tomo)
    for i in range(ncore):
        out[:, chnks[i]:chnks[i+1]] = c_cont[i] # Suboptimal
    return out
