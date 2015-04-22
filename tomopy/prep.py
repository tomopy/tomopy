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
import os
import ctypes
import tomopy.misc.mproc as mp
from scipy.ndimage import filters
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['circular_roi',
           'correct_air',
           'focus_region',
           'normalize',
           'remove_stripe',
           'remove_stripe2',
           'remove_zinger',
           'retrieve_phase']


BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PI = 3.14159265359


def _import_shared_lib(lib_name):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            libpath = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..', 'lib', lib_name + '.pyd'))
            return ctypes.CDLL(libpath)
        else:
            libpath = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..', 'lib', lib_name + '.so'))
            return ctypes.CDLL(libpath)
    except OSError as e:
        logger.warning('OSError: Shared library missing.')


LIB_TOMOPY = _import_shared_lib('libtomopy')


def circular_roi(tomo, ratio=1, val=None):
    """
    Apply circular mask to projection images.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ratio : int, optional
        Ratio of the circular mask's diameter in pixels to
        the number of reconstructed image grid size.
    val : int, optional
        Value for the masked region.

    Returns
    -------
    ndarray
        Masked 3D tomographic data.
    """
    dx, dy, dz = tomo.shape
    ind = np.arange(0, dx)

    ind1 = dy
    ind2 = dz
    rad1 = ind1 / 2
    rad2 = ind2 / 2
    if dy < dz:
        r2 = rad1 * rad1
    else:
        r2 = rad2 * rad2
    y, x = np.ogrid[-rad1:rad1, -rad2:rad2]
    mask = x * x + y * y > ratio * ratio * r2
    print(mask.shape, tomo.shape)
    if val is None:
        val = np.mean(tomo[:, ~mask])

    for m in ind:
        tomo[m, mask] = val
    return tomo


def correct_air(tomo, air=10):
    """
    Weights sinogram such that the left and right image boundaries
    (i.e., typically the air region around the object) are set to one
    and all intermediate values are scaled linearly.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    air : int, optional
        Number of pixels at each boundary to calculate the scaling factor.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    dx, dy, dz = tomo.shape

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(air, np.int32):
        air = np.array(air, dtype='int32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.correct_air.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.correct_air(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx), ctypes.c_int(dy),
        ctypes.c_int(dz), ctypes.c_int(air))
    return tomo


def focus_region(
        tomo, dia, xcoord=0, ycoord=0,
        center=None, pad=False, corr=True):
    """
    Trims sinogram for reconstructing a circular region of interest (ROI).

    Note: Only valid for 0-180 degree span data.

    Parameters
    ----------
    tomo : ndarray
        3D Tomographic data.
    xcoord, ycoord : float, optional
        x- and y-coordinates of the center location of the circular
        ROI in reconstruction image.
    dia : float, optional
        Diameter of the circular ROI.
    center : float, optional
        Rotation axis location of the tomographic data.
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    corr : bool, optional
        If True, correct_air is applied after data is trimmed.

    Returns
    -------
    ndarray
        Modified 3D tomographic tomo.
    float
        New rotation axis location.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    roi = np.ones((dx, dy, dia), dtype='float32')
    if pad:
        roi = np.ones((dx, dy, dz), dtype='float32')
    rad = np.sqrt(xcoord * xcoord + ycoord * ycoord)
    alpha = np.arctan2(xcoord, ycoord)
    l1 = center - dia / 2
    l2 = center - dia / 2 + rad
    delphi = PI / dx
    for m in np.arange(0, dx):
        ind1 = np.ceil(np.cos(alpha - m * delphi) * (l2 - l1) + l1)
        ind2 = np.floor(np.cos(alpha - m * delphi) * (l2 - l1) + l1 + dia)
        if ind1 < 0:
            ind1 = 0
        if ind2 < 0:
            ind2 = 0
        if ind1 > dz:
            ind1 = dz
        if ind2 > dz:
            ind2 = dz
        arr = np.expand_dims(tomo[m, :, ind1:ind2], axis=0)
        if pad:
            if corr:
                roi[m, :, ind1:ind2] = correct_air(arr.copy(), air=5)
            else:
                roi[m, :, ind1:ind2] = arr
        else:
            if corr:
                roi[m, :, 0:(ind2 - ind1)] = correct_air(arr, air=5)
            else:
                roi[m, :, 0:(ind2 - ind1)] = arr
        if not pad:
            center = dz / 2.
    return roi, center


def normalize(tomo, flat, dark, cutoff=None, ind=None):
    """
    Normalize raw projection data using the flat and dark field projections.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    flat : ndarray
        3D flat field data.
    dark : ndarray
        3D dark field data.
    cutoff : float, optional
        Permitted maximum vaue for the normalized data.
    ind : array of int, optional
        Projection indices at which the normalization is applied.

    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    if type(tomo) == str and tomo == 'SHARED':
        tomo = mp.shared_arr
    else:
        arr = mp.distribute_jobs(
            tomo, func=normalize, axis=0,
            args=(flat, dark, cutoff))
        return arr

    dx, dy, dz = tomo.shape
    if ind is None:
        ind = np.arange(0, dx)

    # Calculate average flat and dark fields for normalization.
    flat = flat.mean(axis=0)
    dark = dark.mean(axis=0)

    # Avoid zero division in normalization
    denom = flat - dark
    denom[denom == 0] = 1e-6

    for m in ind:
        proj = tomo[m, :, :]
        proj = np.divide(proj - dark, denom)
        if cutoff is not None:
            proj[proj > cutoff] = cutoff
        tomo[m, :, :] = proj


def remove_stripe(
        tomo, level=None, wname='db5',
        sigma=2, pad=True, ind=None):
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
    ind : array of int, optional
        Sinogram indices at which the stripe removal is applied.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    if type(tomo) == str and tomo == 'SHARED':
        tomo = mp.shared_arr
    else:
        arr = mp.distribute_jobs(
            tomo, func=remove_stripe, axis=1,
            args=(level, wname, sigma, pad))
        return arr

    dx, dy, dz = tomo.shape
    if ind is None:
        ind = np.arange(0, dy)
    if level is None:
        size = np.max(tomo.shape)
        level = int(np.ceil(np.log2(size)))

    # pad temp image.
    nx = dx
    if pad:
        nx = dx + dx / 8

    xshift = int((nx - dx) / 2.)
    sli = np.zeros((nx, dz), dtype='float32')

    for n in ind:
        sli[xshift:dx + xshift, :] = tomo[:, n, :]

        # Wavelet decomposition.
        cH = []
        cV = []
        cD = []
        for m in range(level):
            sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
            cH.append(cHt)
            cV.append(cVt)
            cD.append(cDt)

        # FFT transform of horizontal frequency bands.
        for m in range(level):
            # FFT
            fcV = np.fft.fftshift(np.fft.fft(cV[m], axis=0))
            my, mx = fcV.shape

            # Damping of ring artifact information.
            y_hat = (np.arange(-my, my, 2, dtype='float') + 1) / 2
            damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
            fcV = np.multiply(fcV, np.transpose(np.tile(damp, (mx, 1))))

            # Inverse FFT.
            cV[m] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

        # Wavelet reconstruction.
        for m in range(level)[::-1]:
            sli = sli[0:cH[m].shape[0], 0:cH[m].shape[1]]
            sli = pywt.idwt2((sli, (cH[m], cV[m], cD[m])), wname)

        tomo[:, n, :] = sli[xshift:dx + xshift, 0:dz]


def remove_stripe2(tomo, nblock=0, alpha=1.5, ind=None):
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
    ind : array of int, optional
        Sinogram indices at which the stripe removal is applied.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    if type(tomo) == str and tomo == 'SHARED':
        tomo = mp.shared_arr
    else:
        arr = mp.distribute_jobs(
            tomo, func=remove_stripe2, axis=1,
            args=(nblock, alpha))
        return arr

    dx, dy, dz = tomo.shape
    if ind is None:
        ind = np.arange(0, dy)

    for n in ind:
        sino = tomo[:, n, :]
        if (nblock == 0):
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
        tomo[:, n, :] = d


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
        if (np.linalg.norm(r) < 0.0000001):
            break
        B = np.dot(r, z) / np.dot(w, z)
        w = -r + np.dot(B, w)
        z = _ringMatXvec(h, w) + alpha * w
        a = np.dot(r, w) / np.dot(w, z)
        x = x + np.dot(a, w)
    return x


def _ring(sino, m, n):
    mysino = np.transpose(sino)
    R = np.size(mysino, 0)
    N = np.size(mysino, 1)

    # Remove NaN.
    pos = np.where(np.isnan(mysino) is True)
    mysino[pos] = 0

    # Parameter.
    alpha = 1 / (2 * (mysino.sum(0).max() - mysino.sum(0).min()))

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
        alpha = 1 / (2 * (sino_block.sum(0).max() - sino_block.sum(0).min()))
        pp = sino_block.mean(1)

        f = -_ringMatXvec(h, pp)
        q = _ringCGM(h, alpha, f)

        # Update sinogram.
        q.shape = (R, 1)
        K = np.kron(q, np.ones((1, step)))
        new[:, k * step:(k + 1) * step] = np.add(sino_block, K)
    newsino = new.astype(np.float32)
    return np.transpose(newsino)


def remove_zinger(tomo, dif=1000, size=3, ind=None):
    """
    Remove high intensity bright spots from 3D tomographic data.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    dif : float, optional
        Expected difference value between outlier measurements and
        the median filtered raw measurements.
    size : int, optional
        Size of the median filter.
    ind : array of int, optional
        Projection indices at which the zinger removal is applied.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    if type(tomo) == str and tomo == 'SHARED':
        tomo = mp.shared_arr
    else:
        arr = mp.distribute_jobs(
            tomo, func=remove_zinger, axis=0,
            args=(dif, size))
        return arr

    dx, dy, dz = tomo.shape

    if ind is None:
        ind = np.arange(0, dx)

    mask = np.zeros((1, dy, dz))
    for m in ind:
        tmp = filters.median_filter(tomo[m, :, :], (size, size))
        mask = ((tomo[m, :, :] - tmp) >= dif).astype(int)
        tomo[m, :, :] = tmp * mask + tomo[m, :, :] * (1 - mask)


def retrieve_phase(
        tomo, psize=1e-4, dist=50, energy=20,
        alpha=1e-4, pad=True, ind=None):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    psize : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter.
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    ind : array of int, optional
        Projection indices at which the phase retrieval is applied.

    Returns
    -------
    ndarray
        Approximated 3D tomographic phase data.
    """
    if type(tomo) == str and tomo == 'SHARED':
        tomo = mp.shared_arr
    else:
        arr = mp.distribute_jobs(
            tomo, func=retrieve_phase,
            args=(psize, dist, energy, alpha, pad), axis=0)
        return arr

    dx, dy, dz = tomo.shape
    if ind is None:
        ind = np.arange(0, dx)

    # Compute the filter.
    H, xshift, yshift, prj = _paganin_filter(
        tomo, psize, dist, energy, alpha, pad)

    for m in ind:
        proj = tomo[m, :, :]
        if pad:
            prj[xshift:dy + xshift, yshift:dz + yshift] = proj
            fproj = np.fft.fft2(prj)
            filtproj = np.multiply(H, fproj)
            tmp = np.real(np.fft.ifft2(filtproj)) / np.max(H)
            proj = tmp[xshift:dy + xshift, yshift:dz + yshift]
        elif not pad:
            fproj = np.fft.fft2(proj)
            filtproj = np.multiply(H, fproj)
            proj = np.real(np.fft.ifft2(filtproj)) / np.max(H)
        tomo[m, :, :] = proj


def _paganin_filter(tomo, psize, dist, energy, alpha, pad):
    """
    Calculate Paganin-type 2D filter to be used for phase retrieval.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    psize : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    alpha : float
        Regularization parameter.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    ndarray
        2D Paganin filter.
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    ndarray
        Padded 2D projection image.
    """
    dx, dy, dz = tomo.shape
    wavelen = 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy

    if pad:
        # Find pad values.
        val = np.mean((tomo[:, :, 0] + tomo[:, :, dz - 1]) / 2)

        # Fourier pad in powers of 2.
        padpix = np.ceil(PI * wavelen * dist / psize ** 2)

        nx = pow(2, np.ceil(np.log2(dy + padpix)))
        ny = pow(2, np.ceil(np.log2(dz + padpix)))
        xshift = int((nx - dy) / 2.)
        yshift = int((ny - dz) / 2.)

        # Template pad image.
        prj = val * np.ones((nx, ny), dtype='float32')

    elif not pad:
        nx, ny = dy, dz
        xshift, yshift, prj = None, None, None
        prj = np.ones((dy, dz), dtype='float32')

    # Sampling in reciprocal space.
    indx = (1 / ((nx - 1) * psize)) * np.arange(-(nx - 1) * 0.5, nx * 0.5)
    indy = (1 / ((ny - 1) * psize)) * np.arange(-(ny - 1) * 0.5, ny * 0.5)
    du, dv = np.meshgrid(indy, indx)
    w2 = np.square(du) + np.square(dv)

    # Filter in Fourier space.
    H = 1 / (wavelen * dist * w2 / (4 * PI) + alpha)
    H = np.fft.fftshift(H)
    return H, xshift, yshift, prj
