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
Module for reconstruction tasks.
"""

from __future__ import absolute_import, division, print_function

from skimage import io as sio
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy import ndimage
import shutil
from tomopy.io.data import Writer
from tomopy.misc.mask import circ_mask
import tomopy.misc.mproc as mp
import tomopy.extern as ext
from tomopy.util import *
import multiprocessing
import ctypes
import os
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Recon',
           'find_center',
           'write_center']


LIB_TOMOPY = ext.import_shared_lib('libtomopy')


class Recon():

    """
    Class for reconstruction methods.

    Attributes
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center: array, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.
    """

    def __init__(
            self, tomo, theta, center=None, emission=True,
            num_gridx=None, num_gridy=None, ncore=None, nchunk=None):
        """
        Initialize reconstruction parameters.
        """
        self.tomo = as_float32(tomo)
        self.dx, self.dy, self.dz = self.tomo.shape
        self.theta = as_float32(theta)
        self.center = as_float32(center)
        self.emission = emission
        self.ncore = ncore
        self.nchunk = nchunk

        if center is None:
            self.center = np.ones(self.dy, dtype='float32') * self.dz / 2.
        elif np.array(center).size == 1:
            self.center = np.ones(self.dy, dtype='float32') * self.center
        if num_gridx is None:
            self.num_gridx = as_int32(self.dz)
        if num_gridy is None:
            self.num_gridy = as_int32(self.dz)
        if not emission:
            self.tomo = -np.log(self.tomo)

    def _init_recon(self, recon=None):
        """
        Create reconstruction grid.
        """
        if recon is None:
            self.recon = 1e-6 * np.ones(
                (self.dy, self.num_gridx, self.num_gridy),
                dtype='float32')
        else:
            self.recon = as_float32(recon)

    def art(self, num_iter=1, recon=None):
        """
        Reconstruct object from projection data using algebraic reconstruction
        technique (ART) :cite:`Kak:98`.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._art,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def bart(self, num_iter=1, recon=None, num_block=1, ind_block=None):
        """
        Reconstruct object from projection data using block algebraic
        reconstruction technique (BART).

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.
        num_block : int, optional
            Number of data blocks for intermediate updating the object.
        ind_block : array of int, optional

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_block = as_int32(num_block)
        if ind_block is None:
            self.ind_block = np.arange(0, self.dx).astype('float32')
        else:
            self.ind_block = as_float32(ind_block)
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._bart,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter,
                self.num_block, self.ind_block),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def fbp(self, filter_name='shepp'):
        """
        Reconstruct object from projection data using filtered back
        projection (FBP).

        Warning
        -------
        Filter not implemented yet.

        Parameters
        ----------
        filter_name : str, optional
            Filter name for weighting. 'shepp', 'hann', 'hamming', 'ramlak',
            'cosine' or 'none'.

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.filter_name = np.array(filter_name, dtype=(str, 16))
        self._init_recon()
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._fbp,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.filter_name),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def gridrec(self, filter_name='shepp'):
        """
        Reconstruct object from projection data using gridrec algorithm
        :cite:`Dowd:99`.

        Parameters
        ----------
        filter_name : str, optional
            Filter name for weighting. 'shepp', 'hann', 'hamming', 'ramlak',
            'cosine' or 'none'.

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        # Gridrec accepts even number of slices.
        is_odd = False
        if self.dy % 2 != 0:
            is_odd = True
            lasttomo = np.expand_dims(self.tomo[:, -1, :], 1)
            self.tomo = np.append(self.tomo, lasttomo, 1)
            self.dy += 1

        # Chunk size can't be smaller than two for gridrec.
        if self.ncore is None:
            self.ncore = multiprocessing.cpu_count()
        if self.dx < self.ncore:
            self.ncore = self.dx
        if self.nchunk is None:
            self.nchunk = (self.dy - 1) // self.ncore + 1
        if self.nchunk < 2:
            self.nchunk = 2

        self.filter_name = np.array(filter_name, dtype=(str, 16))

        self._init_recon()
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._gridrec,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.filter_name),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)

        # Dump last slice if original number of sice was even.
        if is_odd:
            arr = arr[0:-1, :, :]
        return arr

    def mlem(self, num_iter=1, recon=None):
        """
        Reconstruct object from projection data using maximum-likelihood
        expectation-maximization algorithm. (ML-EM) :cite:`Dempster:77`.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._mlem,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def osem(self, num_iter=1, recon=None, num_block=1, ind_block=None):
        """
        Reconstruct object from projection data using ordered-subset
        expectation-maximization (OS-EM) :cite:`Hudson:94`.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.
        num_block : int, optional
            Number of data blocks for intermediate updating the object.
        ind_block : array of int, optional

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_block = as_int32(num_block)
        if ind_block is None:
            self.ind_block = np.arange(0, self.dx).astype('float32')
        else:
            self.ind_block = as_float32(ind_block)
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._osem,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter,
                self.num_block, self.ind_block),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def ospml_hybrid(
            self, num_iter=1, recon=None, reg_par=None,
            num_block=1, ind_block=None):
        """
        Reconstruct object from projection data using ordered-subset
        penalized maximum likelihood algorithm with weighted linear and
        quadratic penalties.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.
        reg_par : list, optional
            Regularization hyperparameters as an array, (beta, delta).
        num_block : int, optional
            Number of data blocks for intermediate updating the object.
        ind_block : array of int, optional

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_block = as_int32(num_block)
        if reg_par is None:
            self.reg_par = np.ones(10, dtype='float32')
        else:
            self.reg_par = as_float32(reg_par)
        if ind_block is None:
            self.ind_block = np.arange(0, self.dx).astype('float32')
        else:
            self.ind_block = as_float32(ind_block)
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._ospml_hybrid,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter,
                self.reg_par, self.num_block, self.ind_block),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def ospml_quad(
            self, num_iter=1, recon=None, reg_par=None,
            num_block=1, ind_block=None):
        """
        Reconstruct object from projection data using ordered-subset
        penalized maximum likelihood algorithm with quadratic penalty.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.
        reg_par : list, optional
            Regularization hyperparameters as an array, (beta, delta).
        num_block : int, optional
            Number of data blocks for intermediate updating the object.
        ind_block : array of int, optional

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_block = as_int32(num_block)
        if reg_par is None:
            self.reg_par = np.ones(10, dtype='float32')
        else:
            self.reg_par = as_float32(reg_par)
        if ind_block is None:
            self.ind_block = np.arange(0, self.dx).astype('float32')
        else:
            self.ind_block = as_float32(ind_block)
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._ospml_quad,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter,
                self.reg_par, self.num_block, self.ind_block),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def pml_hybrid(
            self, num_iter=1, recon=None, reg_par=None):
        """
        Reconstruct object from projection data using penalized maximum
        likelihood algorithm with weighted linear and quadratic penalties
        :cite:`Chang:04`.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.
        reg_par : list, optional
            Regularization hyperparameters as an array, (beta, delta).
        num_block : int, optional
            Number of data blocks for intermediate updating the object.
        ind_block : array of int, optional

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        if reg_par is None:
            self.reg_par = np.ones(10, dtype='float32')
        else:
            self.reg_par = as_float32(reg_par)
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._pml_hybrid,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter, self.reg_par),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def pml_quad(
            self, num_iter=1, recon=None, reg_par=None):
        """
        Reconstruct object from projection data using penalized maximum
        likelihood algorithm with quadratic penalty.

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.
        reg_par : list, optional
            Regularization hyperparameters as an array, (beta, delta).
        num_block : int, optional
            Number of data blocks for intermediate updating the object.
        ind_block : array of int, optional

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        if reg_par is None:
            self.reg_par = np.ones(10, dtype='float32')
        else:
            self.reg_par = as_float32(reg_par)
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._pml_quad,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter, self.reg_par),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr

    def sirt(self, num_iter=1, recon=None):
        """
        Reconstruct object from projection data using simultaneous
        iterative reconstruction technique (SIRT).

        Parameters
        ----------
        num_iter : int, optional
            Number of algorithm iterations performed.
        recon : ndarray, optional
            Initial values of the reconstruction object.

        Returns
        -------
        ndarray
            Reconstructed 3D object.
        """
        self.num_iter = as_int32(num_iter)
        self._init_recon(recon)
        mp.init_tomo(self.tomo)
        arr = mp.distribute_jobs(
            self.recon,
            func=ext._sirt,
            args=(
                self.dx, self.dy, self.dz, self.theta, self.center,
                self.num_gridx, self.num_gridy, self.num_iter),
            axis=0,
            ncore=self.ncore,
            nchunk=self.nchunk)
        return arr


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
