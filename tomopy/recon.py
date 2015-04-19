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

import warnings
import numpy as np
import ctypes
import os
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'simulate',
    'gridrec',
    'art',
    'bart',
    'fbp',
    'mlem',
    'osem',
    'ospml_hybrid',
    'ospml_quad',
    'pml_hybrid',
    'pml_quad',
    'sirt']


def _import_shared_lib(lib_name):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            libpath = os.path.join('lib', lib_name + '.pyd')
            return ctypes.CDLL(os.path.abspath(libpath))
        else:
            libpath = os.path.join('lib', lib_name + '.so')
            return ctypes.CDLL(os.path.abspath(libpath))
    except OSError as e:
        logger.warning('OSError: Shared library missing.')


LIB_TOMOPY = _import_shared_lib('libtomopy')


def simulate(obj, theta, center=None):
    """
    Simulate parallel projections of a given 3D object.

    Parameters
    ----------
    obj : ndarray
        Voxelized 3D object.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.

    Returns
    -------
    ndarray
        Simulated 3D tomographic data.
    """
    # Estimate data dimensions.
    ox, oy, oz = obj.shape
    dx = len(theta)
    dy = ox
    dz = np.ceil(np.sqrt(oy * oy + oz * oz)).astype('int')
    tomo = np.zeros((dx, dy, dz), dtype='float32')
    if center is None:
        center = dz / 2.

    # Make sure that inputs datatypes are correct.
    if not isinstance(obj, np.float32):
        obj = np.array(obj, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')

    # Call C function to reconstruct recon matrix.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.simulate.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.simulate(
        obj.ctypes.data_as(c_float_p),
        ctypes.c_int(ox),
        ctypes.c_int(oy),
        ctypes.c_int(oz),
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p))
    return tomo


def gridrec(
        tomo, theta, center=None,
        num_gridx=None, num_gridy=None,
        filter_name='shepp'):
    """
    Reconstruct object from projection data using gridrec algorithm
    :cite:`Dowd:99`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.

    filter_name : str, optional
        Filter name for weighting. 'shepp', 'hann', 'hamming', 'ramlak',
        or 'none'.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    # Gridrec reconstructs 2 slices minimum.
    flag = False
    if tomo.shape[1] == 1:
        flag = True
        tomo = np.append(tomo, tomo, 1)

    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    filter_name = np.array(filter_name, dtype=(str, 16))

    c_char_p = ctypes.POINTER(ctypes.c_char)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.gridrec.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.gridrec(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        filter_name.ctypes.data_as(c_char_p))

    # Dump second slice.
    if flag is True:
        recon = recon[0:1]
    return recon


def art(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1):
    """
    Reconstruct object from projection data using algebraic reconstruction
    technique (ART) :cite:`Kak:98`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.art.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.art(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter))
    return recon


def bart(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using block algebraic
    reconstruction technique (BART).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')
    if ind_block is None:
        ind_block = np.arange(0, dx).astype("float32")

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')
    if not isinstance(num_block, np.int32):
        num_block = np.array(num_block, dtype='int32')
    if not isinstance(ind_block, np.float32):
        ind_block = np.array(ind_block, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.bart.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.bart(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter),
        ctypes.c_int(num_block),
        ind_block.ctypes.data_as(c_float_p))
    return recon


def fbp(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None):
    """
    Reconstruct object from projection data using filtered back
    projection (FBP).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.fbp.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.fbp(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy))
    return recon


def mlem(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1):
    """
    Reconstruct object from projection data using maximum-likelihood
    expectation-maximization algorithm. (ML-EM) :cite:`Dempster:77`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.mlem.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.mlem(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter))
    return recon


def osem(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using ordered-subset
    expectation-maximization (OS-EM) :cite:`Hudson:94`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')
    if ind_block is None:
        ind_block = np.arange(0, dx).astype("float32")

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')
    if not isinstance(num_block, np.int32):
        num_block = np.array(num_block, dtype='int32')
    if not isinstance(ind_block, np.float32):
        ind_block = np.array(ind_block, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.osem.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.osem(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter),
        ctypes.c_int(num_block),
        ind_block.ctypes.data_as(c_float_p))
    return recon


def ospml_hybrid(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None, num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using ordered-subset
    penalized maximum likelihood algorithm with weighted linear and
    quadratic penalties :cite:`Gursoy:15`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.
    reg_par : list, optional
        Regularization hyperparameters as an array, (beta, delta).
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')
    if reg_par is None:
        reg_par = np.ones(10, dtype="float32")
    if ind_block is None:
        ind_block = np.arange(0, dx).astype("float32")

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')
    if not isinstance(reg_par, np.float32):
        reg_par = np.array(reg_par, dtype='float32')
    if not isinstance(num_block, np.int32):
        num_block = np.array(num_block, dtype='int32')
    if not isinstance(ind_block, np.float32):
        ind_block = np.array(ind_block, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.ospml_hybrid.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.ospml_hybrid(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter),
        reg_par.ctypes.data_as(c_float_p),
        ctypes.c_int(num_block),
        ind_block.ctypes.data_as(c_float_p))
    return recon


def ospml_quad(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None, num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using ordered-subset
    penalized maximum likelihood algorithm with quadratic penalty.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.
    reg_par : float, optional
        Regularization parameter for smoothing.
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')
    if reg_par is None:
        reg_par = np.ones(10, dtype="float32")
    if ind_block is None:
        ind_block = np.arange(0, dx).astype("float32")

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')
    if not isinstance(reg_par, np.float32):
        reg_par = np.array(reg_par, dtype='float32')
    if not isinstance(num_block, np.int32):
        num_block = np.array(num_block, dtype='int32')
    if not isinstance(ind_block, np.float32):
        ind_block = np.array(ind_block, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.ospml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.ospml_quad(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter),
        reg_par.ctypes.data_as(c_float_p),
        ctypes.c_int(num_block),
        ind_block.ctypes.data_as(c_float_p))
    return recon


def pml_hybrid(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None):
    """
    Reconstruct object from projection data using penalized maximum
    likelihood algorithm with weighted linear and quadratic penalties
    :cite:`Chang:04`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.
    reg_par : list, optional
        Regularization hyperparameters as an array, (beta, delta).
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')
    if reg_par is None:
        reg_par = np.ones(10, dtype="float32")

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')
    if not isinstance(reg_par, np.float32):
        reg_par = np.array(reg_par, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.pml_hybrid.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.pml_hybrid(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter),
        reg_par.ctypes.data_as(c_float_p))
    return recon


def pml_quad(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None):
    """
    Reconstruct object from projection data using penalized maximum
    likelihood algorithm with quadratic penalty.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.
    reg_par : float, optional
        Regularization parameter for smoothing.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')
    if reg_par is None:
        reg_par = np.ones(10, dtype="float32")

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')
    if not isinstance(reg_par, np.float32):
        reg_par = np.array(reg_par, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.pml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.pml_quad(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter),
        reg_par.ctypes.data_as(c_float_p))
    return recon


def sirt(
        tomo, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1):
    """
    Reconstruct object from projection data using simultaneous
    iterative reconstruction technique (SIRT).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center : float, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    recon : ndarray, optional
        Initial values of the reconstruction object.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    num_iter : int, optional
        Number of algorithm iterations performed.

    Returns
    -------
    ndarray
        Reconstructed 3D object.
    """
    dx, dy, dz = tomo.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(tomo, np.float32):
        tomo = np.array(tomo, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    if not isinstance(recon, np.float32):
        recon = np.array(recon, dtype='float32')
    if not isinstance(num_gridx, np.int32):
        num_gridx = np.array(num_gridx, dtype='int32')
    if not isinstance(num_gridy, np.int32):
        num_gridy = np.array(num_gridy, dtype='int32')
    if not isinstance(num_iter, np.int32):
        num_iter = np.array(num_iter, dtype='int32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    LIB_TOMOPY.sirt.restype = ctypes.POINTER(ctypes.c_void_p)
    LIB_TOMOPY.sirt(
        tomo.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p),
        recon.ctypes.data_as(c_float_p),
        ctypes.c_int(num_gridx),
        ctypes.c_int(num_gridy),
        ctypes.c_int(num_iter))
    return recon
