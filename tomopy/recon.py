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


def import_lib(lname):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            lname = 'lib/' + lname + '.pyd'
            libpath = os.path.join(os.path.dirname(__file__), lname)
            return ctypes.CDLL(os.path.abspath(libpath))
        else:
            lname = 'lib/' + lname + '.so'
            libpath = os.path.join(os.path.dirname(__file__), lname)
            return ctypes.CDLL(os.path.abspath(libpath))
    except OSError as e:
        pass

libtomopy_recon = import_lib('libtomopy_recon')


def simulate(obj, theta, center=None):
    """
    Simulate parallel projections of a given 3D object.

    Parameters
    ----------
    obj : 3D array (int or float)
        Voxelized object.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    Returns
    -------
    data : 3D array (float)
        Simulated tomographic data.
    """
    # Estimate data dimensions.
    ox, oy, oz = obj.shape
    dx = theta.size
    dy = ox
    dz = np.ceil(np.sqrt(oy * oy + oz * oz)).astype('int')
    if center is None:
        center = dz / 2.

    # Make sure that inputs datatypes are correct.
    if not isinstance(obj, np.float32):
        obj = np.array(obj, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
    data = np.zeros((dx, dy, dz), dtype='float32')

    # Call C function to reconstruct recon matrix.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.simulate.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.simulate(
        obj.ctypes.data_as(c_float_p),
        ctypes.c_int(ox),
        ctypes.c_int(oy),
        ctypes.c_int(oz),
        data.ctypes.data_as(c_float_p),
        ctypes.c_int(dx),
        ctypes.c_int(dy),
        ctypes.c_int(dz),
        ctypes.c_float(center),
        theta.ctypes.data_as(c_float_p))
    return data


def gridrec(
        data, theta, center=None,
        num_gridx=None, num_gridy=None,
        filter_name='shepp'):
    """
    Reconstruct object from projection data using gridrec algorithm.

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    filter_name : string
        Filter name for weighting. 'shepp', 'hann', 'hamming', 'ramlak',
        or 'none'.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    # Gridrec reconstructs 2 slices minimum.
    flag = False
    if data.shape[1] == 1:
        flag = True
        data = np.append(data, data, 1)

    dx, dy, dz = data.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.gridrec.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.gridrec(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1):
    """
    Reconstruct object from projection data using algebraic reconstruction
    technique (ART).

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.art.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.art(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using block algebraic
    reconstruction technique (BART).

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    num_block : scalar (int)
        Number of data blocks for intermediate updating the object.

    ind_block : 1D array (int)
        Order of projections to be used for updating.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
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
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.bart.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.bart(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None):
    """
    Reconstruct object from projection data using filtered back
    projection (FBP).

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.fbp.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.fbp(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1):
    """
    Reconstruct object from projection data using maximum-likelihood
    expectation-maximization algorithm. (ML-EM).

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.mlem.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.mlem(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using ordered-subset
    expectation-maximization (OS-EM).

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    num_block : scalar (int)
        Number of data blocks for intermediate updating the object.

    ind_block : 1D array (int)
        Order of projections to be used for updating.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
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
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.osem.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.osem(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None, num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using ordered-subset
    penalized maximum likelihood algorithm with weighted linear and
    quadratic penalties.

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    reg_par : array with 2 elements (float)
        Regularization hyperparameters as an array, (beta, delta).

    num_block : scalar (int)
        Number of data blocks for intermediate updating the object.

    ind_block : 1D array (int)
        Order of projections to be used for updating.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
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
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.ospml_hybrid.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.ospml_hybrid(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None, num_block=1, ind_block=None):
    """
    Reconstruct object from projection data using ordered-subset
    penalized maximum likelihood algorithm with quadratic penalty.

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    reg_par : scalar (float)
        Regularization parameter for smoothing.

    num_block : scalar (int)
        Number of data blocks for intermediate updating the object.

    ind_block : 1D array (int)
        Order of projections to be used for updating.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
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
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.ospml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.ospml_quad(
        data.ctypes.data_as(c_float_p),
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
        reg_par.ctypes.data_as(c_float_p),
        ctypes.c_int(num_block),
        ind_block.ctypes.data_as(c_float_p))
    return recon


def pml_hybrid(
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None):
    """
    Reconstruct object from projection data using penalized maximum
    likelihood algorithm with weighted linear and quadratic penalties.

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    reg_par : array with 2 elements (float)
        Regularization hyperparameters as an array, (beta, delta).

    num_block : scalar (int)
        Number of data blocks for intermediate updating the object.

    ind_block : 1D array (int)
        Order of projections to be used for updating.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
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
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.pml_hybrid.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.pml_hybrid(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1,
        reg_par=None):
    """
    Reconstruct object from projection data using penalized maximum
    likelihood algorithm with quadratic penalty.

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    reg_par : scalar (float)
        Regularization parameter for smoothing.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
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
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.pml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.pml_quad(
        data.ctypes.data_as(c_float_p),
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
        data, theta, center=None, emission=False,
        recon=None, num_gridx=None, num_gridy=None, num_iter=1):
    """
    Reconstruct object from projection data using simultaneous
    iterative reconstruction technique (SIRT).

    Parameters
    ----------
    data : 3D array (float)
        Tomographic data.

    theta : 1D array (float)
        Projection angles in radian.

    center : scalar (float)
        Location of rotation axis.

    emission : bool
        Determines whether data is emission or transmission type.

    recon : 3D array (float)
        Initial values of the reconstruction object.

    num_gridx, num_gridy : scalar (int)
        Number of pixels along x- and y-axes in the reconstruction grid.

    num_iter : scalar (int)
        Number of algorithm iterations performed.

    Returns
    -------
    recon : 3D array (float32)
        Reconstructed object.
    """
    dx, dy, dz = data.shape
    if center is None:
        center = dz / 2.
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if recon is None:
        recon = 1e-6 * np.ones((dy, num_gridx, num_gridy), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
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
    libtomopy_recon.sirt.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.sirt(
        data.ctypes.data_as(c_float_p),
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
