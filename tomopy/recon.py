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
Reconstruction module.

:Author: Doga Gursoy
:Organization: Argonne National Laboratory

"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np
import ctypes
import os
import logging


__docformat__ = 'restructuredtext en'
__all__ = ['simulate',
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


# Get the path and import the C-shared library.
try:
    if os.name == 'nt':
        libpath = os.path.join(
            os.path.dirname(__file__), 'lib/libtomopy_recon.pyd')
        libtomopy_recon = ctypes.CDLL(os.path.abspath(libpath))
    else:
        libpath = os.path.join(
            os.path.dirname(__file__), 'lib/libtomopy_recon.so')
        libtomopy_recon = ctypes.CDLL(os.path.abspath(libpath))
except OSError as e:
    pass


class ObjectStruct(ctypes.Structure):

    """
    Data parameter structure.
    """
    _fields_ = [("ox", ctypes.c_int),
                ("oy", ctypes.c_int),
                ("oz", ctypes.c_int)]


class DataStruct(ctypes.Structure):

    """
    Data parameter structure.
    """
    _fields_ = [("dx", ctypes.c_int),
                ("dy", ctypes.c_int),
                ("dz", ctypes.c_int),
                ("center", ctypes.c_float),
                ("proj_angle", ctypes.POINTER(ctypes.c_float))]


class ReconStruct(ctypes.Structure):

    """
    Reconstruction parameter structure.
    """
    _fields_ = [("num_iter", ctypes.c_int),
                ("reg_par", ctypes.POINTER(ctypes.c_float)),
                ("rx", ctypes.c_int),
                ("ry", ctypes.c_int),
                ("rz", ctypes.c_int),
                ("ind_block", ctypes.POINTER(ctypes.c_float)),
                ("num_block", ctypes.c_int)]


def simulate(obj, theta, center=None):
    """
    Simulates projection data for a given 3-D object.

    Parameters
    ----------
    object : ndarray
        Voxelized 3-D object.

    theta : 1-D array
        Projection angles.

    center : scalar
        Rotation center.

    Returns
    -------
    data : ndarray
        Simulated 3-D projection data
    """
    ox, oy, oz = obj.shape
    dx = theta.size
    dy = ox
    dz = np.ceil(np.sqrt(oy * oy + oz * oz)).astype('int')
    if center is None:
        center = dz / 2.0
    data = np.zeros((dx, dy, dz), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(obj, np.float32):
        obj = np.array(obj, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')

    # Initialize struct for object parameteres
    c_float_p = ctypes.POINTER(ctypes.c_float)
    opars = ObjectStruct()
    opars.ox = ox
    opars.oy = oy
    opars.oz = oz

    # Initialize struct for data parameteres
    c_float_p = ctypes.POINTER(ctypes.c_float)
    dpars = DataStruct()
    dpars.dx = dx
    dpars.dy = dy
    dpars.dz = dz
    dpars.center = center
    dpars.proj_angle = theta.ctypes.data_as(c_float_p)

    # Call C function to reconstruct recon matrix.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.simulate.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.simulate(
        obj.ctypes.data_as(c_float_p), ctypes.byref(opars),
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars))
    return data


def _init_recon(
        data, theta, num_iter=1, reg_par=None, center=None,
        num_gridx=None, num_gridy=None, emission=None,
        ind_block=None, num_block=1, recon=None):
    """
    Initialize reconstruction parameters
    """
    dx, dy, dz = data.shape
    if center is None:
        center = dz / 2.
    if ind_block is None:
        ind_block = np.arange(0, dx).astype("float32")
    if num_gridx is None:
        num_gridx = dz
    if num_gridy is None:
        num_gridy = dz
    if reg_par is None:
        reg_par = np.ones(10, dtype="float32")
    if emission is None:
        emission = True
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
    if not isinstance(reg_par, np.float32):
        reg_par = np.array(reg_par, dtype='float32')
    if not isinstance(ind_block, np.float32):
        ind_block = np.array(ind_block, dtype='float32')

    # Initialize struct for data parameteres
    c_float_p = ctypes.POINTER(ctypes.c_float)
    dpars = DataStruct()
    dpars.dx = dx
    dpars.dy = dy
    dpars.dz = dz
    dpars.center = center
    dpars.proj_angle = theta.ctypes.data_as(c_float_p)

    # Initialize struct for reconstruction parameteres
    rpars = ReconStruct()
    rpars.num_iter = num_iter
    rpars.reg_par = reg_par.ctypes.data_as(c_float_p)
    rpars.rx = dy
    rpars.ry = num_gridx
    rpars.rz = num_gridy
    rpars.ind_block = ind_block.ctypes.data_as(c_float_p)
    rpars.num_block = num_block

    return data, dpars, recon, rpars


def art(*args, **kwargs):
    """
    Algebraic reconstruction technique.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.art.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.art(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def bart(*args, **kwargs):
    """
    Block algebraic reconstruction technique.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.bart.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.bart(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def fbp(*args, **kwargs):
    """
    Filtered backprojection.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.fbp.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.fbp(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def mlem(*args, **kwargs):
    """
    Maximum-likelihood expectation-maximization.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.mlem.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.mlem(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def osem(*args, **kwargs):
    """
    Ordered-subset expectation-maximization.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.osem.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.osem(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def ospml_hybrid(*args, **kwargs):
    """
    Ordered-subset penalized maximum likelihood with weighted linear
    and quadratic penalties.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.ospml_hybrid.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.ospml_hybrid(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def ospml_quad(*args, **kwargs):
    """
    Ordered-subset penalized maximum likelihood with quadratic penalty.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.ospml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.ospml_quad(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def pml_hybrid(*args, **kwargs):
    """
    Penalized maximum likelihood with weighted linear and quadratic
    penalties.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.pml_hybrid.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.pml_hybrid(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def pml_quad(*args, **kwargs):
    """
    Penalized maximum likelihood with quadratic penalty.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.pml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.pml_quad(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon


def sirt(*args, **kwargs):
    """
    Simultaneous iterative reconstruction technique.
    """
    data, dpars, recon, rpars = _init_recon(*args, **kwargs)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.sirt.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.sirt(
        data.ctypes.data_as(c_float_p), ctypes.byref(dpars),
        recon.ctypes.data_as(c_float_p), ctypes.byref(rpars))
    return recon
