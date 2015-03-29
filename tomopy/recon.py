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
__all__ = ['theta', 'simulate', 'reconstruct']


# Get the path and import the C-shared library.
try:
    if os.name == 'nt':
        libpath = os.path.join(
            os.path.dirname(__file__), 'lib/libtomopy_recon.pyd')
        libtg = ctypes.CDLL(os.path.abspath(libpath))
    else:
        libpath = os.path.join(
            os.path.dirname(__file__), 'lib/libtomopy_recon.so')
        libtg = ctypes.CDLL(os.path.abspath(libpath))
except OSError as e:
    pass


def theta(
        nprojs=180, theta_start=0,
        theta_end=180, degrees=False):
    """
    Projection angles.

    Parameters
    ----------
    nprojs : scalar
        Number of projections.

    theta_start : scalar
        Projection angle of the first projection.

    theta_end : scalar
        Projection angle of the last projection.

    degrees : bool
        If True, the unit is in degrees,
        otherwise in radians.

    Returns
    -------
    theta : 1-D array (float32)
        Projection angles.
    """
    if degrees:
        theta = np.linspace(
            theta_start, theta_end,
            nprojs, dtype='float32')
    else:
        scl = np.pi/180.
        theta = np.linspace(
            theta_start*scl, theta_end*scl,
            nprojs, dtype='float32')
    return theta


def simulate(model, theta, center=None):
    """
    Python wrapper for the simulate.c function.

    Parameters
    ----------
    model : 3-D array
        1st dim: Slices
        2nd dim: Rows
        3rd dim: Columns

    theta : 1-D array
        Projection angles.

    center : scalar
        Rotation center.

    Returns
    -------
    simdata : simulated 3-D data
        1st dim: Projections
        2nd dim: Slices
        3rd dim: Pixels
    """
    num_slices = np.array(model.shape[0], dtype='int32')
    num_grids_x = np.array(model.shape[1], dtype='int32')
    num_grids_y = np.array(model.shape[2], dtype='int32')

    # Init final simdata matrix.
    nprojs = np.array(theta.size, dtype='int32')
    tmp = np.ceil(np.sqrt(num_grids_x*num_grids_x+num_grids_y*num_grids_y))
    num_pixels = np.array(tmp, dtype='int32')
    if center is None:
        center = num_pixels/2.0
    msize = (nprojs, num_slices, num_pixels)
    simdata = np.zeros(msize, dtype='float32')

    # Make sure that inputs datatypes are correct
    model = np.array(model, dtype='float32')
    theta = np.array(theta, dtype='float32')
    center = np.array(center, dtype='float32')

    # Call C function to reconstruct recon matrix.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtg.simulate.restype = ctypes.POINTER(ctypes.c_void_p)
    libtg.simulate(
        model.ctypes.data_as(c_float_p),
        theta.ctypes.data_as(c_float_p),
        ctypes.c_float(center),
        ctypes.c_int(nprojs),
        ctypes.c_int(num_slices),
        ctypes.c_int(num_pixels),
        ctypes.c_int(num_grids_x),
        ctypes.c_int(num_grids_y),
        simdata.ctypes.data_as(c_float_p))
    return simdata


# Reconstruction

def reconstruct(
        data, theta,
        method='art', num_iters=1, beta=1, delta=1, center=None,
        num_grids_x=None, num_grids_y=None,
        emission=True, subset_ind=None, num_subset=1, recon=None):

    nprojs = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    if center is None:
        center = num_pixels/2.

    subset_ind = np.arange(0, theta.shape[0], 1).astype('int32')

    # Init recon matrix.
    if num_grids_x is None:
        num_grids_x = num_pixels
    if num_grids_y is None:
        num_grids_y = num_pixels
    msize = (num_slices, num_grids_x, num_grids_y)
    recon = 1e-5*np.ones(msize, dtype='float32')

    # Make sure that inputs datatypes are correct
    data = np.array(data, dtype='float32')
    theta = np.array(theta, dtype='float32')
    center = np.array(center, dtype='float32')

    if method.lower() == 'art':
        # Call C function to reconstruct recon matrix.
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.art.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.art(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'bart':
        # Call C function to reconstruct recon matrix.
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.bart.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.bart(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            ctypes.c_int(num_subset),
            subset_ind.ctypes.data_as(c_int_p),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'sirt':
        # Call C function to reconstruct recon matrix.
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.sirt.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.sirt(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'mlem':
        # Call C function to reconstruct recon matrix.
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.mlem.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.mlem(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'osem':
        # Call C function to reconstruct recon matrix.
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.osem.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.osem(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            ctypes.c_int(num_subset),
            subset_ind.ctypes.data_as(c_int_p),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'pml':
        # Call C function to reconstruct recon matrix.
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.pml.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.pml(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_float(beta),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'ospml':
        # Call C function to reconstruct recon matrix.
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.ospml.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.ospml(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_float(beta),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            ctypes.c_int(num_subset),
            subset_ind.ctypes.data_as(c_int_p),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'ospmlh':
        # Call C function to reconstruct recon matrix.
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.ospmlh.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.ospmlh(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_float(beta),
            ctypes.c_float(delta),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            ctypes.c_int(num_iters),
            ctypes.c_int(num_subset),
            subset_ind.ctypes.data_as(c_int_p),
            recon.ctypes.data_as(c_float_p))

    if method.lower() == 'fbp':
        # Call C function to reconstruct recon matrix.
        c_float_p = ctypes.POINTER(ctypes.c_float)
        libtg.fbp.restype = ctypes.POINTER(ctypes.c_void_p)
        libtg.fbp(
            data.ctypes.data_as(c_float_p),
            theta.ctypes.data_as(c_float_p),
            ctypes.c_float(center),
            ctypes.c_int(nprojs),
            ctypes.c_int(num_slices),
            ctypes.c_int(num_pixels),
            ctypes.c_int(num_grids_x),
            ctypes.c_int(num_grids_y),
            recon.ctypes.data_as(c_float_p))

    return recon


def reg_term(
        model, beta=1, delta=1, num_grids_x=None,
        num_grids_y=None, reg=None):
    num_slices = np.array(model.shape[0], dtype='int32')
    num_grids_x = np.array(model.shape[1], dtype='int32')
    num_grids_y = np.array(model.shape[2], dtype='int32')
    msize = (num_slices, num_grids_x, num_grids_y)
    reg = np.zeros(msize, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtg.reg_term.restype = ctypes.POINTER(ctypes.c_void_p)
    libtg.reg_term(model.ctypes.data_as(
        c_float_p),
        ctypes.c_int(num_slices),
        ctypes.c_int(num_grids_x),
        ctypes.c_int(num_grids_y),
        ctypes.c_float(beta),
        ctypes.c_float(delta),
        reg.ctypes.data_as(c_float_p))
    return reg
