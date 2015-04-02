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
__all__ = ['simulate', 'art']


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


def simulate(model, theta, center=None):
    """
    Python wrapper for the simulate.c function.

    Parameters
    ----------
    model : ndarray
        Stacked sinograms as 3-D data.

    theta : 1-D array
        Projection angles.

    center : scalar
        Rotation center.

    Returns
    -------
    data : ndarray
        Simulated 3-D data
    """
    nslice, ngridx, ngridy = model.shape
    nproj = theta.size
    npixel = np.ceil(np.sqrt(ngridx*ngridx+ngridy*ngridy)).astype('int')
    if center is None:
        center = npixel/2.0
    data = np.zeros((nproj, nslice, npixel), dtype='float32')

    # Make sure that inputs datatypes are correct
    if not isinstance(model, np.float32):
        model = np.array(model, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')

    # Call C function to reconstruct recon matrix.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.simulate.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.simulate(
        model.ctypes.data_as(c_float_p),
        theta.ctypes.data_as(c_float_p),
        ctypes.c_float(center),
        ctypes.c_int(nproj),
        ctypes.c_int(nslice),
        ctypes.c_int(npixel),
        ctypes.c_int(ngridx),
        ctypes.c_int(ngridy),
        data.ctypes.data_as(c_float_p))
    return data


class ReconCStruct(ctypes.Structure):
    """
    Reconstruction parameter structure.
    """
    _fields_ = [("niter", ctypes.c_int),
                ("beta", ctypes.c_float),
                ("delta", ctypes.c_float),
                ("center", ctypes.c_float),
                ("nproj", ctypes.c_int),
                ("nslice", ctypes.c_int),
                ("npixel", ctypes.c_int),
                ("ngridx", ctypes.c_int),
                ("ngridy", ctypes.c_int),
                ("isubset", ctypes.c_float),
                ("nsubset", ctypes.c_int),
                ("emission", ctypes.c_bool)]


def _init_recon(
        data, theta,
        niter=1, beta=1, delta=1, center=None,
        nproj=None, nslice=None, npixel=None,
        ngridx=None, ngridy=None, emission=None,
        isubset=None, nsubset=1, recon=None):
    """
    Initialize reconstruction parameters
    """
    nproj, nslice, npixel = data.shape
    if center is None:
        center = npixel/2.
    if isubset is None:
        # isubset = np.arange(0, nproj)
        isubset = 1.
    if ngridx is None:
        ngridx = npixel
    if ngridy is None:
        ngridy = npixel
    if emission is None:
        emission = True
    if recon is None:
        recon = 1e-5*np.ones(
            (nslice, ngridx, ngridy),
            dtype='float32')

    rargs = ReconCStruct()
    rargs.niter = niter
    rargs.beta = beta
    rargs.delta = delta
    rargs.center = center
    rargs.nproj = nproj
    rargs.nslice = nslice
    rargs.npixel = npixel
    rargs.ngridx = ngridx
    rargs.ngridy = ngridy
    rargs.isubset = isubset
    rargs.nsubset = nsubset
    rargs.emission = emission

    # Make sure that inputs datatypes are correct
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32')
    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype='float32')
    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')

    return data, theta, rargs, recon


def art(*args, **kwargs):
    """
    Algebraic reconstruction technique.
    """
    data, theta, rargs, recon = _init_recon(*args, **kwargs)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.art.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.art(
        data.ctypes.data_as(c_float_p),
        theta.ctypes.data_as(c_float_p),
        ctypes.byref(rargs),
        recon.ctypes.data_as(c_float_p))
    return recon


def reg_term(
        model, beta=1, delta=1, ngridx=None,
        ngridy=None, reg=None):
    nslice = np.array(model.shape[0], dtype='int32')
    ngridx = np.array(model.shape[1], dtype='int32')
    ngridy = np.array(model.shape[2], dtype='int32')
    msize = (nslice, ngridx, ngridy)
    reg = np.zeros(msize, dtype='float32')

    c_float_p = ctypes.POINTER(ctypes.c_float)
    libtomopy_recon.reg_term.restype = ctypes.POINTER(ctypes.c_void_p)
    libtomopy_recon.reg_term(model.ctypes.data_as(
        c_float_p),
        ctypes.c_int(nslice),
        ctypes.c_int(ngridx),
        ctypes.c_int(ngridy),
        ctypes.c_float(beta),
        ctypes.c_float(delta),
        reg.ctypes.data_as(c_float_p))
    return reg
