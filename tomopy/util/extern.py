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
Module for external library wrappers.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import ctypes
import numpy as np
import glob
import tomopy.util.dtype as dtype
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['c_shared_lib',
           'c_project',
           'c_normalize_bg',
           'c_remove_stripe_sf',
           'c_sample',
           'c_art',
           'c_bart',
           'c_fbp',
           'c_gridrec',
           'c_mlem',
           'c_osem',
           'c_ospml_hybrid',
           'c_ospml_quad',
           'c_pml_hybrid',
           'c_pml_quad',
           'c_sirt',
           'c_remove_ring']


def c_shared_lib(lib_name):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            ext = '.pyd'
        else:
            ext = '.so'
        _fname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        libpath = glob.glob(_fname + '/' + lib_name + '*' + ext)[0]
        return ctypes.CDLL(libpath)
    except (OSError, IndexError):
        logger.warning('OSError: Shared library missing.')


LIB_TOMOPY = c_shared_lib('libtomopy')


def c_normalize_bg(tomo, air):
    dt, dy, dx = tomo.shape

    LIB_TOMOPY.normalize_bg.restype = dtype.as_c_void_p()
    LIB_TOMOPY.normalize_bg(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(dt),
        dtype.as_c_int(dy),
        dtype.as_c_int(dx),
        dtype.as_c_int(air))


def c_remove_stripe_sf(tomo, size):
    #TODO: we should fix this elsewhere... 
    # TOMO object must be contiguous for c function to work
    contiguous_tomo = np.require(tomo, requirements="AC")
    dx, dy, dz = tomo.shape
    istart = 0
    iend = dy

    LIB_TOMOPY.remove_stripe_sf.restype = dtype.as_c_void_p()
    LIB_TOMOPY.remove_stripe_sf(
        dtype.as_c_float_p(contiguous_tomo),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
        dtype.as_c_int(size),
        dtype.as_c_int(istart),
        dtype.as_c_int(iend))
    tomo[:] = contiguous_tomo[:]

def c_project(obj, center, tomo, theta):
    if len(obj.shape) == 2:
        # no y-axis (only one slice)
        oy = 1
        ox, oz = obj.shape
    else:
        oy, ox, oz = obj.shape

    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.project.restype = dtype.as_c_void_p()
    LIB_TOMOPY.project(
        dtype.as_c_float_p(obj),
        dtype.as_c_int(oy),
        dtype.as_c_int(ox),
        dtype.as_c_int(oz),
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center),
        dtype.as_c_float_p(theta))


def c_sample(mode, arr, dx, dy, dz, level, axis, out):
    LIB_TOMOPY.sample.restype = dtype.as_c_void_p()
    LIB_TOMOPY.sample(
        dtype.as_c_int(mode),
        dtype.as_c_float_p(arr),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
        dtype.as_c_int(level),
        dtype.as_c_int(axis),
        dtype.as_c_float_p(out))
    return out


def c_art(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.art.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.art,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']))


def c_bart(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.bart.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.bart,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']),
            dtype.as_c_int(kwargs['num_block']),
            dtype.as_c_float_p(kwargs['ind_block'])) #TODO: I think this should be int_p


def c_fbp(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.fbp.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.fbp,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_char_p(kwargs['filter_name']),
            dtype.as_c_float_p(kwargs['filter_par'])) # filter_par

def c_gridrec(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.gridrec.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.gridrec,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_char_p(kwargs['filter_name']),
            dtype.as_c_float_p(kwargs['filter_par']))


def c_mlem(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.mlem.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.mlem,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']))


def c_osem(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.osem.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.osem,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']),
            dtype.as_c_int(kwargs['num_block']),
            dtype.as_c_float_p(kwargs['ind_block'])) #TODO: should be int?


def c_ospml_hybrid(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.ospml_hybrid.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.ospml_hybrid,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']),
            dtype.as_c_float_p(kwargs['reg_par']),
            dtype.as_c_int(kwargs['num_block']),
            dtype.as_c_float_p(kwargs['ind_block'])) #TODO: should be int?


def c_ospml_quad(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.ospml_quad.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.ospml_quad,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']),
            dtype.as_c_float_p(kwargs['reg_par']),
            dtype.as_c_int(kwargs['num_block']),
            dtype.as_c_float_p(kwargs['ind_block'])) #TODO: should be int?


def c_pml_hybrid(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.pml_hybrid.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.pml_hybrid,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']),
            dtype.as_c_float_p(kwargs['reg_par']))


def c_pml_quad(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.pml_quad.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.pml_quad,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']),
            dtype.as_c_float_p(kwargs['reg_par']))


def c_sirt(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY.sirt.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.sirt,
            dtype.as_c_float_p(tomo),
            dtype.as_c_int(dy),
            dtype.as_c_int(dt),
            dtype.as_c_int(dx),
            dtype.as_c_float_p(center),
            dtype.as_c_float_p(theta),
            dtype.as_c_float_p(recon),
            dtype.as_c_int(kwargs['num_gridx']),
            dtype.as_c_int(kwargs['num_gridy']),
            dtype.as_c_int(kwargs['num_iter']))


def c_remove_ring(rec, *args):
    istart = 0
    iend = rec.shape[0]
    LIB_TOMOPY.remove_ring.restype = dtype.as_c_void_p()
    return (LIB_TOMOPY.remove_ring,
            dtype.as_c_float_p(rec),
            dtype.as_c_float(args[0]),  # center_x
            dtype.as_c_float(args[1]),  # center_y
            dtype.as_c_int(args[2]),  # dx
            dtype.as_c_int(args[3]),  # dy
            dtype.as_c_int(args[4]),  # dz
            dtype.as_c_float(args[5]),  # thresh_max
            dtype.as_c_float(args[6]),  # thresh_min
            dtype.as_c_float(args[7]),  # thresh
            dtype.as_c_int(args[8]),  # theta_min
            dtype.as_c_int(args[9]),  # rwidth
            dtype.as_c_int(istart),  # istart
            dtype.as_c_int(iend))  # iend
