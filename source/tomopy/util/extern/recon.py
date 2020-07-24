#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2015-2019. UChicago Argonne, LLC. This software was produced  #
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
Module for recon library wrappers.
"""
import numpy as np

import tomopy.util.dtype as dtype
from . import c_shared_lib
from .accel import c_accel_mlem
from .accel import c_accel_sirt


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['c_project',
           'c_project2',
           'c_project3',
           'c_art',
           'c_bart',
           'c_fbp',
           'c_mlem',
           'c_osem',
           'c_ospml_hybrid',
           'c_ospml_quad',
           'c_pml_hybrid',
           'c_pml_quad',
           'c_sirt',
           'c_tv',
           'c_grad',
           'c_tikh',
           'c_vector',
           'c_vector2',
           'c_vector3']

LIB_TOMOPY_RECON = c_shared_lib("libtomopy-recon")


def c_project(obj, center, tomo, theta):
    # TODO: we should fix this elsewhere...
    # TOMO object must be contiguous for c function to work

    contiguous_tomo = np.require(tomo, requirements="AC")
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

    LIB_TOMOPY_RECON.project.restype = dtype.as_c_void_p()
    LIB_TOMOPY_RECON.project(
        dtype.as_c_float_p(obj),
        dtype.as_c_int(oy),
        dtype.as_c_int(ox),
        dtype.as_c_int(oz),
        dtype.as_c_float_p(contiguous_tomo),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center),
        dtype.as_c_float_p(theta))
    tomo[:] = contiguous_tomo[:]


def c_project2(objx, objy, center, tomo, theta):
    # TODO: we should fix this elsewhere...
    # TOMO object must be contiguous for c function to work

    contiguous_tomo = np.require(tomo, requirements="AC")
    if len(objx.shape) == 2:
        # no y-axis (only one slice)
        oy = 1
        ox, oz = objx.shape
    else:
        oy, ox, oz = objx.shape

    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.project2.restype = dtype.as_c_void_p()
    LIB_TOMOPY_RECON.project2(
        dtype.as_c_float_p(objx),
        dtype.as_c_float_p(objy),
        dtype.as_c_int(oy),
        dtype.as_c_int(ox),
        dtype.as_c_int(oz),
        dtype.as_c_float_p(contiguous_tomo),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center),
        dtype.as_c_float_p(theta))
    tomo[:] = contiguous_tomo[:]


def c_project3(objx, objy, objz, center, tomo, theta, axis):
    # TODO: we should fix this elsewhere...
    # TOMO object must be contiguous for c function to work

    contiguous_tomo = np.require(tomo, requirements="AC")
    if len(objx.shape) == 2:
        # no y-axis (only one slice)
        oy = 1
        ox, oz = objx.shape
    else:
        oy, ox, oz = objx.shape

    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.project3.restype = dtype.as_c_void_p()
    LIB_TOMOPY_RECON.project3(
        dtype.as_c_float_p(objx),
        dtype.as_c_float_p(objy),
        dtype.as_c_float_p(objz),
        dtype.as_c_int(oy),
        dtype.as_c_int(ox),
        dtype.as_c_int(oz),
        dtype.as_c_float_p(contiguous_tomo),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center),
        dtype.as_c_float_p(theta),
        dtype.as_c_int(axis))
    tomo[:] = contiguous_tomo[:]


def c_art(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.art.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.art(
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

    LIB_TOMOPY_RECON.bart.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.bart(
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
        dtype.as_c_float_p(kwargs['ind_block']))  # TODO: I think this should be int_p


def c_fbp(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.fbp.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.fbp(
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
        dtype.as_c_float_p(kwargs['filter_par']))  # filter_par


def c_mlem(tomo, center, recon, theta, **kwargs):

    if kwargs['accelerated']:
        return c_accel_mlem(tomo, center, recon, theta, **kwargs)

    else:
        if len(tomo.shape) == 2:
            # no y-axis (only one slice)
            dy = 1
            dt, dx = tomo.shape
        else:
            dy, dt, dx = tomo.shape

        LIB_TOMOPY_RECON.mlem.restype = dtype.as_c_void_p()
        return LIB_TOMOPY_RECON.mlem(
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

    LIB_TOMOPY_RECON.osem.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.osem(
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
        dtype.as_c_float_p(kwargs['ind_block']))  # TODO: should be int?


def c_ospml_hybrid(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.ospml_hybrid.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.ospml_hybrid(
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
        dtype.as_c_float_p(kwargs['ind_block']))  # TODO: should be int?


def c_ospml_quad(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.ospml_quad.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.ospml_quad(
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
        dtype.as_c_float_p(kwargs['ind_block']))  # TODO: should be int?


def c_pml_hybrid(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.pml_hybrid.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.pml_hybrid(
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

    LIB_TOMOPY_RECON.pml_quad.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.pml_quad(
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

    if kwargs['accelerated']:
        return c_accel_sirt(tomo, center, recon, theta, **kwargs)

    else:

        if len(tomo.shape) == 2:
            # no y-axis (only one slice)
            dy = 1
            dt, dx = tomo.shape
        else:
            dy, dt, dx = tomo.shape

        LIB_TOMOPY_RECON.sirt.restype = dtype.as_c_void_p()
        return LIB_TOMOPY_RECON.sirt(
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


def c_tv(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.tv.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.tv(
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


def c_grad(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.grad.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.grad(
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


def c_tikh(tomo, center, recon, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.tikh.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.tikh(
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
        dtype.as_c_float_p(kwargs['reg_data']),
        dtype.as_c_float_p(kwargs['reg_par']))


def c_vector(tomo, center, recon1, recon2, theta, **kwargs):
    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_RECON.vector.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.vector(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center),
        dtype.as_c_float_p(theta),
        dtype.as_c_float_p(recon1),
        dtype.as_c_float_p(recon2),
        dtype.as_c_int(kwargs['num_gridx']),
        dtype.as_c_int(kwargs['num_gridy']),
        dtype.as_c_int(kwargs['num_iter']))


def c_vector2(tomo1, tomo2, center1, center2, recon1, recon2, recon3,
              theta1, theta2, axis1, axis2, **kwargs):
    if len(tomo1.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo1.shape
    else:
        dy, dt, dx = tomo1.shape

    LIB_TOMOPY_RECON.vector2.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.vector2(
        dtype.as_c_float_p(tomo1),
        dtype.as_c_float_p(tomo2),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center1),
        dtype.as_c_float_p(center2),
        dtype.as_c_float_p(theta1),
        dtype.as_c_float_p(theta2),
        dtype.as_c_float_p(recon1),
        dtype.as_c_float_p(recon2),
        dtype.as_c_float_p(recon3),
        dtype.as_c_int(kwargs['num_gridx']),
        dtype.as_c_int(kwargs['num_gridy']),
        dtype.as_c_int(kwargs['num_iter']),
        dtype.as_c_int(axis1),
        dtype.as_c_int(axis2))


def c_vector3(tomo1, tomo2, tomo3, center1, center2, center3, recon1,
              recon2, recon3, theta1, theta2, theta3, axis1, axis2,
              axis3, **kwargs):
    if len(tomo1.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo1.shape
    else:
        dy, dt, dx = tomo1.shape

    LIB_TOMOPY_RECON.vector3.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_RECON.vector3(
        dtype.as_c_float_p(tomo1),
        dtype.as_c_float_p(tomo2),
        dtype.as_c_float_p(tomo3),
        dtype.as_c_int(dy),
        dtype.as_c_int(dt),
        dtype.as_c_int(dx),
        dtype.as_c_float_p(center1),
        dtype.as_c_float_p(center2),
        dtype.as_c_float_p(center3),
        dtype.as_c_float_p(theta1),
        dtype.as_c_float_p(theta2),
        dtype.as_c_float_p(theta3),
        dtype.as_c_float_p(recon1),
        dtype.as_c_float_p(recon2),
        dtype.as_c_float_p(recon3),
        dtype.as_c_int(kwargs['num_gridx']),
        dtype.as_c_int(kwargs['num_gridy']),
        dtype.as_c_int(kwargs['num_iter']),
        dtype.as_c_int(axis1),
        dtype.as_c_int(axis2),
        dtype.as_c_int(axis3))
