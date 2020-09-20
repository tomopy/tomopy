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
Module for external library wrappers.
"""

import tomopy.util.dtype as dtype
from . import c_shared_lib
from . import _missing_library

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['c_accel_mlem',
           'c_accel_sirt']

LIB_TOMOPY_ACCEL = c_shared_lib("libtomopy-accel", error=False)


def c_accel_mlem(tomo, center, recon, theta, **kwargs):

    if LIB_TOMOPY_ACCEL is None:
        _missing_library("MLEM ACCEL")

    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_ACCEL.cxx_mlem.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_ACCEL.cxx_mlem(
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
        dtype.as_c_int(kwargs['pool_size']),
        dtype.as_c_char_p(kwargs['interpolation']),
        dtype.as_c_char_p(kwargs['device']),
        dtype.as_c_int_p(kwargs['grid_size']),
        dtype.as_c_int_p(kwargs['block_size']))


def c_accel_sirt(tomo, center, recon, theta, **kwargs):

    if LIB_TOMOPY_ACCEL is None:
        _missing_library("SIRT ACCEL")

    if len(tomo.shape) == 2:
        # no y-axis (only one slice)
        dy = 1
        dt, dx = tomo.shape
    else:
        dy, dt, dx = tomo.shape

    LIB_TOMOPY_ACCEL.cxx_sirt.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_ACCEL.cxx_sirt(
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
        dtype.as_c_int(kwargs['pool_size']),
        dtype.as_c_char_p(kwargs['interpolation']),
        dtype.as_c_char_p(kwargs['device']),
        dtype.as_c_int_p(kwargs['grid_size']),
        dtype.as_c_int_p(kwargs['block_size']))
