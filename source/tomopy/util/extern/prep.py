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

import numpy as np

import tomopy.util.dtype as dtype
from . import c_shared_lib


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['c_normalize_bg',
           'c_remove_stripe_sf',
           'c_stripes_detect3d',
           'c_stripesmask3d']

LIB_TOMOPY_PREP = c_shared_lib("tomo-prep")


def c_normalize_bg(tomo, air):
    dt, dy, dx = tomo.shape

    LIB_TOMOPY_PREP.normalize_bg.restype = dtype.as_c_void_p()
    LIB_TOMOPY_PREP.normalize_bg(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(dt),
        dtype.as_c_int(dy),
        dtype.as_c_int(dx),
        dtype.as_c_int(air))


def c_remove_stripe_sf(tomo, size):

    # TODO: we should fix this elsewhere...
    # TOMO object must be contiguous for c function to work
    contiguous_tomo = np.require(tomo, requirements="AC")
    dx, dy, dz = tomo.shape
    istart = 0
    iend = dy

    LIB_TOMOPY_PREP.remove_stripe_sf.restype = dtype.as_c_void_p()
    LIB_TOMOPY_PREP.remove_stripe_sf(
        dtype.as_c_float_p(contiguous_tomo),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
        dtype.as_c_int(size),
        dtype.as_c_int(istart),
        dtype.as_c_int(iend))
    tomo[:] = contiguous_tomo[:]

def c_stripes_detect3d(
    input,
    output,
    size,
    radius,
    ncore,
    dx,
    dy,
    dz,
):
    LIB_TOMOPY_PREP.stripesdetect3d_main_float.restype = dtype.as_c_void_p()
    LIB_TOMOPY_PREP.stripesdetect3d_main_float(
        dtype.as_c_float_p(input),
        dtype.as_c_float_p(output),
        dtype.as_c_int(size),
        dtype.as_c_int(radius),
        dtype.as_c_int(ncore),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
    )
    return output

def c_stripesmask3d(
    input,
    output,
    threshold_val,
    min_stripe_length,
    min_stripe_depth,
    min_stripe_width,
    sensitivity_perc,
    ncore,
    dx,
    dy,
    dz,
):
    LIB_TOMOPY_PREP.stripesmask3d_main_float.restype = dtype.as_c_void_p()
    LIB_TOMOPY_PREP.stripesmask3d_main_float(
        dtype.as_c_float_p(input),
        dtype.as_c_bool_p(output),
        dtype.as_c_float(threshold_val),
        dtype.as_c_int(min_stripe_length),
        dtype.as_c_int(min_stripe_depth),
        dtype.as_c_int(min_stripe_width),
        dtype.as_c_float(sensitivity_perc),
        dtype.as_c_int(ncore),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
    )
    return output
