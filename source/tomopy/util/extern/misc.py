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

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = "restructuredtext en"
__all__ = [
    "c_sample",
    "c_remove_ring",
    "c_median_filt3d_float32",
    "c_median_filt3d_uint16",
    "c_inpainter",
]

LIB_TOMOPY_MISC = c_shared_lib("tomo-misc")


def c_sample(
    mode,
    arr,
    dx,
    dy,
    dz,
    level,
    axis,
    out,
):
    LIB_TOMOPY_MISC.sample.restype = dtype.as_c_void_p()
    LIB_TOMOPY_MISC.sample(
        dtype.as_c_int(mode),
        dtype.as_c_float_p(arr),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
        dtype.as_c_int(level),
        dtype.as_c_int(axis),
        dtype.as_c_float_p(out),
    )
    return out


def c_remove_ring(rec, *args):
    istart = 0
    iend = rec.shape[0]
    LIB_TOMOPY_MISC.remove_ring.restype = dtype.as_c_void_p()
    return LIB_TOMOPY_MISC.remove_ring(
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
        dtype.as_c_int(args[10]),  # int_mode
        dtype.as_c_int(istart),  # istart
        dtype.as_c_int(iend),  # iend
    )


def c_median_filt3d_float32(
    input,
    output,
    kernel_half_size,
    absdif,
    ncore,
    dx,
    dy,
    dz,
):
    LIB_TOMOPY_MISC.medianfilter_main_float.restype = dtype.as_c_void_p()
    LIB_TOMOPY_MISC.medianfilter_main_float(
        dtype.as_c_float_p(input),
        dtype.as_c_float_p(output),
        dtype.as_c_int(kernel_half_size),
        dtype.as_c_float(absdif),
        dtype.as_c_int(ncore),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
    )
    return output


def c_median_filt3d_uint16(
    input,
    output,
    kernel_half_size,
    absdif,
    ncore,
    dx,
    dy,
    dz,
):
    LIB_TOMOPY_MISC.medianfilter_main_uint16.restype = dtype.as_c_void_p()
    LIB_TOMOPY_MISC.medianfilter_main_uint16(
        dtype.as_c_uint16_p(input),
        dtype.as_c_uint16_p(output),
        dtype.as_c_int(kernel_half_size),
        dtype.as_c_float(absdif),
        dtype.as_c_int(ncore),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
    )
    return output


def c_inpainter(
    input,
    mask,
    output,
    iterations,
    kernel_half_size,
    method_type,
    ncore,
    dx,
    dy,
    dz,
):
    LIB_TOMOPY_MISC.Inpainter_morph_main.restype = dtype.as_c_void_p()
    LIB_TOMOPY_MISC.Inpainter_morph_main(
        dtype.as_c_float_p(input),
        dtype.as_c_bool_p(mask),
        dtype.as_c_float_p(output),
        dtype.as_c_int(iterations),
        dtype.as_c_int(kernel_half_size),
        dtype.as_c_int(method_type),
        dtype.as_c_int(ncore),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
    )
    return output
