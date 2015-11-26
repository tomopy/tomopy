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
Module for internal utility functions.
"""

from __future__ import absolute_import, division, print_function

import os.path
import ctypes
import tomopy.util.dtype as dtype
import tomopy.util.mproc as mproc
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['c_shared_lib',
           'c_project',
           'c_normalize_bg',
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
           'c_remove_rings']


def c_shared_lib(lib_name):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            ext = '.pyd'
        else:
            ext = '.so'
        libpath = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'lib', lib_name + ext))
        return ctypes.CDLL(libpath)
    except OSError as e:
        logger.warning('OSError: Shared library missing.')


LIB_TOMOPY = c_shared_lib('libtomopy')


def c_normalize_bg(dx, dy, dz, air, istart, iend):
    tomo = mproc.SHARED_ARRAY

    LIB_TOMOPY.normalize_bg.restype = dtype.as_c_void_p()
    LIB_TOMOPY.normalize_bg(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
        dtype.as_c_int(air),
        dtype.as_c_int(istart),
        dtype.as_c_int(iend))


def c_project(ox, oy, oz, theta, center, dx, dy, dz, istart, iend):
    obj = mproc.SHARED_OBJ
    tomo = mproc.SHARED_ARRAY

    LIB_TOMOPY.project.restype = dtype.as_c_void_p()
    LIB_TOMOPY.project(
        dtype.as_c_float_p(obj),
        dtype.as_c_int(ox),
        dtype.as_c_int(oy),
        dtype.as_c_int(oz),
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(dx),
        dtype.as_c_int(dy),
        dtype.as_c_int(dz),
        dtype.as_c_float_p(center),
        dtype.as_c_float_p(theta),
        dtype.as_c_int(istart),
        dtype.as_c_int(iend))


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


def c_art(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.art.restype = dtype.as_c_void_p()
    LIB_TOMOPY.art(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_bart(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.bart.restype = dtype.as_c_void_p()
    LIB_TOMOPY.bart(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_int(args[5]['num_block']),
        dtype.as_c_float_p(args[5]['ind_block']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_fbp(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.fbp.restype = dtype.as_c_void_p()
    LIB_TOMOPY.fbp(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_char_p(args[5]['filter_name']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_gridrec(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.gridrec.restype = dtype.as_c_void_p()
    LIB_TOMOPY.gridrec(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_char_p(args[5]['filter_name']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_mlem(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.mlem.restype = dtype.as_c_void_p()
    LIB_TOMOPY.mlem(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_osem(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.osem.restype = dtype.as_c_void_p()
    LIB_TOMOPY.osem(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_int(args[5]['num_block']),
        dtype.as_c_float_p(args[5]['ind_block']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_ospml_hybrid(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.ospml_hybrid.restype = dtype.as_c_void_p()
    LIB_TOMOPY.ospml_hybrid(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_float_p(args[5]['reg_par']),
        dtype.as_c_int(args[5]['num_block']),
        dtype.as_c_float_p(args[5]['ind_block']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_ospml_quad(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.ospml_quad.restype = dtype.as_c_void_p()
    LIB_TOMOPY.ospml_quad(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_float_p(args[5]['reg_par']),
        dtype.as_c_int(args[5]['num_block']),
        dtype.as_c_float_p(args[5]['ind_block']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_pml_hybrid(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.pml_hybrid.restype = dtype.as_c_void_p()
    LIB_TOMOPY.pml_hybrid(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_float_p(args[5]['reg_par']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_pml_quad(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.pml_quad.restype = dtype.as_c_void_p()
    LIB_TOMOPY.pml_quad(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_float_p(args[5]['reg_par']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_sirt(*args):
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    LIB_TOMOPY.sirt.restype = dtype.as_c_void_p()
    LIB_TOMOPY.sirt(
        dtype.as_c_float_p(tomo),
        dtype.as_c_int(args[0]),  # dx
        dtype.as_c_int(args[1]),  # dy
        dtype.as_c_int(args[2]),  # dz
        dtype.as_c_float_p(args[3]),  # center
        dtype.as_c_float_p(args[4]),  # theta
        dtype.as_c_float_p(recon),
        dtype.as_c_int(args[5]['num_gridx']),
        dtype.as_c_int(args[5]['num_gridy']),
        dtype.as_c_int(args[5]['num_iter']),
        dtype.as_c_int(args[6]),  # istart
        dtype.as_c_int(args[7]))  # iend


def c_remove_rings(*args):
    data = mproc.SHARED_ARRAY

    LIB_TOMOPY.remove_rings.restype = dtype.as_c_void_p()
    LIB_TOMOPY.remove_rings(
        dtype.as_c_float_p(data),
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
        dtype.as_c_int(args[10]),  # istart
        dtype.as_c_int(args[11]))  # iend
