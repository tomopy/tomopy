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
import numpy as np
from tomopy.util import *
import tomopy.misc.mproc as mp
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['import_shared_lib',
           'c_project',
           'c_normalize_bg']


def import_shared_lib(lib_name):
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
                '..', 'lib', lib_name + ext))
        return ctypes.CDLL(libpath)
    except OSError as e:
        logger.warning('OSError: Shared library missing.')


LIB_TOMOPY = import_shared_lib('libtomopy')


def c_normalize_bg(dx, dy, dz, air, istart, iend):
    tomo = mp.SHARED_ARRAY

    LIB_TOMOPY.correct_air.restype = as_c_void_p()
    LIB_TOMOPY.correct_air(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_int(air),
        as_c_int(istart),
        as_c_int(iend))


def c_project(ox, oy, oz, theta, center, dx, dy, dz, istart, iend):
    obj = mp.SHARED_OBJ
    tomo = mp.SHARED_ARRAY

    LIB_TOMOPY.project.restype = as_c_void_p()
    LIB_TOMOPY.project(
        as_c_float_p(obj),
        as_c_int(ox),
        as_c_int(oy),
        as_c_int(oz),
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_int(istart),
        as_c_int(iend))


def _art(
        dx, dy, dz, theta, center, num_gridx, num_gridy,
        num_iter, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.art.restype = as_c_void_p()
    LIB_TOMOPY.art(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_int(istart),
        as_c_int(iend))


def _bart(
        dx, dy, dz, theta, center, num_gridx, num_gridy,
        num_iter, num_block, ind_block, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.bart.restype = as_c_void_p()
    LIB_TOMOPY.bart(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_int(num_block),
        as_c_float_p(ind_block),
        as_c_int(istart),
        as_c_int(iend))


def _fbp(
        dx, dy, dz, theta, center, num_gridx, num_gridy,
        filter_name, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.fbp.restype = as_c_void_p()
    LIB_TOMOPY.fbp(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_char_p(filter_name),
        as_c_int(istart),
        as_c_int(iend))


def _gridrec(
        dx, dy, dz, theta, center, num_gridx, num_gridy,
        filter_name, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.gridrec.restype = as_c_void_p()
    LIB_TOMOPY.gridrec(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_char_p(filter_name),
        as_c_int(istart),
        as_c_int(iend))


def _mlem(
        dx, dy, dz, theta, center, num_gridx,
        num_gridy, num_iter, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.mlem.restype = as_c_void_p()
    LIB_TOMOPY.mlem(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_int(istart),
        as_c_int(iend))


def _osem(
        dx, dy, dz, theta, center, num_gridx, num_gridy, num_iter,
        num_block, ind_block, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.osem.restype = as_c_void_p()
    LIB_TOMOPY.osem(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_int(num_block),
        as_c_float_p(ind_block),
        as_c_int(istart),
        as_c_int(iend))


def _ospml_hybrid(
        dx, dy, dz, theta, center, num_gridx, num_gridy, num_iter,
        reg_par, num_block, ind_block, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.ospml_hybrid.restype = as_c_void_p()
    LIB_TOMOPY.ospml_hybrid(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_float_p(reg_par),
        as_c_int(num_block),
        as_c_float_p(ind_block),
        as_c_int(istart),
        as_c_int(iend))


def _ospml_quad(
        dx, dy, dz, theta, center, num_gridx, num_gridy, num_iter,
        reg_par, num_block, ind_block, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.ospml_quad.restype = as_c_void_p()
    LIB_TOMOPY.ospml_quad(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_float_p(reg_par),
        as_c_int(num_block),
        as_c_float_p(ind_block),
        as_c_int(istart),
        as_c_int(iend))


def _pml_hybrid(
        dx, dy, dz, theta, center, num_gridx, num_gridy,
        num_iter, reg_par, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.pml_hybrid.restype = as_c_void_p()
    LIB_TOMOPY.pml_hybrid(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_float_p(reg_par),
        as_c_int(istart),
        as_c_int(iend))


def _pml_quad(
        dx, dy, dz, theta, center, num_gridx, num_gridy, num_iter,
        reg_par, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.pml_quad.restype = as_c_void_p()
    LIB_TOMOPY.pml_quad(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_float_p(reg_par),
        as_c_int(istart),
        as_c_int(iend))


def _sirt(
        dx, dy, dz, theta, center, num_gridx, num_gridy,
        num_iter, istart, iend):
    tomo = mp.SHARED_TOMO
    recon = mp.SHARED_ARRAY

    LIB_TOMOPY.sirt.restype = as_c_void_p()
    LIB_TOMOPY.sirt(
        as_c_float_p(tomo),
        as_c_int(dx),
        as_c_int(dy),
        as_c_int(dz),
        as_c_float_p(center),
        as_c_float_p(theta),
        as_c_float_p(recon),
        as_c_int(num_gridx),
        as_c_int(num_gridy),
        as_c_int(num_iter),
        as_c_int(istart),
        as_c_int(iend))
