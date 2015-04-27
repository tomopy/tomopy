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

import os
import ctypes
import numpy as np
import multiprocessing
import logging
logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['init_shared',
           'import_shared_lib',
           'as_float32',
           'as_int32',
           'as_c_float_p',
           'as_c_int',
           'as_c_char_p',
           'as_c_void_p']


def import_shared_lib(lib_name):
    """
    Get the path and import the C-shared library.
    """
    try:
        if os.name == 'nt':
            libpath = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..', 'lib', lib_name + '.pyd'))
            return ctypes.CDLL(libpath)
        else:
            libpath = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..', 'lib', lib_name + '.so'))
            return ctypes.CDLL(libpath)
    except OSError as e:
        logger.warning('OSError: Shared library missing.')


def as_float32(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype='float32')
    elif not arr.dtype == np.float32:
        arr = np.array(arr, dtype='float32')
    return arr


def as_int32(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype='int32')
    elif not arr.dtype == np.float32:
        arr = np.array(arr, dtype='int32')
    return arr


def as_c_float_p(arr):
    c_float_p = ctypes.POINTER(ctypes.c_float)
    return arr.ctypes.data_as(c_float_p)


def as_c_int(arr):
    return ctypes.c_int(arr)


def as_c_char_p(arr):
    c_char_p = ctypes.POINTER(ctypes.c_char)
    return arr.ctypes.data_as(c_char_p)


def as_c_void_p():
    return ctypes.POINTER(ctypes.c_void_p)
