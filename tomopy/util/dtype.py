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
Module for internal utility functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ctypes
import numpy as np
import multiprocessing as mp
import logging
import six

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['as_ndarray',
           'as_dtype',
           'as_float32',
           'as_int32',
           'as_uint8',
           'as_uint16',
           'as_c_float_p',
           'as_c_int',
           'as_c_int_p',
           'as_c_float',
           'as_c_char_p',
           'as_c_void_p']


def as_ndarray(arr, dtype=None, copy=False):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=dtype, copy=copy)
    return arr


def as_dtype(arr, dtype, copy=False):
    if not arr.dtype == dtype:
        arr = np.array(arr, dtype=dtype, copy=copy)
    return arr


def as_float32(arr):
    arr = as_ndarray(arr, np.float32)
    return as_dtype(arr, np.float32)


def as_int32(arr):
    arr = as_ndarray(arr, np.int32)
    return as_dtype(arr, np.int32)


def as_uint16(arr):
    arr = as_ndarray(arr, np.uint16)
    return as_dtype(arr, np.uint16)


def as_uint8(arr):
    arr = as_ndarray(arr, np.uint8)
    return as_dtype(arr, np.uint8)


def as_c_float_p(arr):
    c_float_p = ctypes.POINTER(ctypes.c_float)
    return arr.ctypes.data_as(c_float_p)


def as_c_int(arr):
    return ctypes.c_int(arr)


def as_c_int_p(arr):
    c_int_p = ctypes.POINTER(ctypes.c_int)
    return arr.ctypes.data_as(c_int_p)


def as_c_float(arr):
    return ctypes.c_float(arr)


def as_c_char_p(arr):
    return ctypes.c_char_p(six.b(arr))


def as_c_void_p():
    return ctypes.POINTER(ctypes.c_void_p)


def as_sharedmem(arr, copy=False):
    # first check to see if it already a shared array
    if not copy and is_sharedmem(arr):
        return arr
    # get ctype from numpy array
    temp_arr = np.empty((1), dtype=arr.dtype)
    ctype = type(np.ctypeslib.as_ctypes(temp_arr)._type_())
    # create shared ctypes object with no lock
    shared_obj = mp.RawArray(ctype, arr.size)
    # create numpy array from shared object
    # shared_arr = np.ctypeslib.as_array(shared_obj)
    shared_arr = np.frombuffer(shared_obj, dtype=arr.dtype)
    shared_arr = np.reshape(shared_arr, arr.shape)
    # copy data to shared array
    shared_arr[:] = arr[:]
    return shared_arr


def to_numpy_array(obj, dtype, shape):
    return np.frombuffer(obj, dtype=dtype).reshape(shape)


def is_sharedmem(arr):
    # attempt to determine if data is in shared memory
    try:
        base = arr.base
        if base is None:
            return False
        elif type(base).__module__.startswith('multiprocessing.sharedctypes'):
            return True
        else:
            return is_sharedmem(base)
    except:
        return False


def get_shared_mem(arr):
    try:
        while isinstance(arr, np.ndarray):
            arr = arr.base
    except:
        pass
    return arr


def is_contiguous(arr):
    return arr.flags.c_contiguous


def empty_shared_array(shape, dtype=np.float32):
    # create a shared ndarray with the provided shape and type
    # get ctype from np dtype
    temp_arr = np.empty((1), dtype)
    ctype = type(np.ctypeslib.as_ctypes(temp_arr)._type_())
    # create shared ctypes object with no lock
    size = 1
    for dim in shape:
        size *= dim
    shared_obj = mp.RawArray(ctype, int(size))
    # create numpy array from shared object
    arr = np.frombuffer(shared_obj, dtype)
    arr = arr.reshape(shape)
    return arr
