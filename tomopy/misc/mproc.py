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
Module for multiprocessing tasks.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
import ctypes
from contextlib import closing


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['distribute_jobs']


def distribute_jobs(arr, func, args, axis, ncore=None, nchunk=None):
    """
    Distribute N-dimensional shared-memory array in chunks into cores.

    Parameters
    ----------
    func : func
        Function to be parallelized.
    args : list
        Arguments of the function in a list.
    axis : int
        Axis along which parallelization is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Output array.
    """
    # Arrange number of processors.
    if ncore is None:
        ncore = mp.cpu_count()
    dims = arr.shape[axis]

    # Maximum number of processors for the task.
    if dims < ncore:
        ncore = dims

    # Arrange chunk size.
    if nchunk is None:
        nchunk = (dims - 1) // ncore + 1

    # Determine pool size.
    npool = dims // nchunk + 1

    # Populate arguments for workers.
    arg = []
    for m in range(npool):
        istart = m * nchunk
        iend = (m + 1) * nchunk
        if istart >= dims:
            npool -= 1
            break
        if iend > dims:
            iend = dims

        _arg = []
        _arg.append(func)
        for a in args:
            _arg.append(a)
        _arg.append(istart)
        _arg.append(iend)
        arg.append(_arg)

    shared_arr = mp.Array(ctypes.c_float, arr.size)
    shared_arr = _to_numpy_array(shared_arr, arr.shape)
    shared_arr[:] = arr

    # write to arr from different processes
    with closing(mp.Pool(
            initializer=_init_shared, initargs=(shared_arr,))) as p:
        p.map_async(_arg_parser, arg)
    p.join()
    p.close()
    return shared_arr


def _arg_parser(args):
    func = args[0]
    func(*args[1::])


def _init_shared(shared_arr_):
    global SHARED_ARRAY
    SHARED_ARRAY = shared_arr_


def _to_numpy_array(mp_arr, dshape):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    return np.reshape(a, dshape)
