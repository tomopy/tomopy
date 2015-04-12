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

:Author: Doga Gursoy
:Organization: Argonne National Laboratory

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing as mp
import ctypes
from contextlib import closing


__docformat__ = 'restructuredtext en'
__all__ = ['distribute_jobs']


def distribute_jobs(data, func, args, axis, ncore=None, nchunk=None):
    """
    Distribute N-dimensional shared-memory data in chunks into cores.

    Parameters
    ----------
    func : srt
        Name of the function to be parallelized.

    args : list
        Arguments to that function in a list.

    axis : scalar
        The axis for slicing data.

    ncore : scalar, optional
        Number of processor that will be assigned to jobs.

    nchunk : scalar, optional
        Number of data size for each processor.

    Returns
    -------
    out : ndarray
        Output data.
    """
    # Arrange number of processors.
    if ncore is None:
        ncore = mp.cpu_count()
    dims = data.shape[axis]

    # Maximum number of available processors for the task.
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
        ind_start = m * nchunk
        ind_end = (m + 1) * nchunk
        if ind_start >= dims:
            npool -= 1
            break
        if ind_end > dims:
            ind_end = dims

        arr = []
        arr.append(func)
        arr.append("SHARED")
        for a in str(args):
            arr.append(a)
        arr.append(range(ind_start, ind_end))
        arg.append(arr)

    shared_data = mp.Array(ctypes.c_float, data.size)
    shared_data = _to_numpy_array(shared_data, data.shape)
    shared_data[:] = data

    # write to arr from different processes
    with closing(mp.Pool(
            initializer=_init_shared, initargs=(shared_data,))) as p:
        p.map_async(_arg_parser, arg)
    p.join()
    p.close()
    return shared_data


def _arg_parser(args):
    func = args[0]
    func(*args[1::])


def _init_shared(shared_data_):
    global shared_data
    shared_data = shared_data_


def _to_numpy_array(mp_arr, dshape):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    return np.reshape(a, dshape)
