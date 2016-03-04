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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import multiprocessing as mp
import math
from contextlib import closing
from .dtype import as_sharedmem
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['distribute_jobs']

#global shared variables
SHARED_ARRAYS = None
SHARED_OUT = None

def distribute_jobs(arr,
                    func,
                    axis,
                    args=None,
                    kwargs=None,
                    ncore=None,
                    nchunk=None,
                    out=None):
    """
    Distribute N-dimensional shared-memory array into cores by splitting along
    an axis.

    Parameters
    ----------
    arr : ndarray, or iterable(ndarray)
        Array(s) to be split up for processing.
    func : func
        Function to be parallelized.  Should return an ndarray.
    args : list
        Arguments of the function in a list.
    kwargs : list
        Keyword arguments of the function in a dictionary.
    axis : int
        Axis along which parallelization is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size to use when parallelizing data.  None will maximize the chunk
        size for the number of cores used.  Zero will use a chunk size of one, 
        but will also remove the dimension from the array.
    out : ndarray, optional
        Output array.  Results of functions will be compiled into this array.
        If not provided, new array will be created for output.

    Returns
    -------
    ndarray
        Output array.
    """
    if isinstance(arr, np.ndarray):
        arrs = [arr]
    else:
        # assume this is multiple arrays
        arrs = list(arr)

    axis_size = arrs[0].shape[axis]
    # limit chunk size to size of array along axis     
    if nchunk and nchunk > axis_size:
        nchunk = axis_size
    # default ncore to max and limit number of cores to max number
    if ncore is None or ncore > mp.cpu_count():
        ncore = mp.cpu_count()
    # limit number of cores based on nchunk so that all cores are used
    if ncore > math.ceil(axis_size / (nchunk or 1)):
        ncore = int(math.ceil(axis_size / (nchunk or 1)))
    # default nchunk to only use each core for one call
    if nchunk is None:
        nchunk = int(math.ceil(axis_size / ncore))

    # prepare all args (func, args, kwargs)
    # NOTE: args will include shared_arr slice as first arg
    args = args or tuple()
    kwargs = kwargs or dict()

    # prepare global sharedmem arrays
    shared_arrays = []
    shared_out = None
    for arr in arrs:
        arr_shared = as_sharedmem(arr)
        shared_arrays.append(arr_shared)
        if out is not None and np.may_share_memory(arr, out) and \
            arr.shape == out.shape and arr.dtype == out.dtype:
            # assume these are the same array
            shared_out = arr_shared
    if out is None:
        # default out to last array in list
        out = shared_arrays[-1]
        shared_out = out
    if shared_out is None:
        shared_out = as_sharedmem(out)

    # kick off process
    # testing single threaded
#    init_shared(shared_arrays, shared_out)
#    for i in xrange(0, axis_size, nchunk or 1):
#        logger.warning("loop %d of %d"%(i+1, axis_size))
#        if nchunk:
#            _arg_parser((func, args, kwargs, np.s_[i:i+nchunk], axis))
#        else:
#            _arg_parser((func, args, kwargs, i, axis))

    # if nchunk is zero, remove dimension from slice.
    map_args = []
    for i in xrange(0, axis_size, nchunk or 1):
        if nchunk:
            map_args.append((func, args, kwargs, np.s_[i:i+nchunk], axis))                
        else:
            map_args.append((func, args, kwargs, i, axis))

    with closing(mp.Pool(processes=ncore,
                         initializer=init_shared,
                         initargs=(shared_arrays, shared_out))) as p:
        if p._pool:
            proclist = p._pool[:]
            res = p.map_async(_arg_parser, map_args)
            try:
                while not res.ready():
                    if any(proc.exitcode for proc in proclist):
                        p.terminate()
                        raise RuntimeError("Child process terminated before finishing")
                    res.wait(timeout=1)
            except KeyboardInterrupt:
                p.terminate()
                raise
        else:
            p.map(_arg_parser, map_args)
    try:
        p.join()
    except:
        p.terminate()
        raise

    # NOTE: will only copy if out wasn't sharedmem
    out[:] = shared_out[:]
    return out


def init_shared(shared_arrays, shared_out):
    global SHARED_ARRAYS
    global SHARED_OUT
    SHARED_ARRAYS = shared_arrays
    SHARED_OUT = shared_out


def _arg_parser(params):
    global SHARED_ARRAYS
    global SHARED_OUT
    func, args, kwargs, slc, axis = params
    func_args = tuple((slice_axis(a, slc, axis) for a in SHARED_ARRAYS)) + args
    #NOTE: will only copy if actually different arrays
    result = func(*func_args, **kwargs)
    if result is not None and isinstance(result, np.ndarray):
        outslice = slice_axis(SHARED_OUT, slc, axis)
        outslice[:] = result[:]

# apply slice to specific axis on ndarray
def slice_axis(arr, slc, axis):
    return arr[[slice(None) if i != axis else slc for i in xrange(arr.ndim)]]
