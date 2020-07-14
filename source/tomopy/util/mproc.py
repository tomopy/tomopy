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
Module for multiprocessing tasks.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import multiprocessing as mp
import math
from contextlib import closing
from .dtype import as_sharedmem, to_numpy_array, get_shared_mem
import logging
import numexpr as ne

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['distribute_jobs']

# global shared variables
SHARED_ARRAYS = None
SHARED_OUT = None
SHARED_QUEUE = None
INTYPE = None
INSHAPE = None
OUTTYPE = None
OUTSHAPE = None
ON_HOST = False
DEBUG = False


def get_rank():
    """ Get the rank of the process """
    try:
        from mpi4py import MPI
        comm_w = MPI.COMM_WORLD
        return comm_w.Get_rank()
    except:
        return 0


def get_nproc():
    """ Get the number of processes """
    try:
        from mpi4py import MPI
        comm_w = MPI.COMM_WORLD
        return comm_w.Get_size()
    except:
        return 1

def barrier():
    """ Barrier for MPI processes """
    try:
        from mpi4py import MPI
        comm_w = MPI.COMM_WORLD
        comm_w.Barrier()
    except:
        pass


def set_debug(val=True):
    """
    Set the global DEBUG variable.

    If DEBUG is True, all computations will be run on the host process instead
    of distributing work over different processes. This can help to debug
    functions that give errors or cause segmentation faults. If DEBUG is False
    (the default value), work is distributed over different processes.
    """
    global DEBUG
    DEBUG = val


def get_ncore_nchunk(axis_size, ncore=None, nchunk=None):
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
    return ncore, nchunk


def get_ncore_slices(axis_size, ncore=None, nchunk=None):
    # default ncore to max (also defaults ncore == 0)
    ncore = min(mp.cpu_count() if not ncore else ncore, axis_size)
    if nchunk is None:
        # calculate number of slices to send to each GPU
        chunk_size = axis_size // ncore
        leftover = axis_size % ncore
        sizes = np.ones(ncore, dtype=np.int) * chunk_size
        # evenly distribute leftover across workers
        sizes[:leftover] += 1
        offsets = np.zeros(ncore+1, dtype=np.int)
        offsets[1:] = np.cumsum(sizes)
        slcs = [np.s_[offsets[i]:offsets[i+1]] for i in range(offsets.shape[0]-1)]
    elif nchunk == 0:
        # nchunk == 0 is a special case, we will collapse the dimension
        slcs = [np.s_[i] for i in range(axis_size)]
    else:
        # calculate offsets based on chunk size
        slcs = [np.s_[offset:offset+nchunk] for offset in range(0, axis_size, nchunk)]
    return ncore, slcs


def get_worker_ncore_slices(axis_size, ncore=None, nchunk=None):
    # default ncore to max (also defaults ncore == 0)
    if not ncore:
        ncore = mp.cpu_count()
    if nchunk is None:
        # calculate number of slices to send to each GPU
        chunk_size = axis_size // ncore
        leftover = axis_size % ncore
        sizes = np.ones(ncore, dtype=np.int) * chunk_size
        # evenly distribute leftover across workers
        sizes[:leftover] += 1
        offsets = np.zeros(ncore+1, dtype=np.int)
        offsets[1:] = np.cumsum(sizes)
        slcs = [np.s_[offsets[i]:offsets[i+1]]
                for i in range(offsets.shape[0]-1)]
    elif nchunk == 0:
        # nchunk == 0 is a special case, we will collapse the dimension
        slcs = [np.s_[i] for i in range(axis_size)]
    else:
        # calculate offsets based on chunk size
        slcs = [np.s_[offset:offset+nchunk]
                for offset in range(0, axis_size, nchunk)]

    # create a barrier
    barrier()
    _size = get_nproc()
    if _size > 1:
        _nrank = get_rank()
        _nsplit = len(slcs) // _size
        _nmodulo = len(slcs) % _size
        _offset = _nsplit * _nrank
        if _nrank == 0:
            _nsplit += _nmodulo
        else:
            _offset += _nmodulo
        slcs = slcs[_offset:(_offset+_nsplit)]

    return ncore, slcs


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
    ncore, nchunk = get_ncore_nchunk(axis_size, ncore, nchunk)

    # prepare all args (func, args, kwargs)
    # NOTE: args will include shared_arr slice as first arg
    args = args or tuple()
    kwargs = kwargs or dict()

    # prepare global sharedmem arrays
    shared_arrays = []
    shared_shape = []
    shared_out = None
    shared_out_shape = None
    for arr in arrs:
        arr_shared = as_sharedmem(arr)
        shared_arrays.append(get_shared_mem(arr_shared))
        shared_shape.append(arr.shape)
        if out is not None and np.may_share_memory(arr, out) and \
                arr.shape == out.shape and arr.dtype == out.dtype:
            # assume these are the same array
            shared_out = arr_shared
    if out is None:
        # default out to last array in list
        out = shared_arrays[-1]
        shared_out = out
        out = to_numpy_array(out, arrs[-1].dtype, shared_shape[-1])
        shared_out_shape = shared_shape[-1]
        shared_out_type = arrs[-1].dtype
    else:
        shared_out_shape = out.shape
        shared_out_type = out.dtype
        shared_out = as_sharedmem(out)

    # Set up queue
    man = mp.Manager()
    queue = man.Queue()

    # if nchunk is zero, remove dimension from slice.
    map_args = []
    for i in range(0, axis_size, nchunk or 1):
        if nchunk:
            map_args.append((func, args, kwargs, np.s_[i:i + nchunk], axis))
        else:
            map_args.append((func, args, kwargs, i, axis))

    init_shared(shared_arrays, shared_out, arr.dtype, shared_shape,
                shared_out_type, shared_out_shape, queue=queue, on_host=True)

    if ncore > 1 and DEBUG is False:
        with closing(mp.Pool(processes=ncore,
                             initializer=init_shared,
                             initargs=(shared_arrays, shared_out, arr.dtype,
                                       shared_shape, shared_out_type,
                                       shared_out_shape, queue))) as p:
            if p._pool:
                proclist = p._pool[:]
                res = p.map_async(_arg_parser, map_args)
                try:
                    while not res.ready():
                        if any(proc.exitcode for proc in proclist):
                            p.terminate()
                            raise RuntimeError(
                                "Child process terminated before finishing")
                        res.wait(timeout=1)
                except KeyboardInterrupt:
                    p.terminate()
                    raise
            else:
                res = p.map_async(_arg_parser, map_args)
        try:
            p.close()
            res.get()
            p.join()
        except:
            p.terminate()
            raise

        clear_queue(queue, shared_arrays, shared_out)
    else:
        for m in map_args:
            _arg_parser(m)

    # NOTE: will only copy if out wasn't sharedmem
    out[:] = to_numpy_array(shared_out, shared_out_type, shared_out_shape)
    clear_shared()
    return out


def init_shared(
        shared_arrays, shared_out, intype, inshape,
        outtype, outshape, queue=None, on_host=False):
    global SHARED_ARRAYS
    global SHARED_OUT
    global SHARED_QUEUE
    global ON_HOST
    global INTYPE
    global OUTTYPE
    global INSHAPE
    global OUTSHAPE
    SHARED_ARRAYS = shared_arrays
    SHARED_OUT = shared_out
    SHARED_QUEUE = queue
    ON_HOST = on_host
    INTYPE = intype
    OUTTYPE = outtype
    INSHAPE = inshape
    OUTSHAPE = outshape


def clear_shared():
    global SHARED_ARRAYS
    global SHARED_OUT
    global SHARED_QUEUE
    global INTYPE
    global OUTTYPE
    SHARED_ARRAYS = None
    SHARED_OUT = None
    SHARED_QUEUE = None
    INTYPE = None
    OUTTYPE = None


def _arg_parser(params):
    global SHARED_ARRAYS
    global SHARED_OUT
    global SHARED_QUEUE
    global INTYPE
    global OUTTYPE
    global INSHAPE
    global OUTSHAPE
    func, args, kwargs, slc, axis = params
    func_args = tuple((slice_axis(to_numpy_array(a, INTYPE, INSHAPE[idx]), slc, axis) for idx, a in enumerate(SHARED_ARRAYS))) + args
    # NOTE: will only copy if actually different arrays
    try:
        result = func(*func_args, **kwargs)

        if result is not None and isinstance(result, np.ndarray):
            outslice = slice_axis(to_numpy_array(SHARED_OUT, OUTTYPE, OUTSHAPE), slc, axis)
            outslice[:] = result[:]
    except RunOnHostException:
        SHARED_QUEUE.put(params)

# apply slice to specific axis on ndarray


def slice_axis(arr, slc, axis):
    return arr[tuple(slice(None) if i != axis else slc for i in range(arr.ndim))]


def clear_queue(queue, shared_arrays, shared_out):
    while not queue.empty():
        params = queue.get(False)
        _arg_parser(params)


class RunOnHostException(Exception):
    pass


class set_numexpr_threads(object):

    def __init__(self, nthreads):
        cpu_count = mp.cpu_count()
        if nthreads is None or nthreads > cpu_count:
            self.n = cpu_count
        else:
            self.n = nthreads

    def __enter__(self):
        self.oldn = ne.set_num_threads(self.n)

    def __exit__(self, exc_type, exc_value, traceback):
        ne.set_num_threads(self.oldn)
