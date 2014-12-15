# -*- coding: utf-8 -*-
"""
Module for multiprocessing.
"""
import numpy as np
import multiprocessing as mp
import ctypes
from contextlib import closing

# --------------------------------------------------------------------


def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_  # must be inhereted, not passed as an argument


def tonumpyarray(mp_arr, dshape):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    return np.reshape(a, dshape)

# --------------------------------------------------------------------


def distribute_jobs(data, func, args, axis,
                    num_cores=None, chunk_size=None):
    """
    Distribute 3-D shared-memory data in chunks into cores.

    Parameters
    ----------
    func : srt
        Name of the function to be parallelized.

    args : list
        Arguments to that function in a list.

    axis : scalar
        The dimension of data that the job distribution
        will be performed. Data dimensions are like
        [projections, slices, pixels], so, if axis=0
        projections will be distributed across processors
        (e.g. for phase retrieval). If axis=1, slices will be
        distributed across processors (e.g. for ring removal),
        and for axis=2, pixels will be distributed across
        processors(but this is rare).

    num_cores : scalar, optional
        Number of processor that will be assigned to jobs.
        If unspecisified maximum amount of processors will be used.

    chunk_size : scalar, optional
        Number of packet size for each processor.
        For example, if axis=0, and chunk_size=8, each processor
        gets 8 projections.if axis=1, and chunk_size=8, each processor
        gets 8 slices, etc. If unspecified, the whole data
        will be distributed to processors in equal chunks such that
        each processor will get a single job to do.

    Returns
    -------
    out : ndarray
        3-D output data after transformation.
    """

    # Arrange number of processors.
    if num_cores is None:
        num_cores = mp.cpu_count()
    dims = data.shape[axis]

    # Maximum number of available processors for the task.
    if dims < num_cores:
        num_cores = dims

    # Arrange chunk size.
    if chunk_size is None:
        chunk_size = (dims - 1) / num_cores + 1

    # Determine pool size.
    pool_size = dims / chunk_size + 1

    # Populate jobs.
    arg = []
    for m in range(pool_size):
        ind_start = m * chunk_size
        ind_end = (m + 1) * chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims

        arg += [(range(ind_start, ind_end), data.shape, args)]

    shared_arr = mp.Array(ctypes.c_float, data.size)  # takes time
    arr = tonumpyarray(shared_arr, data.shape)
    arr[:] = data

    # write to arr from different processes
    with closing(mp.Pool(initializer=init, initargs=(shared_arr,))) as p:
        p.map_async(func, arg)
    p.join()
    p.close()

    return arr
