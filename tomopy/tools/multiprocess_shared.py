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
    shared_arr = shared_arr_ # must be inhereted, not passed as an argument

def tonumpyarray(mp_arr, dshape):
    a = np.frombuffer(mp_arr.get_obj())
    return np.reshape(a, dshape)

# --------------------------------------------------------------------

def distribute_jobs(data, func, args, axis,
                    num_cores=None, chunk_size=None):

    # Arrange number of processors.
    if num_cores is None:
        num_cores = mp.cpu_count()
    dims = data.shape[axis]
    
    # Maximum number of available processors for the task.
    if dims < num_cores:
        num_cores = dims
    
    # Arrange chunk size.
    if chunk_size is None:
        chunk_size = dims / num_cores
    
    # Determine pool size.
    pool_size = dims / chunk_size + 1
    
    # Populate jobs.
    arg = []
    for m in range(pool_size):
        ind_start = m*chunk_size
        ind_end = (m+1)*chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims
            
        arg += [(range(ind_start, ind_end), data.shape, args)]

    shared_arr = mp.Array(ctypes.c_double, data.size) # takes time
    arr = tonumpyarray(shared_arr, data.shape)
    arr[:] = data

    # write to arr from different processes
    with closing(mp.Pool(initializer=init, initargs=(shared_arr,))) as p:
        p.map_async(func, arg)
    p.join()
    p.close()
    return arr.astype('float32', copy=False) # takes time to convert










