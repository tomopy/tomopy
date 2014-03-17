# -*- coding: utf-8 -*-
"""
Module for multiprocessing.
"""
import numpy as np
import multiprocessing as mp

# --------------------------------------------------------------------

class Multiprocess(object):
    def __init__(self, target_func, num_cores=None):
        """
        Class to wrap multiprocessing arbitrary task.
    
        Call sequence:
        1) Instantiate a multiprocess object.
        2) Add jobs to job queue using add_job method.
        3) Call multiprocess objects close_out method.
        4) Deal with the results in whatever foul, depraved way you deem fit.
    
        """
        if num_cores is None:
            num_cores = mp.cpu_count()
            
        self.total_jobs = 0
        self.jobs = mp.JoinableQueue()
        self.results = mp.Queue()

        tup = (self.jobs, self.results)
        self.num_cores = num_cores
        self.p = [mp.Process(target=target_func, 
                             args=tup) for i in range(num_cores)]

        for process in self.p:
            process.start()

    def add_job(self, job):
        self.total_jobs += 1
        self.jobs.put(job)

    def close_out(self):
        # Add Poison Pills
        for i in range(self.num_cores):
            self.jobs.put((None,))

        completed_jobs = 0
        res_list = []
        while True:
            if not self.results.empty():
                res_list.append(self.results.get())
                completed_jobs += 1
            if completed_jobs==self.total_jobs:
                break
        
        self.jobs.join()
        self.jobs.close()
        self.results.close()

        for process in self.p:
            process.join()
            
        return res_list

# --------------------------------------------------------------------

def worker(func):
    """
    Decorator for multiprocessing tasks.
    """
    def worker_in(*args, **kwargs):
	#name = mp.current_process().name
        jobs_completed = 0
        jobs, results = args[0], args[1]
        while True:
            job_args = jobs.get()
            if job_args[0] is None: # Deal with Poison Pill
                #print '{}: Exiting. {:d} jobs completed.'.format(name, jobs_completed)
                jobs.task_done()
                break
            res = func(job_args)
            jobs_completed += 1
            jobs.task_done()
            results.put(res)
        return worker_in
    return worker_in

# --------------------------------------------------------------------

def distribute_jobs(data, func, args, axis, 
                    num_cores=None, chunk_size=None):
    """
    Distribute 3-D volume jobs in chunks into cores.
    
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
        chunk_size = dims / num_cores
    
    # Determine pool size.
    pool_size = dims / chunk_size + 1
    
    # Create multi-processing object.
    multip = Multiprocess(worker(func), num_cores=num_cores)
                          
    # Populate jobs.
    for m in range(pool_size):
        ind_start = m*chunk_size
        ind_end = (m+1)*chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims
            
        if not isinstance(ind_start, np.int32):
            ind_start = np.array(ind_start, dtype=np.int32, copy=False)
            
        if not isinstance(ind_end, np.int32):
            ind_end = np.array(ind_end, dtype=np.int32, copy=False)
        
        # Add to queue.
        if axis == 0:
            multip.add_job((data[ind_start:ind_end, :, :], args, ind_start, ind_end))
        elif axis == 1:
            multip.add_job((data[:, ind_start:ind_end, :], args, ind_start, ind_end))
        elif axis == 2:
            multip.add_job((data[:, :, ind_start:ind_end], args, ind_start, ind_end))

    # Collect results.
    for each in multip.close_out():
        if axis == 0:
            data[each[0]:each[1], :, :] = each[2]
        elif axis == 1:
            data[:, each[0]:each[1], :] = each[2]
        elif axis == 2:
            data[:, :, each[0]:each[1]] = each[2]
    return data













