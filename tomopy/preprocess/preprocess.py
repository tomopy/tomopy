# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from median_filter import median_filter
from normalize import normalize
import phase_retrieval
from stripe_removal import stripe_removal
import numpy as np
import multiprocessing as mp
from tomopy.tools.multiprocess import multiprocess
import logging
logger = logging.getLogger("tomopy")


def median_filter_wrapper(TomoObj, size=5,
                          num_processors=None, chunk_size=None):
    if not TomoObj.FLAG_DATA:
        logger.warning("median filtering (data missing) [bypassed]")
        return
        
    # Arrange number of processors.
    if num_processors is None:
        num_processors = mp.cpu_count()
    dims = TomoObj.data.shape[1]
    
    # Maximum number of available processors for the task.
    if dims < num_processors:
        num_processors = dims
    
    # Arrange chunk size.
    if chunk_size is None:
        chunk_size = dims / num_processors
        
    # Determine pool size.
    pool_size = dims / chunk_size + 1

    # Create multi-processing object.
    multip = multiprocess(median_filter,
                          num_processes=num_processors)
    
    # Populate jobs.
    for m in range(pool_size):
        ind_start = m*chunk_size
        ind_end = (m+1)*chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims
        args = (TomoObj.data[:, ind_start:ind_end, :], size, ind_start, ind_end)
        multip.add_job(args)
        
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[:, each[0]:each[1], :] = each[2]
    
    # Update provenance.
    TomoObj.provenance['median_filter'] = {'size':size}
    
    logger.info("median filtering [ok]")


def normalize_wrapper(TomoObj, cutoff=None,
                      num_processors=None, chunk_size=None):
    if not TomoObj.FLAG_DATA:
        logger.warning("normalization (data missing) [bypassed]")
        return
        
    if not TomoObj.FLAG_WHITE:
        logger.warning("normalization (white-data missing) [bypassed]")
        return
        
    if not TomoObj.FLAG_DARK:
        logger.warning("normalization (dark-data missing) [bypassed]")
        return

    # Calculate average white and dark fields for normalization.
    avg_white = np.mean(TomoObj.data_white, axis=0)
    avg_dark = np.mean(TomoObj.data_dark, axis=0)
    
    # Arrange number of processors.
    if num_processors is None:
        num_processors = mp.cpu_count()
    dims = TomoObj.data.shape[0]
    
    # Maximum number of available processors for the task.
    if dims < num_processors:
        num_processors = dims
    
    # Arrange chunk size.
    if chunk_size is None:
        chunk_size = dims / num_processors
        
    # Determine pool size.
    pool_size = dims / chunk_size + 1

    # Create multi-processing object.
    multip = multiprocess(normalize,
                          num_processes=num_processors)
	    
    # Populate jobs.
    for m in range(pool_size):
        ind_start = m*chunk_size
        ind_end = (m+1)*chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims
        args = (TomoObj.data[ind_start:ind_end, :, :],
                avg_white, avg_dark, cutoff, ind_start, ind_end)
        multip.add_job(args)
        
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[each[0]:each[1], :, :] = each[2]
    
    # Update provenance.
    TomoObj.provenance['normalize'] = {'cutoff':cutoff}
    
    logger.info("normalization [ok]")


def phase_retrieval_wrapper(TomoObj, pixel_size=None, dist=None, 
                            energy=None, alpha=1e-5, padding=True,
                            num_processors=None, chunk_size=None):
    if not TomoObj.FLAG_DATA:
        logger.warning("phase retrieval (data missing) [bypassed]")
        return
        
    if TomoObj.data.shape[1] < 16:
        logger.warning("phase retrieval (at least 16 slices are needed) [bypassed]")
        return
        
    if pixel_size is None:
        logger.warning("phase retrieval (pixel_size missing) [bypassed]")
        return
        
    if dist is None:
        logger.warning("phase retrieval (dist missing) [bypassed]")
        return
        
    if energy is None:
        logger.warning("phase retrieval (energy missing) [bypassed]")
        return
        
    # Compute the filter.
    H, x_shift, y_shift, tmp_data = phase_retrieval.paganin_filter(TomoObj.data, 
                                    pixel_size, dist, energy, alpha, padding)

    # Arrange number of processors.
    if num_processors is None:
        num_processors = mp.cpu_count()
    
    # Maximum number of available processors for the task.
    dims = TomoObj.data.shape[0]
    if dims < num_processors:
        num_processors = dims
    
    # Arrange chunk size.
    if chunk_size is None:
        chunk_size = dims / num_processors + 1
        
    # Determine pool size.
    pool_size = dims / chunk_size + 1

    # Create multi-processing object.
    multip = multiprocess(phase_retrieval.phase_retrieval,
                          num_processes=num_processors)

    # Populate jobs.
    for m in range(pool_size):
        ind_start = m*chunk_size
        ind_end = (m+1)*chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims
        args = (TomoObj.data[ind_start:ind_end, :, :],
                H, x_shift, y_shift, tmp_data, padding, ind_start, ind_end)
        multip.add_job(args)
    
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[each[0]:each[1], :, :] = each[2]
   	    
    # Update provenance.
    TomoObj.provenance['phase_retrieval'] = {'pixel_size':pixel_size, 
	                                     'dist':dist, 
	                                     'energy':energy, 
	                                     'alpha':alpha, 
	                                     'padding':padding}
	    
    logger.info("phase retrieval [ok]")


def stripe_removal_wrapper(TomoObj, level=None, wname='db5', sigma=2,
                           num_processors=None, chunk_size=None):
    if not TomoObj.FLAG_DATA:
        logger.warning("normalization (data missing) [bypassed]")
        return

    # Find the higest level possible.
    if level is None:
        size = np.max(TomoObj.data.shape)
        level = int(np.ceil(np.log2(size)))
        
    # Arrange number of processors.
    if num_processors is None:
        num_processors = mp.cpu_count()
    dims = TomoObj.data.shape[1]
    
    # Maximum number of available processors for the task.
    if dims < num_processors:
        num_processors = dims
    
    # Arrange chunk size.
    if chunk_size is None:
        chunk_size = dims / num_processors
        
    # Determine pool size.
    pool_size = dims / chunk_size + 1

    # Create multi-processing object.
    multip = multiprocess(stripe_removal,
                          num_processes=num_processors)
    
    # Populate jobs.
    for m in range(pool_size):
        ind_start = m*chunk_size
        ind_end = (m+1)*chunk_size
        if ind_start >= dims:
            break
        if ind_end > dims:
            ind_end = dims
        args = (TomoObj.data[:, ind_start:ind_end, :], 
                level, wname, sigma, ind_start, ind_end)
        multip.add_job(args)
        
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[:, each[0]:each[1], :] = each[2]
    
    # Update provenance.
    TomoObj.provenance['stripe_removal'] = {'level':level, 
                                            'wname':wname, 
                                            'sigma':sigma}
    
    logger.info("stripe removal [ok]")


setattr(Dataset, 'median_filter', median_filter_wrapper)
setattr(Dataset, 'normalize', normalize_wrapper)
setattr(Dataset, 'phase_retrieval', phase_retrieval_wrapper)
setattr(Dataset, 'stripe_removal', stripe_removal_wrapper)

median_filter_wrapper.__doc__ = median_filter.__doc__
normalize_wrapper.__doc__ = normalize.__doc__
phase_retrieval_wrapper.__doc__ = phase_retrieval.__doc__
stripe_removal_wrapper.__doc__ = stripe_removal.__doc__
