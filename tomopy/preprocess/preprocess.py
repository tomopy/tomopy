# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from median_filter import median_filter
from normalize import normalize
from phase_retrieval import phase_retrieval
from stripe_removal import stripe_removal
import numpy as np
from tomopy.tools.multiprocess import multiprocess
import logging
logger = logging.getLogger("tomopy")


def median_filter_wrapper(TomoObj, size=5):
    if not TomoObj.FLAG_DATA:
        logger.warning("median filtering (data missing) [bypassed]")
        return
        
    # Create multi-processing object.
    multip = multiprocess(median_filter)
	    
    # Populate jobs.
    for m in range(TomoObj.data.shape[1]):
	args = (TomoObj.data[:, m, :], size, m)
        multip.add_job(args)
		
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[:, each[0], :] = each[1]
    
    # Update provenance.
    TomoObj.provenance['median_filter'] = {'size':size}
    
    logger.info("median filtering [ok]")


def normalize_wrapper(TomoObj, cutoff=None):
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
        
    # Create multi-processing object.
    multip = multiprocess(normalize)
	    
    # Populate jobs.
    for m in range(TomoObj.data.shape[0]):
	args = (TomoObj.data[m, :, :], avg_white, avg_dark, cutoff, m)
        multip.add_job(args)
		
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[each[0], :, :] = each[1]
    
    # Update provenance.
    TomoObj.provenance['normalize'] = {'cutoff':cutoff}
    
    logger.info("normalization [ok]")


def phase_retrieval_wrapper(TomoObj, pixel_size=None, dist=None, energy=None, alpha=1e-5, padding=True):
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
    
    # Create multi-processing object.
    multip = multiprocess(phase_retrieval)
	    
    # Populate jobs.
    for m in range(TomoObj.data.shape[0]):
	args = (TomoObj.data[m, :, :], pixel_size, dist, energy, alpha, padding, m)
        multip.add_job(args)
		
    # Collect results.
    for each in multip.close_out():
        TomoObj.data[each[0], :, :] = each[1]
   	    
    # Update provenance.
    TomoObj.provenance['phase_retrieval'] = {'pixel_size':pixel_size, 
	                                     'dist':dist, 
	                                     'energy':energy, 
	                                     'alpha':alpha, 
	                                     'padding':padding}
	    
    logger.info("phase retrieval [ok]")


def stripe_removal_wrapper(TomoObj, level=None, wname='db5', sigma=2):
    if not TomoObj.FLAG_DATA:
        logger.warning("normalization (data missing) [bypassed]")
        return

    # Find the higest level possible.
    if level is None:
        size = np.max(TomoObj.data.shape)
        level = int(np.ceil(np.log2(size)))
        
    # Create multi-processing object.
    multip = multiprocess(stripe_removal)

    # Populate jobs.
    for m in range(TomoObj.data.shape[1]):
        args = (TomoObj.data[:, m, :], level, wname, sigma, m)
        multip.add_job(args)

    # Collect results.
    for each in multip.close_out():
        TomoObj.data[:, each[0], :] = each[1]
    
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
