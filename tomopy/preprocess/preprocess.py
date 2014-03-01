# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers for the other
modules in preprocess package to link them to TomoPy session. 
Each wrapper first checks the arguments and then calls the method.
The linking is mostly realized through the multiprocessing module.
"""
import numpy as np
import logging
logger = logging.getLogger("tomopy")

# Import main TomoPy object.
from tomopy.dataio.reader import Session

# Import available functons in the package.
from median_filter import _median_filter
from normalize import _normalize
from phase_retrieval import _phase_retrieval, _paganin_filter
from stripe_removal import _stripe_removal

# Import multiprocessing module.
from tomopy.tools.multiprocess import distribute_jobs


def median_filter(tomo, size=None, 
                  num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("median filtering (data " +
                       "missing) [bypassed]")
        return
        

    # Set default parameters.
    if size is None:
        size = 5
        logger.debug("median_filter: size is " +
                       "set to " + str(size) + " [ok]")
        
        
    # Distribute jobs.
    _func = _median_filter
    _args = (size)
    _axis = 1 # Slice axis
    tomo.data = distribute_jobs(tomo.data, 
                                _func, _args, _axis, 
                                num_cores, chunk_size)
   
    # Update provenance and log.
    tomo.provenance['median_filter'] = {'size':size}
    logger.info("median filtering [ok]")



def normalize(tomo, cutoff=None, 
              num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("normalization (data " +
                       "missing) [bypassed]")
        return
        
    if not tomo.FLAG_WHITE: # This is checked before but check anyway.
        logger.warning("normalization (white-data " +
                       "missing) [bypassed]")
        return
        
    if not tomo.FLAG_DARK: # This is checked before but check anyway.
        logger.warning("normalization (dark-data " +
                       "missing) [bypassed]")
        return
        

    # Set default parameters.
    if cutoff is None:
        logger.debug("normalize: cutoff is set to None [ok]")


    # Calculate average white and dark fields for normalization.
    avg_white = np.mean(tomo.data_white, axis=0)
    avg_dark = np.mean(tomo.data_dark, axis=0)
    
    
    # Distribute jobs.
    _func = _normalize
    _args = (avg_white, avg_dark, cutoff)
    _axis = 0 # Projection axis
    tomo.data = distribute_jobs(tomo.data, 
                                _func, _args, _axis, 
                                num_cores, chunk_size)


    # Update provenance and log.
    tomo.provenance['normalize'] = {'cutoff':cutoff}
    logger.info("normalization [ok]")



def phase_retrieval(tomo, pixel_size=None, dist=None, 
                    energy=None, alpha=None, padding=None,
                    num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("phase retrieval (data " +
                       "missing) [bypassed]")
        return
        
    if tomo.data.shape[1] < 16:
        logger.warning("phase retrieval (at least 16 " +
                       "slices are needed) [bypassed]")
        return
        
    if pixel_size is None:
        logger.warning("phase retrieval (pixel_size " +
                       "missing) [bypassed]")
        return
        
    if dist is None:
        logger.warning("phase retrieval (dist " +
                       "missing) [bypassed]")
        return
        
    if energy is None:
        logger.warning("phase retrieval (energy " +
                       "missing) [bypassed]")
        return
    

    # Set default parameters.
    if alpha is None:
        alpha = 1e-5
        logger.debug("phase_retrieval: alpha is set " +
                       "to " + str(alpha) + " [ok]")
  
    if padding is None:
        padding = True
        logger.debug("phase_retrieval: padding is set " +
                       "to " + str(padding) + " [ok]")
        
        
    # Compute the filter.
    H, x_shift, y_shift, tmp_proj = _paganin_filter(tomo.data,
                                    pixel_size, dist, energy, alpha, padding)
                                   
                     
    # Distribute jobs.
    _func = _phase_retrieval
    _args = (H, x_shift, y_shift, tmp_proj, padding)
    _axis = 0 # Projection axis
    tomo.data = distribute_jobs(tomo.data, 
                                _func, _args, _axis, 
                                num_cores, chunk_size)

    # Update provenance and log.
    tomo.provenance['phase_retrieval'] = {'pixel_size':pixel_size, 
	                                  'dist':dist,
                                          'energy':energy,
	                                  'alpha':alpha,
	                                  'padding':padding}
    logger.info("phase retrieval [ok]")



def stripe_removal(tomo, level=None, wname=None, sigma=None,
                   num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("stripe removal (data " +
                       "missing) [bypassed]")
        return
        
    # Set default parameters.
    if wname is None:
        wname = 'db5'
        logger.debug("stripe_removal wavelet is " +
                       "set to " + wname + " [ok]")

    if sigma is None:
        sigma = 4
        logger.debug("stripe_removal: sigma is " +
                       "set to " + str(sigma) + " [ok]")


    # Find the higest level possible.
    if level is None:
        size = np.max(tomo.data.shape)
        level = int(np.ceil(np.log2(size)))
        

    # Distribute jobs.
    _func = _stripe_removal
    _args = (level, wname, sigma)
    _axis = 1 # Slice axis
    tomo.data = distribute_jobs(tomo.data, 
                                _func, _args, _axis,
                                num_cores, chunk_size)
    

    # Update provenance and log.
    tomo.provenance['stripe_removal'] = {'level':level, 
                                         'wname':wname, 
                                         'sigma':sigma}
    logger.info("stripe removal [ok]")
    
    

# Hook all these methods to TomoPy.
setattr(Session, 'median_filter', median_filter)
setattr(Session, 'normalize', normalize)
setattr(Session, 'phase_retrieval', phase_retrieval)
setattr(Session, 'stripe_removal', stripe_removal)

# Use original function docstrings for the wrappers.
median_filter.__doc__ = _median_filter.__doc__
normalize.__doc__ = _normalize.__doc__
phase_retrieval.__doc__ = _phase_retrieval.__doc__
stripe_removal.__doc__ = _stripe_removal.__doc__
