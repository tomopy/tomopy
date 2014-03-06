# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers for the other
modules in preprocess package to link them to TomoPy session. 
Each wrapper first checks the arguments and then calls the method.
The linking is mostly realized through the multiprocessing module.
"""
import numpy as np

# Import main TomoPy object.
from tomopy.dataio.reader import Session

# Import available functons in the package.
from apply_padding import _apply_padding
from correct_drift import _correct_drift
from median_filter import _median_filter
from normalize import _normalize
from phase_retrieval import _phase_retrieval, _paganin_filter
from stripe_removal import _stripe_removal

# Import multiprocessing module.
from tomopy.tools.multiprocess import distribute_jobs


# --------------------------------------------------------------------

def apply_padding(tomo,
                  num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("apply padding to data (data " +
                       "missing) [bypassed]")
        return

    tomo.data = _apply_padding(tomo.data)

    # Update data dimensions.
    tomo.num_pixels = np.array(tomo.data.shape[2], dtype='int32')

    ## Distribute jobs.
    #_func = _apply_padding
    #_args = ()
    #_axis = 1 # Slice axis
    #tomo.data = distribute_jobs(tomo.data, 
    #                            _func, _args, _axis, 
    #                            num_cores, chunk_size)
   
    # Update provenance and log.
    tomo.provenance['apply_padding'] = {}
    tomo.logger.info("apply data padding [ok]")

# --------------------------------------------------------------------

def correct_drift(tomo, air_pixels=None, 
                  num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("data drift correction (data " +
                       "missing) [bypassed]")
        return
        
    # Set default parameters.
    if air_pixels is None:
        air_pixels = 20
        tomo.logger.debug("correct_data: num_air_pixels is " +
                       "set to " + str(air_pixels) + " [ok]")

    tomo.data = _correct_drift(tomo.data, air_pixels)

    ## Distribute jobs.
    #_func = _correct_drift
    #_args = ()
    #_axis = 1 # Slice axis
    #tomo.data = distribute_jobs(tomo.data, 
    #                            _func, _args, _axis, 
    #                            num_cores, chunk_size)
   
    # Update provenance and log.
    tomo.provenance['correct_data'] = {'air_pixels':air_pixels}
    tomo.logger.info("data drift correction [ok]")

# --------------------------------------------------------------------

def median_filter(tomo, size=None, 
                  num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("median filtering (data " +
                       "missing) [bypassed]")
        return
        

    # Set default parameters.
    if size is None:
        size = 5
        tomo.logger.debug("median_filter: size is " +
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
    tomo.logger.info("median filtering [ok]")

# --------------------------------------------------------------------

def normalize(tomo, cutoff=None, 
              num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("normalization (data " +
                       "missing) [bypassed]")
        return
        
    if not tomo.FLAG_WHITE: # This is checked before but check anyway.
        tomo.logger.warning("normalization (white-data " +
                       "missing) [bypassed]")
        return
        
    if not tomo.FLAG_DARK: # This is checked before but check anyway.
        tomo.logger.warning("normalization (dark-data " +
                       "missing) [bypassed]")
        return
        

    # Set default parameters.
    if cutoff is None:
        tomo.logger.debug("normalize: cutoff is set to None [ok]")


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
    tomo.logger.info("normalization [ok]")

# --------------------------------------------------------------------

def phase_retrieval(tomo, pixel_size=None, dist=None, 
                    energy=None, alpha=None, padding=None,
                    num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("phase retrieval (data " +
                       "missing) [bypassed]")
        return
        
    if tomo.data.shape[1] < 16:
        tomo.logger.warning("phase retrieval (at least 16 " +
                       "slices are needed) [bypassed]")
        return
        
    if pixel_size is None:
        tomo.logger.warning("phase retrieval (pixel_size " +
                       "missing) [bypassed]")
        return
        
    if dist is None:
        tomo.logger.warning("phase retrieval (dist " +
                       "missing) [bypassed]")
        return
        
    if energy is None:
        tomo.logger.warning("phase retrieval (energy " +
                       "missing) [bypassed]")
        return
    

    # Set default parameters.
    if alpha is None:
        alpha = 1e-5
        tomo.logger.debug("phase_retrieval: alpha is set " +
                       "to " + str(alpha) + " [ok]")
  
    if padding is None:
        padding = True
        tomo.logger.debug("phase_retrieval: padding is set " +
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
    tomo.logger.info("phase retrieval [ok]")

# --------------------------------------------------------------------

def stripe_removal(tomo, level=None, wname=None, sigma=None,
                   num_cores=None, chunk_size=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("stripe removal (data " +
                       "missing) [bypassed]")
        return
        
    # Set default parameters.
    if wname is None:
        wname = 'db5'
        tomo.logger.debug("stripe_removal wavelet is " +
                       "set to " + wname + " [ok]")

    if sigma is None:
        sigma = 4
        tomo.logger.debug("stripe_removal: sigma is " +
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
    tomo.logger.info("stripe removal [ok]")
    
# --------------------------------------------------------------------
    
# Hook all these methods to TomoPy.
setattr(Session, 'apply_padding', apply_padding)
setattr(Session, 'correct_drift', correct_drift)
setattr(Session, 'median_filter', median_filter)
setattr(Session, 'normalize', normalize)
setattr(Session, 'phase_retrieval', phase_retrieval)
setattr(Session, 'stripe_removal', stripe_removal)

# Use original function docstrings for the wrappers.
apply_padding.__doc__ = _apply_padding.__doc__
correct_drift.__doc__ = _correct_drift.__doc__
median_filter.__doc__ = _median_filter.__doc__
normalize.__doc__ = _normalize.__doc__
phase_retrieval.__doc__ = _phase_retrieval.__doc__
stripe_removal.__doc__ = _stripe_removal.__doc__
