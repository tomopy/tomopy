# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers to 
hook the methods in preprocess package to X-ray 
absorption tomography data object.
"""

import numpy as np

# Import main TomoPy object.
from tomopy.xtomo.xtomo_dataset import XTomoDataset

# Import available functons in the package.
from tomopy.algorithms.preprocess.apply_padding import apply_padding
from tomopy.algorithms.preprocess.circular_roi import circular_roi
from tomopy.algorithms.preprocess.correct_drift import correct_drift
from tomopy.algorithms.preprocess.correct_tilt import correct_tilt
from tomopy.algorithms.preprocess.downsample import downsample2d, downsample3d
from tomopy.algorithms.preprocess.median_filter import median_filter
from tomopy.algorithms.preprocess.normalize import normalize
from tomopy.algorithms.preprocess.phase_retrieval import phase_retrieval
from tomopy.algorithms.preprocess.stripe_removal import stripe_removal
from tomopy.algorithms.preprocess.zinger_removal import zinger_removal

# Import multiprocessing module.
from tomopy.tools.multiprocess_shared import distribute_jobs

# --------------------------------------------------------------------

def _apply_padding(xtomo, num_pad=None,
                   num_cores=None, chunk_size=None,
                   overwrite=True):

    # Set default parameters.
    num_pixels = xtomo.data.shape[2]
    if num_pad is None:
        num_pad = np.ceil(num_pixels * np.sqrt(2))
    elif num_pad < num_pixels:
        num_pad = num_pixels
                         
    # Check input.
    if not isinstance(num_pad, np.int32):
        num_pad = np.array(num_pad, dtype='int32')

    data = apply_padding(xtomo.data, num_pad)
    
    # Update log.
    xtomo.logger.debug("apply_padding: num_pad: " + str(num_pad))
    xtomo.logger.info("apply_padding [ok]")

    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data

# --------------------------------------------------------------------

def _circular_roi(xtomo, ratio=1, overwrite=True):

    data = circular_roi(xtomo.data, ratio)
                                         
    # Update log.
    xtomo.logger.debug("circular_roi: ratio: " + str(ratio))
    xtomo.logger.info("circular_roi [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data

# --------------------------------------------------------------------

def _correct_drift(xtomo, air_pixels=20, 
                   num_cores=None, chunk_size=None,
                   overwrite=True):
    
    # Check input.
    if air_pixels <= 0:
        air_pixels = 0
    if not isinstance(air_pixels, np.int32):
        air_pixels = np.array(air_pixels, dtype='int32')
    
    data = correct_drift(xtomo.data, air_pixels)
   
    # Update log.
    xtomo.logger.debug("correct_drift: air_pixels: " + str(air_pixels))
    xtomo.logger.info("correct_drift [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data
    
# --------------------------------------------------------------------

def _correct_tilt(xtomo, angle=0, overwrite=True):

    data = correct_tilt(xtomo.data, angle)
    data_white = correct_tilt(xtomo.data_white, angle)
    data_dark = correct_tilt(xtomo.data_dark, angle)
                                         
    # Update log.
    xtomo.logger.debug("correct_tilt: ratio: " + str(angle))
    xtomo.logger.info("correct_tilt [ok]")
    
    # Update returned values.
    if overwrite: 
    	xtomo.data = data
    	xtomo.data_white = data
    	xtomo.data_dark = data
    else: return data, data_white, data_dark

# --------------------------------------------------------------------

def _downsample2d(xtomo, level=1,
                  num_cores=None, chunk_size=None,
                  overwrite=True):
    
    # Check input.
    if level < 0:
        level = 0
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data = downsample2d(xtomo.data, level)
    
    # Update log.
    xtomo.logger.debug("downsample2d: level: " + str(level))
    xtomo.logger.info("downsample2d [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data
	
# --------------------------------------------------------------------

def _downsample3d(xtomo, level=1,
                  num_cores=None, chunk_size=None,
                  overwrite=True):

    # Check input.
    if level < 0:
        level = 0
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data = downsample3d(xtomo.data, level)
    
    # Update log.
    xtomo.logger.debug("downsample3d: level: " + str(level))
    xtomo.logger.info("downsample3d [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data	

# --------------------------------------------------------------------

def _median_filter(xtomo, size=5, 
                   num_cores=None, chunk_size=None,
                   overwrite=True):
                  
    # Check input.
    if size < 1:
        size = 1
        
    # Distribute jobs.
    _func = median_filter
    _args = (size)
    _axis = 1 # Slice axis
    data = distribute_jobs(xtomo.data, _func, _args, _axis, 
                           num_cores, chunk_size)
   
    # Update log.
    xtomo.logger.debug("median_filter: size: " + str(size))
    xtomo.logger.info("median_filter [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data	

# --------------------------------------------------------------------

def _normalize(xtomo, cutoff=None, 
               num_cores=None, chunk_size=None,
               overwrite=True):

    # Calculate average white and dark fields for normalization.
    avg_white = np.mean(xtomo.data_white, axis=0)
    avg_dark = np.mean(xtomo.data_dark, axis=0)
    
    # Distribute jobs.
    _func = normalize
    _args = (avg_white, avg_dark, cutoff)
    _axis = 0 # Projection axis
    data = distribute_jobs(xtomo.data, _func, _args, _axis, 
			   num_cores, chunk_size)

    # Update log.
    xtomo.logger.debug("normalize: cutoff: " + str(cutoff))
    xtomo.logger.info("normalize [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data	

# --------------------------------------------------------------------

def _phase_retrieval(xtomo, pixel_size=1e-4, dist=50, 
                     energy=20, alpha=1e-4, padding=True,
                     num_cores=None, chunk_size=None,
                     overwrite=True):             

    # Distribute jobs.
    _func = phase_retrieval
    _args = (pixel_size, dist, energy, alpha, padding)
    _axis = 0 # Projection axis
    data = distribute_jobs(xtomo.data, _func, _args, _axis, 
                           num_cores, chunk_size)

    # Update log.
    xtomo.logger.debug("phase_retrieval: pixel_size: " + str(pixel_size))
    xtomo.logger.debug("phase_retrieval: dist: " + str(dist))
    xtomo.logger.debug("phase_retrieval: energy: " + str(energy))
    xtomo.logger.debug("phase_retrieval: alpha: " + str(alpha))
    xtomo.logger.debug("phase_retrieval: padding: " + str(padding))
    xtomo.logger.info("phase_retrieval [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data	

# --------------------------------------------------------------------

def _stripe_removal(xtomo, level=None, wname='db5', sigma=2, padding=False,
                    num_cores=None, chunk_size=None,
                    overwrite=True):

    # Find the higest level possible.
    if level is None:
        size = np.max(xtomo.data.shape)
        level = int(np.ceil(np.log2(size)))
        
    # Distribute jobs.
    _func = stripe_removal
    _args = (level, wname, sigma, padding)
    _axis = 1 # Slice axis
    data = distribute_jobs(xtomo.data, _func, _args, _axis,
                           num_cores, chunk_size)
			
    # Update log.
    xtomo.logger.debug("stripe_removal: level: " + str(level))
    xtomo.logger.debug("stripe_removal: wname: " + str(wname))
    xtomo.logger.debug("stripe_removal: sigma: " + str(sigma))
    xtomo.logger.debug("stripe_removal: padding: " + str(padding))
    xtomo.logger.info("stripe_removal [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data = data
    else: return data	

# --------------------------------------------------------------------

def _zinger_removal(xtomo, zinger_level=1000, median_width=3,
                    num_cores=None, chunk_size=None,
                    overwrite=True):

    # Distribute jobs.
    _func = zinger_removal
    _args = (zinger_level, median_width)
    _axis = 0 # Projection axis
    data = distribute_jobs(xtomo.data, _func, _args, _axis,
                           num_cores, chunk_size)

    data_white = distribute_jobs(xtomo.data_white, _func, _args, _axis,
                           num_cores, chunk_size)
    
    data_dark = distribute_jobs(xtomo.data_dark, _func, _args, _axis,
                           num_cores, chunk_size)

    # Update log.
    xtomo.logger.debug("zinger_removal: zinger_level: " + str(zinger_level))
    xtomo.logger.debug("zinger_removal: median_width: " + str(median_width))
    xtomo.logger.info("zinger_removal [ok]")

    # Update returned values.
    if overwrite:
        xtomo.data = data
        xtomo.data_white = data_white
        xtomo.data_dark = data_dark
    else: return data, data_white, data_dark

# --------------------------------------------------------------------
    
# Hook all these methods to TomoPy.
setattr(XTomoDataset, 'apply_padding', _apply_padding)
setattr(XTomoDataset, 'circular_roi', _circular_roi)
setattr(XTomoDataset, 'correct_drift', _correct_drift)
setattr(XTomoDataset, 'correct_tilt', _correct_tilt)
setattr(XTomoDataset, 'downsample2d', _downsample2d)
setattr(XTomoDataset, 'downsample3d', _downsample3d)
setattr(XTomoDataset, 'median_filter', _median_filter)
setattr(XTomoDataset, 'normalize', _normalize)
setattr(XTomoDataset, 'phase_retrieval', _phase_retrieval)
setattr(XTomoDataset, 'stripe_removal', _stripe_removal)
setattr(XTomoDataset, 'zinger_removal', _zinger_removal)
