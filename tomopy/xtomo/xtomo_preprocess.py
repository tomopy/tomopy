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
from tomopy.algorithms.preprocess.correct_fov import correct_fov
from tomopy.algorithms.preprocess.correct_tilt import correct_tilt
from tomopy.algorithms.preprocess.downsample import downsample2d, downsample3d
from tomopy.algorithms.preprocess.focus_region import focus_region
from tomopy.algorithms.preprocess.median_filter import median_filter
from tomopy.algorithms.preprocess.normalize import normalize
from tomopy.algorithms.preprocess.phase_retrieval import phase_retrieval
from tomopy.algorithms.preprocess.stripe_removal import stripe_removal
from tomopy.algorithms.preprocess.zinger_removal import zinger_removal
from tomopy.algorithms.preprocess.stripe_removal2 import stripe_removal2

# Import multiprocessing module.
from tomopy.tools.multiprocess_shared import distribute_jobs

# --------------------------------------------------------------------

def _apply_padding(self, num_pad=None, pad_val=0.,
                   num_cores=None, chunk_size=None,
                   overwrite=True):

    # Set default parameters.
    num_pixels = self.data.shape[2]
    if num_pad is None:
        num_pad = np.ceil(num_pixels * np.sqrt(2))
    elif num_pad < num_pixels:
        num_pad = num_pixels
                         
    # Check input.
    if not isinstance(num_pad, np.int32):
        num_pad = np.array(num_pad, dtype='int32')

    data = apply_padding(self.data, num_pad, pad_val)
    data_white = apply_padding(self.data_white, num_pad, pad_val)
    data_dark = apply_padding(self.data_dark, num_pad, pad_val)
    
    # Update log.
    self.logger.debug("apply_padding: num_pad: " + str(num_pad))
    self.logger.debug("apply_padding: pad_val: " + str(pad_val))
    self.logger.info("apply_padding [ok]")

    # Update returned values.
    if overwrite:
        self.data = data
        self.data_white = data_white
        self.data_dark = data_dark
    else: return data, data_white, data_dark


# --------------------------------------------------------------------

def _circular_roi(self, ratio=1, overwrite=True):

    data = circular_roi(self.data, ratio)
                                         
    # Update log.
    self.logger.debug("circular_roi: ratio: " + str(ratio))
    self.logger.info("circular_roi [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data

# --------------------------------------------------------------------

def _correct_drift(self, air_pixels=20, 
                   num_cores=None, chunk_size=None,
                   overwrite=True):
    
    # Check input.
    if air_pixels <= 0:
        air_pixels = 0
    if not isinstance(air_pixels, np.int32):
        air_pixels = np.array(air_pixels, dtype='int32')
    
    data = correct_drift(self.data, air_pixels)
   
    # Update log.
    self.logger.debug("correct_drift: air_pixels: " + str(air_pixels))
    self.logger.info("correct_drift [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data

# --------------------------------------------------------------------

def _correct_fov(self, num_overlap_pixels=None, overwrite=True):
    
    data = correct_fov(self.data, num_overlap_pixels)
    
    # Update log.
    self.logger.debug("correct_fov: num_overlap_pixels: " + str(num_overlap_pixels))
    self.logger.info("correct_fov [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data

# --------------------------------------------------------------------

def _correct_tilt(self, angle=0, overwrite=True):

    data = correct_tilt(self.data, angle)
    data_white = correct_tilt(self.data_white, angle)
    data_dark = correct_tilt(self.data_dark, angle)
                                         
    # Update log.
    self.logger.debug("correct_tilt: ratio: " + str(angle))
    self.logger.info("correct_tilt [ok]")
    
    # Update returned values.
    if overwrite: 
    	self.data = data
    	self.data_white = data
    	self.data_dark = data
    else: return data, data_white, data_dark

# --------------------------------------------------------------------

def _diagnose(self):
    
    # Update log.
    self.logger.debug("diagnose: data: shape: " + str(self.data.shape))
    self.logger.debug("diagnose: data: dtype: " + str(self.data.dtype))
    self.logger.debug("diagnose: data: size: %.2fMB", self.data.nbytes*9.53674e-7)
    self.logger.debug("diagnose: data: nans: " + str(np.sum(np.isnan(self.data))))
    self.logger.debug("diagnose: data: -inf: " + str(np.sum(np.isneginf(self.data))))
    self.logger.debug("diagnose: data: +inf: " + str(np.sum(np.isposinf(self.data))))
    self.logger.debug("diagnose: data: positives: " + str(np.sum(self.data>0)))
    self.logger.debug("diagnose: data: negatives: " + str(np.sum(self.data<0)))
    self.logger.debug("diagnose: data: mean: " + str(np.mean(self.data)))
    self.logger.debug("diagnose: data: min: " + str(np.min(self.data)))
    self.logger.debug("diagnose: data: max: " + str(np.max(self.data)))
    
    self.logger.debug("diagnose: data_white: shape: " + str(self.data_white.shape))
    self.logger.debug("diagnose: data_white: dtype: " + str(self.data_white.dtype))
    self.logger.debug("diagnose: data_white: size: %.2fMB", self.data_white.nbytes*9.53674e-7)
    self.logger.debug("diagnose: data_white: nans: " + str(np.sum(np.isnan(self.data_white))))
    self.logger.debug("diagnose: data_white: -inf: " + str(np.sum(np.isneginf(self.data_white))))
    self.logger.debug("diagnose: data_white: +inf: " + str(np.sum(np.isposinf(self.data_white))))
    self.logger.debug("diagnose: data_white: positives: " + str(np.sum(self.data_white>0)))
    self.logger.debug("diagnose: data_white: negatives: " + str(np.sum(self.data_white<0)))
    self.logger.debug("diagnose: data_white: mean: " + str(np.mean(self.data_white)))
    self.logger.debug("diagnose: data_white: min: " + str(np.min(self.data_white)))
    self.logger.debug("diagnose: data_white: max: " + str(np.max(self.data_white)))
    
    self.logger.debug("diagnose: data_dark: shape: " + str(self.data_dark.shape))
    self.logger.debug("diagnose: data_dark: dtype: " + str(self.data_dark.dtype))
    self.logger.debug("diagnose: data_dark: size: %.2fMB", self.data_dark.nbytes*9.53674e-7)
    self.logger.debug("diagnose: data_dark: nans: " + str(np.sum(np.isnan(self.data_dark))))
    self.logger.debug("diagnose: data_dark: -inf: " + str(np.sum(np.isneginf(self.data_dark))))
    self.logger.debug("diagnose: data_dark: +inf: " + str(np.sum(np.isposinf(self.data_dark))))
    self.logger.debug("diagnose: data_dark: positives: " + str(np.sum(self.data_dark>0)))
    self.logger.debug("diagnose: data_dark: negatives: " + str(np.sum(self.data_dark<0)))
    self.logger.debug("diagnose: data_dark: mean: " + str(np.mean(self.data_dark)))
    self.logger.debug("diagnose: data_dark: min: " + str(np.min(self.data_dark)))
    self.logger.debug("diagnose: data_dark: max: " + str(np.max(self.data_dark)))
    
    self.logger.debug("diagnose: theta: shape: " + str(self.theta.shape))
    self.logger.debug("diagnose: theta: dtype: " + str(self.theta.dtype))
    self.logger.debug("diagnose: theta: size: %.2fMB", self.theta.nbytes*9.53674e-7)
    self.logger.debug("diagnose: theta: nans: " + str(np.sum(np.isnan(self.theta))))
    self.logger.debug("diagnose: theta: -inf: " + str(np.sum(np.isneginf(self.theta))))
    self.logger.debug("diagnose: theta: +inf: " + str(np.sum(np.isposinf(self.theta))))
    self.logger.debug("diagnose: theta: positives: " + str(np.sum(self.theta>0)))
    self.logger.debug("diagnose: theta: negatives: " + str(np.sum(self.theta<0)))
    self.logger.debug("diagnose: theta: mean: " + str(np.mean(self.theta)))
    self.logger.debug("diagnose: theta: min: " + str(np.min(self.theta)))
    self.logger.debug("diagnose: theta: max: " + str(np.max(self.theta)))
    
    self.logger.info("diagnose [ok]")

# --------------------------------------------------------------------

def _downsample2d(self, level=1,
                  num_cores=None, chunk_size=None,
                  overwrite=True):
    
    # Check input.
    if level < 0:
        level = 0
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data = downsample2d(self.data, level)
    
    # Update log.
    self.logger.debug("downsample2d: level: " + str(level))
    self.logger.info("downsample2d [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data
	
# --------------------------------------------------------------------

def _downsample3d(self, level=1,
                  num_cores=None, chunk_size=None,
                  overwrite=True):

    # Check input.
    if level < 0:
        level = 0
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data = downsample3d(self.data, level)
    
    # Update log.
    self.logger.debug("downsample3d: level: " + str(level))
    self.logger.info("downsample3d [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data	
	
# --------------------------------------------------------------------

def _focus_region(self, xcoord, ycoord, diameter, 
                  padded=False, correction=True,
                  overwrite=True):
         
    # Check input.         
    if not isinstance(xcoord, np.float):
        xcoord = np.array(xcoord, dtype='float')
    if not isinstance(ycoord, np.float):
        ycoord = np.array(ycoord, dtype='float')
    if not isinstance(diameter, np.float):
        diameter = np.array(diameter, dtype='float')

    data = focus_region(self.data, xcoord, ycoord, diameter, 
                        self.center, padded, correction)

    # Adjust center
    if padded is False: center = data.shape[2]/2.
    else: center = self.center
    
    # Update log.
    self.logger.debug("focus_region: xcoord: " + str(xcoord))
    self.logger.debug("focus_region: ycoord: " + str(ycoord))
    self.logger.debug("focus_region: diameter: " + str(diameter))
    self.logger.debug("focus_region: center: " + str(self.center))
    self.logger.debug("focus_region: padded: " + str(padded))
    self.logger.debug("focus_region: correction: " + str(correction))
    self.logger.info("focus_region [ok]")
    
    # Update returned values.
    if overwrite: 
        self.data = data
        self.center = center
    else: return data, center

# --------------------------------------------------------------------

def _median_filter(self, size=5, axis=1,
                   num_cores=None, chunk_size=None,
                   overwrite=True):
                  
    # Check input.
    if size < 1:
        size = 1
        
    # Distribute jobs.
    _func = median_filter
    _args = (size, axis)
    _axis = axis
    data = distribute_jobs(self.data, _func, _args, _axis, 
                           num_cores, chunk_size)
   
    # Update log.
    self.logger.debug("median_filter: size: " + str(size))
    self.logger.debug("median_filter: axis: " + str(axis))
    self.logger.info("median_filter [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data	

# --------------------------------------------------------------------

def _normalize(self, cutoff=None, negvals=1,
               num_cores=None, chunk_size=None,
               overwrite=True):

    # Calculate average white and dark fields for normalization.
    avg_white = np.mean(self.data_white, axis=0)
    avg_dark = np.mean(self.data_dark, axis=0)
    
    # Distribute jobs.
    _func = normalize
    _args = (avg_white, avg_dark, cutoff, negvals)
    _axis = 0 # Projection axis
    data = distribute_jobs(self.data, _func, _args, _axis, 
			   num_cores, chunk_size)

    # Update log.
    self.logger.debug("normalize: cutoff: " + str(cutoff))
    self.logger.debug("normalize: negvals: " + str(negvals))
    self.logger.info("normalize [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data	

# --------------------------------------------------------------------

def _phase_retrieval(self, pixel_size=1e-4, dist=50, 
                     energy=20, alpha=1e-4, padding=True,
                     num_cores=None, chunk_size=None,
                     overwrite=True):             

    # Distribute jobs.
    _func = phase_retrieval
    _args = (pixel_size, dist, energy, alpha, padding)
    _axis = 0 # Projection axis
    data = distribute_jobs(self.data, _func, _args, _axis, 
                           num_cores, chunk_size)

    # Update log.
    self.logger.debug("phase_retrieval: pixel_size: " + str(pixel_size))
    self.logger.debug("phase_retrieval: dist: " + str(dist))
    self.logger.debug("phase_retrieval: energy: " + str(energy))
    self.logger.debug("phase_retrieval: alpha: " + str(alpha))
    self.logger.debug("phase_retrieval: padding: " + str(padding))
    self.logger.info("phase_retrieval [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data	

# --------------------------------------------------------------------

def _stripe_removal(self, level=None, wname='db5', sigma=2, padding=False,
                    num_cores=None, chunk_size=None,
                    overwrite=True):

    # Find the higest level possible.
    if level is None:
        size = np.max(self.data.shape)
        level = int(np.ceil(np.log2(size)))
        
    # Distribute jobs.
    _func = stripe_removal
    _args = (level, wname, sigma, padding)
    _axis = 1 # Slice axis
    data = distribute_jobs(self.data, _func, _args, _axis,
                           num_cores, chunk_size)
			
    # Update log.
    self.logger.debug("stripe_removal: level: " + str(level))
    self.logger.debug("stripe_removal: wname: " + str(wname))
    self.logger.debug("stripe_removal: sigma: " + str(sigma))
    self.logger.debug("stripe_removal: padding: " + str(padding))
    self.logger.info("stripe_removal [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data	

# --------------------------------------------------------------------

def _zinger_removal(self, zinger_level=1000, median_width=3,
                    num_cores=None, chunk_size=None,
                    overwrite=True):

    # Distribute jobs.
    _func = zinger_removal
    _args = (zinger_level, median_width)
    _axis = 0 # Projection axis
    data = distribute_jobs(self.data, _func, _args, _axis,
                           num_cores, chunk_size)

    data_white = distribute_jobs(self.data_white, _func, _args, _axis,
                           num_cores, chunk_size)
    
    data_dark = distribute_jobs(self.data_dark, _func, _args, _axis,
                           num_cores, chunk_size)

    # Update log.
    self.logger.debug("zinger_removal: zinger_level: " + str(zinger_level))
    self.logger.debug("zinger_removal: median_width: " + str(median_width))
    self.logger.info("zinger_removal [ok]")

    # Update returned values.
    if overwrite:
        self.data = data
        self.data_white = data_white
        self.data_dark = data_dark
    else: return data, data_white, data_dark

# --------------------------------------------------------------------

def _stripe_removal2(self, nblocks=0, alpha=1.5, num_cores=None, 
                    chunk_size=None,
                    overwrite=True):

        
    # Distribute jobs.
    _func = stripe_removal2
    _args = (nblocks, alpha)
    _axis = 1 # Slice axis

    data = distribute_jobs(self.data, _func, _args, _axis,
                           num_cores, chunk_size)
			
    # Update log.
    self.logger.debug("stripe_removal2: nblocks: " + str(nblocks))
    self.logger.debug("stripe_removal2: alpha: " + str(alpha))
    self.logger.info("stripe_removal2 [ok]")
    
    # Update returned values.
    if overwrite: self.data = data
    else: return data	

# --------------------------------------------------------------------
    
# Hook all these methods to TomoPy.
setattr(XTomoDataset, 'apply_padding', _apply_padding)
setattr(XTomoDataset, 'circular_roi', _circular_roi)
setattr(XTomoDataset, 'correct_drift', _correct_drift)
setattr(XTomoDataset, 'correct_fov', _correct_fov)
setattr(XTomoDataset, 'correct_tilt', _correct_tilt)
setattr(XTomoDataset, 'diagnose', _diagnose)
setattr(XTomoDataset, 'downsample2d', _downsample2d)
setattr(XTomoDataset, 'downsample3d', _downsample3d)
setattr(XTomoDataset, 'focus_region', _focus_region)
setattr(XTomoDataset, 'median_filter', _median_filter)
setattr(XTomoDataset, 'normalize', _normalize)
setattr(XTomoDataset, 'phase_retrieval', _phase_retrieval)
setattr(XTomoDataset, 'stripe_removal', _stripe_removal)
setattr(XTomoDataset, 'zinger_removal', _zinger_removal)
setattr(XTomoDataset, 'stripe_removal2', _stripe_removal2)

