# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers to 
hook the methods in postprocess package to X-ray 
absorption tomography data object.
"""

# Import main TomoPy object.
from tomopy.xtomo.xtomo_dataset import XTomoDataset

# Import available functons in the package.
from tomopy.algorithms.postprocess.adaptive_segment import _adaptive_segment
from tomopy.algorithms.postprocess.apply_mask import _apply_mask
from tomopy.algorithms.postprocess.remove_background import _remove_background
from tomopy.algorithms.postprocess.region_segment import _region_segment
from tomopy.algorithms.postprocess.threshold_segment import _threshold_segment

# Import multiprocessing module.
from tomopy.tools.multiprocess import distribute_jobs


# --------------------------------------------------------------------

def adaptive_segment(xtomo, block_size=256, offset=0,
                     num_cores=None, chunk_size=None,
                     overwrite=True):    
    
    # Normalize data first.
    data = xtomo.data_recon - xtomo.data_recon.min()
    data /= data.max() 

    # Distribute jobs.
    _func = _adaptive_segment
    _args = (block_size, offset)
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)
                                         
    # Update log.
    xtomo.logger.debug("adaptive_segment: block_size: " + str(block_size))
    xtomo.logger.debug("adaptive_segment: offset: " + str(offset))
    xtomo.logger.info("adaptive_segment [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

def apply_mask(xtomo, ratio=1, overwrite=True):    
    
    # Normalize data first.
    data = xtomo.data_recon - xtomo.data_recon.min()
    data /= data.max() 

    # Distribute jobs.
    data_recon = _apply_mask(xtomo.data_recon, ratio)
                                         
    # Update log.
    xtomo.logger.debug("apply_mask: ratio: " + str(ratio))
    xtomo.logger.info("apply_mask [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

def region_segment(xtomo, low=None, high=None,
                   num_cores=None, chunk_size=None,
                   overwrite=True):
    
    # Normalize data first.
    data = xtomo.data_recon - xtomo.data_recon.min()
    data /= data.max()
    
    # Distribute jobs.
    _func = _region_segment
    _args = (low, high)
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)

    # Update provenance.
    xtomo.logger.debug("region_segment: low: " + str(low))
    xtomo.logger.debug("region_segment: high: " + str(high))
    xtomo.logger.info("region_segment [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

def remove_background(xtomo, 
                      num_cores=None, chunk_size=None,
                      overwrite=True):
    
    # Distribute jobs.
    _func = _remove_background
    _args = ()
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(xtomo.data_recon, _func, _args, _axis, 
                                 num_cores, chunk_size)
                                         
    # Update provenance.
    xtomo.logger.info("remove_background [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

def threshold_segment(xtomo, cutoff=None,
                      num_cores=None, chunk_size=None,
                      overwrite=True):
    
    # Normalize data first.
    data = xtomo.data_recon - xtomo.data_recon.min()
    data /= data.max()

    # Distribute jobs.
    _func = _threshold_segment
    _args = ()
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)
                                                      
    # Update provenance.
    xtomo.logger.debug("threshold_segment: cutoff: " + str(cutoff))
    xtomo.logger.info("threshold_segment [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

# Hook all these methods to TomoPy.
setattr(XTomoDataset, 'adaptive_segment', adaptive_segment)
setattr(XTomoDataset, 'apply_mask', apply_mask)
setattr(XTomoDataset, 'remove_background', remove_background)
setattr(XTomoDataset, 'region_segment', threshold_segment)
setattr(XTomoDataset, 'threshold_segment', threshold_segment)