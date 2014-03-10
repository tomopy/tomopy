# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers for the other
modules in postprocess package to link them to TomoPy session. 
Each wrapper first checks the arguments and then calls the method.
The linking is mostly realized through the multiprocessing module.
"""
# Import main TomoPy object.
from tomopy.dataio.reader import Session

# Import available functons in the package.
from adaptive_segment import _adaptive_segment
from remove_background import _remove_background
from region_segment import _region_segment
from threshold_segment import _threshold_segment

# Import multiprocessing module.
from tomopy.tools.multiprocess import distribute_jobs


# --------------------------------------------------------------------

def adaptive_segment(tomo, block_size=None, offset=None,
                     num_cores=None, chunk_size=None,
                     overwrite=True):
                     
    # Make checks first. 
    if not tomo.FLAG_DATA_RECON:
        tomo.logger.warning("adaptive thresholding based segmentation " +
                            "(recon data missing) [bypassed]")
        return
    
    
    # Set default parameters.
    if block_size == None:
        block_size = 256
        tomo.logger.debug("adaptive_segment: block_size is " +
                          "set to " + str(block_size) + " [ok]")

    if offset == None:
        offset = 0
        tomo.logger.debug("adaptive_segment: offset is " +
                          "set to " + str(offset) + " [ok]")
    
    
    # Normalize data first.
    data = tomo.data_recon - tomo.data_recon.min()
    data /= data.max() 

    # Distribute jobs.
    _func = _adaptive_segment
    _args = (block_size, offset)
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)
                                         
    # Update provenance.
    tomo.provenance['adaptive_segment'] = {'block_size':block_size, 
                                           'offset':offset}
    tomo.logger.info("adaptive thresholding based segmentation [ok]")
    
    # Update returned values.
    if overwrite:
	tomo.data_recon = data_recon
    else:
	return data_recon

# --------------------------------------------------------------------

def region_segment(tomo, low=None, high=None,
                   num_cores=None, chunk_size=None,
                   overwrite=True):
                   
    # Make checks first. 
    if not tomo.FLAG_DATA_RECON:
        tomo.logger.warning("region based segmentation " +
                            "(recon data missing) [bypassed]")
        return

    if low is None:
        tomo.logger.warning("region based segmentation " +
                            "(low value for segmentation " +
                            "missing) [bypassed]")
        return
        
    if high is None:
        tomo.logger.warning("region based segmentation " +
                            "(high value for segmentation " +
                            "missing) [bypassed]")
        return
    
    # Normalize data first.
    data = tomo.data_recon - tomo.data_recon.min()
    data /= data.max()
    
    # Distribute jobs.
    _func = _region_segment
    _args = (low, high)
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)

    # Update provenance.
    tomo.provenance['region_segment'] = {'low':low, 'high':high}
    tomo.logger.info("region based segmentation [ok]")
    
    # Update returned values.
    if overwrite:
	tomo.data_recon = data_recon
    else:
	return data_recon

# --------------------------------------------------------------------

def remove_background(tomo, 
                      num_cores=None, chunk_size=None,
                      overwrite=True):
                      
    # Make checks first. 
    if not tomo.FLAG_DATA_RECON:
        tomo.logger.warning("background removal " +
                            "(recon data missing) [bypassed]")
        return
    
    # Distribute jobs.
    _func = _remove_background
    _args = ()
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)
                                         
    # Update provenance.
    tomo.provenance['remove_background'] = {}
    tomo.logger.info("background removal [ok]")
    
    # Update returned values.
    if overwrite:
	tomo.data_recon = data_recon
    else:
	return data_recon

# --------------------------------------------------------------------

def threshold_segment(tomo, cutoff=None,
                      num_cores=None, chunk_size=None,
                      overwrite=True):
                      
    # Make checks first. 
    if not tomo.FLAG_DATA_RECON:
        tomo.logger.warning("threshold based segmentation " +
                            "(recon data missing) [bypassed]")
        return
    
    # Normalize data first.
    data = tomo.data_recon - tomo.data_recon.min()
    data /= data.max()

    # Distribute jobs.
    _func = _threshold_segment
    _args = ()
    _axis = 0 # Slice axis
    data_recon = distribute_jobs(data, _func, _args, _axis, 
                                 num_cores, chunk_size)
                                                      
    # Update provenance.
    tomo.provenance['threshold_segment'] = {'cutoff':cutoff}
    tomo.logger.info("threshold based segmentation [ok]")
    
    # Update returned values.
    if overwrite:
	tomo.data_recon = data_recon
    else:
	return data_recon

# --------------------------------------------------------------------

# Hook all these methods to TomoPy.
setattr(Session, 'adaptive_segment', adaptive_segment)
setattr(Session, 'remove_background', remove_background)
setattr(Session, 'region_segment', threshold_segment)
setattr(Session, 'threshold_segment', threshold_segment)

# Use original function docstrings for the wrappers.
adaptive_segment.__doc__ = _adaptive_segment.__doc__
remove_background.__doc__ = _remove_background.__doc__
region_segment.__doc__ = _region_segment.__doc__
threshold_segment.__doc__ = _threshold_segment.__doc__