# -*- coding: utf-8 -*-
import numpy as np
from tomopy.dataio.reader import Dataset
from region_segment import region_segment
from threshold_segment import threshold_segment
import multiprocessing as mp
from tomopy.tools.multiprocess import distribute_jobs
import logging
logger = logging.getLogger("tomopy")

from scipy import ndimage
from skimage import morphology
from skimage.filter import sobel



def region_segment_wrapper(TomoObj, low, high,
                           num_cores=None, chunk_size=None):
    if not TomoObj.FLAG_DATA_RECON:
        logger.warning("region based segmentation (recon data missing) [bypassed]")
        return
    
    # Normalize data first.
    data = TomoObj.data_recon - TomoObj.data_recon.min()
    data /= data.max()
    
    # Distribute jobs.
    axis = 0 # Slice axis
    args = (low, high)
    TomoObj.data_recon = distribute_jobs(data, region_segment, args,
                                         axis, num_cores, chunk_size)

    # Update provenance.
    TomoObj.provenance['region_segment'] = {'low':low, 'high':high}

    logger.info("region based segmentation [ok]")


def threshold_segment_wrapper(TomoObj, cutoff=None,
                           num_cores=None, chunk_size=None):
    if not TomoObj.FLAG_DATA_RECON:
        logger.warning("threshold based segmentation (recon data missing) [bypassed]")
        return
    
    # Normalize data first.
    data = TomoObj.data_recon - TomoObj.data_recon.min()
    data /= data.max()

    if cutoff == None:
        cutoff = 0.5

    # Distribute jobs.
    axis = 0 # Slice axis
    args = (cutoff)
    TomoObj.data_recon = distribute_jobs(data, threshold_segment, args,
                                         axis, num_cores, chunk_size)
                                                      
    # Update provenance.
    TomoObj.provenance['threshold_segment'] = {'cutoff':cutoff}
    
    logger.info("threshold based segmentation [ok]")


setattr(Dataset, 'region_segment', region_segment_wrapper)
setattr(Dataset, 'threshold_segment', threshold_segment_wrapper)

region_segment_wrapper.__doc__ = region_segment.__doc__
threshold_segment_wrapper.__doc__ = threshold_segment.__doc__