# -*- coding: utf-8 -*-
import numpy as np
from tomopy.dataio.reader import Dataset
from region_segment import region_segment
import multiprocessing as mp
from tomopy.tools import multiprocess
import logging
logger = logging.getLogger("tomopy")

from scipy import ndimage
from skimage import morphology
from skimage.filter import sobel



def region_segment_wrapper(TomoObj, low, high,
                           num_cores=None, chunk_size=None):
    if not TomoObj.FLAG_DATA_RECON:
        logger.warning("segmentation (data missing) [bypassed]")
        return
    
    # Distribute jobs.
    axis = 0 # Slice axis
    args = (low, high)
    data = TomoObj.data_recon - np.min(TomoObj.data_recon)
    data /= np.max(data)
    
    TomoObj.data_recon = multiprocess.distribute_jobs(data,
                                 region_segment, args,
                                 axis, num_cores, chunk_size)

    # Update provenance.
    TomoObj.provenance['region_segment'] = {'low':low, 'high':high}

    logger.info("region based segmentation [ok]")


setattr(Dataset, 'region_segment', region_segment_wrapper)

region_segment_wrapper.__doc__ = region_segment.__doc__