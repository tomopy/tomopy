# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from median_filter import median_filter
from normalize import normalize
from phase_retrieval import phase_retrieval
from stripe_removal import stripe_removal
import numpy as np
import multiprocessing as mp
from tomopy.tools.multiprocess import multiprocess
import logging
logger = logging.getLogger("tomopy")

pool_size = mp.cpu_count()

def median_filter_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        multip = multiprocess(median_filter, num_processes=pool_size)
        for m in range(TomoObj.data.shape[2]):
            multip.add_job(TomoObj.data[:, :, m], *args, **kwargs)
	m = 0
        for each in multip.close_out():
            TomoObj.data[:, :, m] = each
	    m += 1
        TomoObj.provenance['median_filter'] = (args, kwargs)
        logger.info("median filtering [ok]")
    else:
        logger.warning("median filtering (data missing) [bypassed]")

def normalize_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA and TomoObj.FLAG_WHITE:
        avg_white = np.mean(TomoObj.data_white, axis=0)
        for m in range(TomoObj.data.shape[0]):
            TomoObj.data[m, :, :] = normalize(TomoObj.data[m, :, :], avg_white, *args, **kwargs)
        TomoObj.provenance['normalization'] = (args, kwargs)
        logger.info("normalization [ok]")
    else:
        logger.warning("normalization (data missing) [bypassed]")

def phase_retrieval_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        if TomoObj.data.shape[1] >= 16:
	    multip = multiprocess(phase_retrieval, num_processes=pool_size)
	    for m in range(TomoObj.data.shape[0]):
		multip.add_job(TomoObj.data[m, :, :], *args, **kwargs)
	    m = 0
            for each in multip.close_out():
		TomoObj.data[m, :, :] = each
		m += 1
	    TomoObj.provenance['phase_retrieval'] = (args, kwargs)
	    logger.info("phase retrieval [ok]")
        else:
            logger.info("phase retrieval (at least 16 slices needed) [bypassed]")
    else:
        logger.info("phase retrieval (data missing) [bypassed]")

def stripe_removal_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
	multip = multiprocess(stripe_removal, num_processes=pool_size)
	for m in range(TomoObj.data.shape[1]):
	    multip.add_job(TomoObj.data[:, m, :])
        m = 0
	for each in multip.close_out():
	    TomoObj.data[:, m, :] = each
            m += 1
        TomoObj.provenance['stripe_removal'] = (args, kwargs)
        logger.info("stripe removal [ok]")
    else:
        logger.warning("stripe removal (data missing) [bypassed]")


setattr(Dataset, 'median_filter', median_filter_wrapper)
setattr(Dataset, 'normalize', normalize_wrapper)
setattr(Dataset, 'phase_retrieval', phase_retrieval_wrapper)
setattr(Dataset, 'stripe_removal', stripe_removal_wrapper)

median_filter_wrapper.__doc__ = median_filter.__doc__
normalize_wrapper.__doc__ = normalize.__doc__
phase_retrieval_wrapper.__doc__ = phase_retrieval.__doc__
stripe_removal_wrapper.__doc__ = stripe_removal.__doc__