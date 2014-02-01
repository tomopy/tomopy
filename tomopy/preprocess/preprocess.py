# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from median_filter import median_filter
from normalize import normalize
from phase_retrieval import phase_retrieval
from stripe_removal import stripe_removal
import numpy as np
import logging
logger = logging.getLogger("tomopy")


def median_filter_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        for m in range(TomoObj.data.shape[2]):
            TomoObj.data[:, :, m] = median_filter(TomoObj.data[:, :, m], *args, **kwargs)
        logger.info("median filtering [ok]")
    else:
        logger.info("median filtering [bypassed]")

def normalize_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA and TomoObj.FLAG_WHITE:
        avg_white = np.mean(TomoObj.data_white, axis=0)
        for m in range(TomoObj.data.shape[0]):
            TomoObj.data[m, :, :] = normalize(TomoObj.data[m, :, :], avg_white, *args, **kwargs)
        logger.info("normalization [ok]")
    else:
        logger.info("normalization [bypassed]")

def phase_retrieval_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        for m in range(TomoObj.data.shape[0]):
            TomoObj.data[m, :, :] = phase_retrieval(TomoObj.data[m, :, :], *args, **kwargs)
        logger.info("phase retrieval [ok]")
    else:
        logger.info("phase retrieval [bypassed]")

def stripe_removal_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        for m in range(TomoObj.data.shape[1]):
            TomoObj.data[:, m, :] = stripe_removal(TomoObj.data[:, m, :], *args, **kwargs)
        logger.info("stripe removal [ok]")
    else:
        logger.info("stripe removal [bypassed]")


setattr(Dataset, 'median_filter', median_filter_wrapper)
setattr(Dataset, 'normalize', normalize_wrapper)
setattr(Dataset, 'phase_retrieval', phase_retrieval_wrapper)
setattr(Dataset, 'stripe_removal', stripe_removal_wrapper)

median_filter_wrapper.__doc__ = median_filter.__doc__
normalize_wrapper.__doc__ = normalize.__doc__
phase_retrieval_wrapper.__doc__ = phase_retrieval.__doc__
stripe_removal_wrapper.__doc__ = stripe_removal.__doc__