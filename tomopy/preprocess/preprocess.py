# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from funcs.median_filter import median_filter
from funcs.normalize import normalize
from funcs.phase_retrieval import phase_retrieval
from funcs.stripe_removal import stripe_removal
import numpy as np
import logging
logger = logging.getLogger(__name__)


def median_filter_wrapper(TomoObj, *args, **kwargs):
    logger.info("performing median filtering")
    for m in range(TomoObj.data.shape[2]):
        TomoObj.data[:, :, m] = median_filter(TomoObj.data[:, :, m], *args, **kwargs)

def normalize_wrapper(TomoObj, *args, **kwargs):
    logger.info("performing normalization")
    avg_white = np.mean(TomoObj.data_white, axis=0)
    for m in range(TomoObj.data.shape[0]):
        TomoObj.data[m, :, :] = normalize(TomoObj.data[m, :, :], avg_white, *args, **kwargs)

def phase_retrieval_wrapper(TomoObj, *args, **kwargs):
    logger.info("performing phase retrieval")
    for m in range(TomoObj.data.shape[0]):
        TomoObj.data[m, :, :] = phase_retrieval(TomoObj.data[m, :, :], *args, **kwargs)

def stripe_removal_wrapper(TomoObj, *args, **kwargs):
    logger.info("performing stripe removal")
    for m in range(TomoObj.data.shape[1]):
        TomoObj.data[:, m, :] = stripe_removal(TomoObj.data[:, m, :], *args, **kwargs)


setattr(Dataset, 'median_filter', median_filter_wrapper)
setattr(Dataset, 'normalize', normalize_wrapper)
setattr(Dataset, 'phase_retrieval', phase_retrieval_wrapper)
setattr(Dataset, 'stripe_removal', stripe_removal_wrapper)

median_filter_wrapper.__doc__ = median_filter.__doc__
normalize_wrapper.__doc__ = normalize.__doc__
phase_retrieval_wrapper.__doc__ = phase_retrieval.__doc__
stripe_removal_wrapper.__doc__ = stripe_removal.__doc__