# -*- coding: utf-8 -*-

"""
This module containes a set of thin wrappers to hook the methods in
preprocess package to X-ray absorption tomography data object.
"""

import numpy as np
import scipy as sp
import ipdb
from numpy import pad
import scipy.ndimage as spn
import os
import matplotlib.pyplot as plt
from matplotlib import animation

if hasattr(plt, 'style'):
    plt.style.use('ggplot')

# Import main TomoPy object.
from tomopy.xftomo.xftomo_dataset import XFTomoDataset

# Import available functons in the package.
from tomopy.algorithms.preprocess.align_projections import align_projections
from tomopy.algorithms.preprocess.median_filter import median_filter
from tomopy.algorithms.preprocess.zinger_removal import zinger_removal
from tomopy.algorithms.preprocess.to_gif import to_gif

# Import multiprocessing module.
from tomopy.tools.multiprocess_shared import distribute_jobs

# --------------------------------------------------------------------

def _apply_padding(self, num_pad=None, pad_val=0.,
                   num_cores=None, chunk_size=None,
                   overwrite=True):
    # Set default parameters.
    num_slices = self.data.shape[2]
    num_pixels = self.data.shape[3]
    if num_pad is None:
        num_pad = np.ceil(num_pixels * np.sqrt(2))
    elif num_pad < num_pixels:
        num_pad = num_pixels

    # Check input.
    if not isinstance(num_pad, np.int32):
        num_pad = np.array(num_pad, dtype='int32')

    x0 = int((num_pad-num_slices)/2)
    x1 = int(num_pad-num_slices-x0)
    y0 = int((num_pad-num_pixels)/2)
    y1 = int(num_pad-num_pixels-y0)

    data = pad(self.data, mode='constant', constant_values=(pad_val,),
        pad_width=((0,0),(0,0),(x0,x1),(y0,y1)))

    # Update log.
    self.logger.debug("apply_padding: num_pad: " + str(num_pad))
    self.logger.debug("apply_padding: pad_val: " + str(pad_val))
    self.logger.info("apply_padding [ok]")

    # Update returned values.
    if overwrite:
        self.data = data
    else:
        return data


# --------------------------------------------------------------------


def _align_projections(self, align_to_channel=None, method='rotation_and_scale_invariant_phase_correlation', output_gifs=False, output_filename='/tmp/projections.gif', overwrite=True):

    if align_to_channel:
        data = self.data[align_to_channel,:,:,:]
    else:
        data = np.sum(self.data,axis=0)

    if output_gifs:
        unaligned_data=data

    # Zinger removal
    data = distribute_jobs(data, zinger_removal, (10000, 3), 0, None, None)

    # Edge detection filter
    if method not in ['least_squares_fit']:
        for i in range(data.shape[0]):
            data[i,:,:] = np.hypot(spn.sobel(data[i,:,:], 0), spn.sobel(data[i,:,:], 1))
            data[i,:,:] = spn.median_filter(data[i,:,:], 3)

    data, translations = align_projections(data, method=method, theta=self.theta)

    if output_gifs:
        to_gif([unaligned_data, data], output_filename=output_filename)

        self.logger.debug('projection alignment gifs written: {:s}'.format(output_filename))

    data=np.zeros_like(self.data)
    for channel in range(self.data.shape[0]):
        data[channel,:,:,:], shifts = align_projections(self.data[channel,:,:,:], compute_alignment=False, alignment_translations=translations)

    # Update log.
    self.logger.debug("aligned projections using: {:s}".format(method))
    self.logger.info("aligned_projections[ok]")

    # Update returned values.
    if overwrite:
        self.data = data
        self.alignment_translations = translations
    else:
        return data, translations


# --------------------------------------------------------------------

def _zinger_removal(self, zinger_level=10000, median_width=3,
                    num_cores=None, chunk_size=None,
                    overwrite=True):
    # Distribute jobs.
    _func = zinger_removal
    _args = (zinger_level, median_width)
    _axis = 0  # Projection axis
    data=np.zeros_like(self.data)
    for channel in range(self.data.shape[0]):
        data[channel,:,:,:] = distribute_jobs(self.data[channel,:,:,:], _func, _args, _axis,
                               num_cores, chunk_size)

    # Update log.
    self.logger.debug("zinger_removal: zinger_level: " + str(zinger_level))
    self.logger.debug("zinger_removal: median_width: " + str(median_width))
    self.logger.info("zinger_removal [ok]")

    # Update returned values.
    if overwrite:
        self.data = data
    else:
        return data

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
    data=np.zeros_like(self.data)
    for channel in range(self.data.shape[0]):
        data[channel,:,:,:] = distribute_jobs(self.data[channel,:,:,:], _func, _args, _axis,
                           num_cores, chunk_size)

    # Update log.
    self.logger.debug("median_filter: size: " + str(size))
    self.logger.debug("median_filter: axis: " + str(axis))
    self.logger.info("median_filter [ok]")

    # Update returned values.
    if overwrite:
        self.data = data
    else:
        return data

# --------------------------------------------------------------------

def _diagnose(self):
    # Update log.
    self.logger.debug("diagnose: data: shape: " + str(self.data.shape))
    self.logger.debug("diagnose: data: dtype: " + str(self.data.dtype))
    self.logger.debug("diagnose: data: size: %.2fMB",
                      self.data.nbytes * 9.53674e-7)
    self.logger.debug(
        "diagnose: data: nans: " + str(np.sum(np.isnan(self.data))))
    self.logger.debug(
        "diagnose: data: -inf: " + str(np.sum(np.isneginf(self.data))))
    self.logger.debug(
        "diagnose: data: +inf: " + str(np.sum(np.isposinf(self.data))))
    self.logger.debug(
        "diagnose: data: positives: " + str(np.sum(self.data > 0)))
    self.logger.debug(
        "diagnose: data: negatives: " + str(np.sum(self.data < 0)))
    self.logger.debug("diagnose: data: mean: " + str(np.mean(self.data)))
    self.logger.debug("diagnose: data: min: " + str(np.min(self.data)))
    self.logger.debug("diagnose: data: max: " + str(np.max(self.data)))

    self.logger.debug("diagnose: theta: shape: " + str(self.theta.shape))
    self.logger.debug("diagnose: theta: dtype: " + str(self.theta.dtype))
    self.logger.debug("diagnose: theta: size: %.2fMB",
                      self.theta.nbytes * 9.53674e-7)
    self.logger.debug(
        "diagnose: theta: nans: " + str(np.sum(np.isnan(self.theta))))
    self.logger.debug(
        "diagnose: theta: -inf: " + str(np.sum(np.isneginf(self.theta))))
    self.logger.debug(
        "diagnose: theta: +inf: " + str(np.sum(np.isposinf(self.theta))))
    self.logger.debug(
        "diagnose: theta: positives: " + str(np.sum(self.theta > 0)))
    self.logger.debug(
        "diagnose: theta: negatives: " + str(np.sum(self.theta < 0)))
    self.logger.debug("diagnose: theta: mean: " + str(np.mean(self.theta)))
    self.logger.debug("diagnose: theta: min: " + str(np.min(self.theta)))
    self.logger.debug("diagnose: theta: max: " + str(np.max(self.theta)))

    self.logger.info("diagnose [ok]")


# --------------------------------------------------------------------


# Hook all these methods to TomoPy.
setattr(XFTomoDataset, 'apply_padding', _apply_padding)
setattr(XFTomoDataset, 'align_projections', _align_projections)
setattr(XFTomoDataset, 'median_filter', _median_filter)
setattr(XFTomoDataset, 'zinger_removal', _zinger_removal)
