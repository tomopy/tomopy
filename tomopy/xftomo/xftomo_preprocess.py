# -*- coding: utf-8 -*-

"""
This module containes a set of thin wrappers to hook the methods in
preprocess package to X-ray absorption tomography data object.
"""

import numpy as np
from images2gif import writeGif
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Import main TomoPy object.
from tomopy.xftomo.xftomo_dataset import XFTomoDataset

# Import available functons in the package.
from tomopy.algorithms.preprocess.align_projections import align_projections

# Import multiprocessing module.
from tomopy.tools.multiprocess_shared import distribute_jobs

# --------------------------------------------------------------------


def _align_projections(self, align_to_channel=None, method='scale_and_rotation_invariant_phase_correlation', output_gifs=False, output_dir=None, overwrite=True):

    if align_to_channel:
        data = self.data[align_to_channel,:,:,:]
    else:
        data = np.sum(self.data,axis=0)

    if output_gifs:
        data_copy=data

    data, translations = align_projections(data)

    if output_gifs:
        fig, axes = plt.subplots(ncols=2, nrows=1)
        ax1, ax2 = axes.ravel()
        images = []
        for i in range(data.shape[0]):
            plot1.imshow(data_copy[i,:,:])
            plot2.imshow(data[i,:,:])
            fig.canvas.draw()
            w,h = fig.canvas.get_width_height()
            buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.unit8)
            buf = np.roll(buff, 3, axis=2)
            images.append(buf)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        writeGif(os.path.join(output_dir, 'alignment.gif'))
        self.logger.debug('projection alignment gifs written: {:s}'.format(os.path.join(output_dir, 'alignment.gif')))

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
setattr(XTomoDataset, 'align_projections', _align_projections)
