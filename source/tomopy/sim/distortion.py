#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2015-2019. UChicago Argonne, LLC. This software was produced  #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################
"""
Module for adding simulated distortions to data.

Contains functions for adding different types of noise and various artifacts
that real data sets contain. Examples include zingers or illumination drift.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np

import tomopy.util.dtype as dtype

logger = logging.getLogger(__name__)

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2019, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'add_gaussian',
    'add_poisson',
    'add_salt_pepper',
    'add_focal_spot_blur',
    'add_rings',
    'add_zingers',
]


def add_gaussian(tomo, mean=0, std=None):
    """Add Gaussian noise.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    mean : float, optional
        Mean of the Gaussian distribution.
    std : float, optional
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    ndarray
        3D tomographic data after Gaussian noise added.
    """
    tomo = dtype.as_ndarray(tomo)
    if std is None:
        std = tomo.max() * 0.05
    dx, dy, dz = tomo.shape
    tomo += std * np.random.randn(dx, dy, dz) + mean
    return tomo


def add_poisson(tomo):
    """Add Poisson noise.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    Returns
    -------
    ndarray
        3D tomographic data after Poisson noise added.
    """
    return np.random.poisson(tomo)


def add_salt_pepper(tomo, prob=0.01, val=None):
    """Add salt and pepper noise.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    prob : float, optional
        Independent probability that each element of a pixel might be
        corrupted by the salt and pepper type noise.
    val : float, optional
        Value to be assigned to the corrupted pixels. If None, val is the
        maximum value of tomo.

    Returns
    -------
    ndarray
        3D tomographic data after salt and pepper noise added.
    """
    tomo = dtype.as_ndarray(tomo)
    dx, dy, dz = tomo.shape
    ind = np.random.rand(dx, dy, dz) < prob
    if val is None:
        val = tomo.max()
    tomo[ind] = val
    return tomo


def add_rings(tomo, std=0.05):
    """Add rings.

    Rings are caused by inconsistent pixel sensitivity across the detector.

    The sensitivity of the pixels is modeled as normally distributed with an
    average sensitivity of 1 and a standard deviation given.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    std : float
        The standard deviation of the pixel sensitivity.

    Returns
    -------
    ndarray
        Tomographic data with zingers added.
    """
    sensitivity = np.random.normal(loc=1,
                                   scale=std,
                                   size=(1, new_tomo.shape[1],
                                         new_tomo.shape[2]))
    return np.copy(tomo) * sensitivity


def add_focal_spot_blur(tomo, spotsize):
    """Add focal spot blur.

    Raises
    ------
    Not implemented yet.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    spotsize : float
        Focal spot size of circular x-ray source.
    """
    raise NotImplementedError('Focal spot blur is not implemented.')


def add_zingers(tomo, f=0.01, sat=2**16):
    """Add zingers.

    Zingers are caused by stray X-rays hitting the detector and causing pixels
    to saturate.

    The zingers are uniformly distributed across the data set with the given
    frequency.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    f : float
        The fraction of measurements that are zingers.
    sat : float
        The pixel saturation value.

    Returns
    -------
    ndarray
        Tomographic data with zingers added.
    """
    zingers = np.random.uniform(0, 1, tomo.shape)
    zingers = zingers <= f  # five percent of measurements are zingers
    new_tomo = np.copy(tomo)
    new_tomo[zingers] = sat
    return new_tomo
