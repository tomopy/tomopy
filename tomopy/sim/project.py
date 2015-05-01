#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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
Module for simulation of x-rays.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import shutil
import tomopy.extern as ext
from tomopy.util import *
import tomopy.misc.mproc as mp
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['angles',
           'project',
           'fan_to_para',
           'para_to_fan',
           'add_poisson',
           'add_focal_spot_blur']


def add_poisson(tomo):
    """
    Add Poisson noise.

    Warning
    -------
    Not implementd yet.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    Returns
    -------
    ndarray
        3D tomographic data after Poisson noise added.
    """
    logger.warning('Not implemented.')


def angles(nang, ang1=0., ang2=180.):
    """
    Return uniformly distributed projection angles in radian.

    Parameters
    ----------
    nang : int, optional
        Number of projections.

    ang1 : float, optional
        First projection angle in degrees.

    ang2 : float, optional
        Last projection angle in degrees.

    Returns
    -------
    array
        Projection angles
    """
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)


def project(obj, theta, center=None, ncore=None, nchunk=None):
    """
    Project x-rays through a given 3D object.

    Parameters
    ----------
    obj : ndarray
        Voxelized 3D object.
    theta : array
        Projection angles in radian.
    center: array, optional
        Location of rotation axis.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    obj = as_float32(obj)

    # Estimate data dimensions.
    ox, oy, oz = obj.shape
    dx = len(theta)
    dy = ox
    dz = np.ceil(np.sqrt(oy * oy + oz * oz)).astype('int')
    tomo = np.zeros((dx, dy, dz), dtype='float32')
    if center is None:
        center = np.ones(dy, dtype='float32') * dz / 2.
    elif np.array(center).size == 1:
        center = np.ones(dy, dtype='float32') * center

    theta = as_float32(theta)
    center = as_float32(center)

    mp.init_obj(obj)
    arr = mp.distribute_jobs(
        tomo,
        func=ext.c_project,
        args=(ox, oy, oz, theta, center, dx, dy, dz),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def fan_to_para(tomo, dist, geom):
    """
    Convert fan-beam data to parallel-beam data.

    Warning
    -------
    Not implemented yet.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    dist : float
        Distance from fan-beam vertex to rotation center.

    geom : str
        Fan beam geometry. 'arc' or 'line'.

    Returns
    -------
    ndarray
        Transformed 3D tomographic data.
    """
    logger.warning('Not implemented.')


def para_to_fan(tomo, dist, geom):
    """
    Convert parallel-beam data to fan-beam data.

    Warning
    -------
    Not implemented yet.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.

    dist : float
        Distance from fan-beam vertex to rotation center.

    geom : str
        Fan beam geometry. 'arc' or 'line'.

    Returns
    -------
    ndarray
        Transformed 3D tomographic data.
    """
    logger.warning('Not implemented.')


def add_focal_spot_blur(tomo, spotsize):
    """
    Add focal spot blur.

    Warning
    -------
    Not implemented yet.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    spotsize : float
        Focal spot size of circular x-ray source.
    """
    logger.warning('Not implemented.')


def _get_magnification(r1, r2):
    """
    Calculate magnification factor of the object.

    Parameters
    ----------
    r1 : float
        Source to object distance.

    r2 : float
        Object to detector distance.

    Returns
    -------
    float
        Magnification factor.
    """
    return (r1 + r2) / r1


def _get_otf(dx, dy, px, py, spotsize):
    """
    Calculate optical transfer function (OTF).

    Warning
    -------
    Not implemented yet.

    Parameters
    ----------
    dx, dy : int
        Number of detector pixels along x and y directions.
    px, py : float
        Pixel size in x and y directions.
    spotsize : float
        Focal spot size of circular x-ray source.

    Returns
    -------
    array
        2D OTF function.
    """
    logger.warning('Not implemented.')
