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
Module for simulation of x-rays.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tomopy.util.extern as extern
import tomopy.util.dtype as dtype
import tomopy.util.mproc as mproc
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['angles',
           'project',
           'project2',
           'project3',
           'fan_to_para',
           'para_to_fan',
           'add_gaussian',
           'add_poisson',
           'add_salt_pepper',
           'add_focal_spot_blur']


def add_gaussian(tomo, mean=0, std=None):
    """
    Add Gaussian noise.

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
    """
    Add Poisson noise.

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
    """
    Add salt and pepper noise.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    prob : float, optional
        Independent probability that each element of a pixel might be
        corrupted by the salt and pepper type noise.
    val : float, optional
        Value to be assigned to the corrupted pixels.

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


def _round_to_even(num):
    return (np.ceil(num / 2.) * 2).astype('int')


def project(
        obj, theta, center=None, emission=True, pad=True,
        sinogram_order=False, ncore=None, nchunk=None):
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
    emission : bool, optional
        Determines whether output data is emission or transmission type.
    pad : bool, optional
        Determines if the projection image width will be padded or not. If True,
        then the diagonal length of the object cross-section will be used for the
        output size of the projection image width.
    sinogram_order: bool, optional
        Determines whether output data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    obj = dtype.as_float32(obj)
    theta = dtype.as_float32(theta)

    # Estimate data dimensions.
    oy, ox, oz = obj.shape
    dt = theta.size
    dy = oy
    if pad is True:
        dx = _round_to_even(np.sqrt(ox * ox + oz * oz) + 2)
    elif pad is False:
        dx = ox
    shape = dy, dt, dx
    tomo = dtype.empty_shared_array(shape)
    tomo[:] = 0.0
    center = get_center(shape, center)

    tomo = mproc.distribute_jobs(
        (obj, center, tomo),
        func=extern.c_project,
        args=(theta,),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    # NOTE: returns sinogram order with emmission=True
    if not emission:
        # convert data to be transmission type
        np.exp(-tomo, tomo)
    if not sinogram_order:
        # rotate to radiograph order
        tomo = np.swapaxes(tomo, 0, 1)  # doesn't copy data
        # copy data to sharedmem
        tomo = dtype.as_sharedmem(tomo, copy=True)

    return tomo


def project2(
        objx, objy, theta, center=None, emission=True, pad=True,
        sinogram_order=False, axis=0, ncore=None, nchunk=None):
    """
    Project x-rays through a given 3D object.

    Parameters
    ----------
    objx : ndarray
        (x, y) components of vector of a voxelized 3D object.
    theta : array
        Projection angles in radian.
    center: array, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether output data is emission or transmission type.
    pad : bool, optional
        Determines if the projection image width will be padded or not. If True,
        then the diagonal length of the object cross-section will be used for the
        output size of the projection image width.
    sinogram_order: bool, optional
        Determines whether output data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    objx = dtype.as_float32(objx)
    objy = dtype.as_float32(objy)
    theta = dtype.as_float32(theta)

    # Estimate data dimensions.
    oy, ox, oz = objx.shape
    dt = theta.size
    dy = oy
    if pad is True:
        dx = _round_to_even(np.sqrt(ox * ox + oz * oz) + 2)
    elif pad is False:
        dx = ox
    shape = dy, dt, dx
    tomo = dtype.empty_shared_array(shape)
    tomo[:] = 0.0
    center = get_center(shape, center)

    extern.c_project2(objx, objy, center, tomo, theta)
    
    # NOTE: returns sinogram order with emmission=True
    if not emission:
        # convert data to be transmission type
        np.exp(-tomo, tomo)
    if not sinogram_order:
        # rotate to radiograph order
        tomo = np.swapaxes(tomo, 0, 1)  # doesn't copy data
        # copy data to sharedmem
        tomo = dtype.as_sharedmem(tomo, copy=True)

    return tomo


def project3(
        objx, objy, objz, theta, center=None, 
        emission=True, pad=True,
        sinogram_order=False, axis=0, ncore=None, nchunk=None):
    """
    Project x-rays through a given 3D object.

    Parameters
    ----------
    objx : ndarray
        (x, y) components of vector of a voxelized 3D object.
    theta : array
        Projection angles in radian.
    center: array, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether output data is emission or transmission type.
    pad : bool, optional
        Determines if the projection image width will be padded or not. If True,
        then the diagonal length of the object cross-section will be used for the
        output size of the projection image width.
    sinogram_order: bool, optional
        Determines whether output data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        3D tomographic data.
    """
    objx = dtype.as_float32(objx)
    objy = dtype.as_float32(objy)
    objz = dtype.as_float32(objz)
    theta = dtype.as_float32(theta)

    # Estimate data dimensions.
    oy, ox, oz = objx.shape
    dt = theta.size
    dy = oy
    if pad is True:
        dx = _round_to_even(np.sqrt(ox * ox + oz * oz) + 2)
    elif pad is False:
        dx = ox
    shape = dy, dt, dx
    # print (shape)
    tomo = dtype.empty_shared_array(shape)
    tomo[:] = 0.0
    center = get_center(shape, center)

    extern.c_project3(objx, objy, objz, center, tomo, theta, axis)
    
    # NOTE: returns sinogram order with emmission=True
    if not emission:
        # convert data to be transmission type
        np.exp(-tomo, tomo)
    if not sinogram_order:
        # rotate to radiograph order
        tomo = np.swapaxes(tomo, 0, 1)  # doesn't copy data
        # copy data to sharedmem
        tomo = dtype.as_sharedmem(tomo, copy=True)
        
    return tomo


def get_center(shape, center):
    if center is None:
        center = np.ones(shape[0], dtype='float32') * (shape[2] / 2.)
    elif np.array(center).size == 1:
        center = np.ones(shape[0], dtype='float32') * center
    return dtype.as_float32(center)


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
