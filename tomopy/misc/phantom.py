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
Module for generating synthetic phantoms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import skimage
import skimage.transform
import tifffile
import os.path
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['baboon',
           'barbara',
           'cameraman',
           'checkerboard',
           'lena',
           'peppers',
           'shepp2d',
           'shepp3d',
           'phantom']


DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))

try:
    resize_kwargs = {'anti_aliasing': False}
    ignore = skimage.transform.resize(np.zeros(5), [2], mode='constant',
                                      **resize_kwargs)
except TypeError:
    logger.debug("Determined that the anti_aliasing keyword is not needed.")
    resize_kwargs = dict()


def baboon(size=512, dtype='float32'):
    """
    Load test baboon image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'baboon.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    im = im.astype(dtype)
    return im


def barbara(size=512, dtype='float32'):
    """
    Load test Barbara image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'barbara.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    return im.astype(dtype)


def cameraman(size=512, dtype='float32'):
    """
    Load test cameraman image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'cameraman.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    return im.astype(dtype)


def checkerboard(size=512, dtype='float32'):
    """
    Load test checkerboard image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'checkerboard.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    return im.astype(dtype)


def lena(size=512, dtype='float32'):
    """
    Load test Lena image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'lena.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    return im.astype(dtype)


def peppers(size=512, dtype='float32'):
    """
    Load test peppers image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'peppers.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    return im.astype(dtype)


def shepp2d(size=512, dtype='float32'):
    """
    Load test Shepp-Logan image array.

    Parameters
    ----------
    size : int or tuple of int, optional
        Size of the output image.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 2)
    fname = os.path.join(DATA_PATH, 'shepp2d.tif')
    im = tifffile.imread(fname)
    im = skimage.transform.resize(im, size, order=3,
                                  preserve_range=True, mode='constant',
                                  **resize_kwargs)
    im = np.expand_dims(im, 0)
    return im.astype(dtype)


def _totuple(size, dim):
    """
    Converts size to tuple.
    """
    if not isinstance(size, tuple):
        if dim == 2:
            size = (size, size)
        elif dim == 3:
            size = (size, size, size)
    return size


def shepp3d(size=128, dtype='float32'):
    """
    Load 3D Shepp-Logan image array.

    Parameters
    ----------
    size : int or tuple, optional
        Size of the 3D data.
    dtype : str, optional
        The desired data-type for the array.

    Returns
    -------
    ndarray
        Output 3D test image.
    """
    size = _totuple(size, 3)
    shepp_params = _array_to_params(_get_shepp_array())
    return phantom(size, shepp_params, dtype).clip(0, np.inf)


def phantom(size, params, dtype='float32'):
    """
    Generate a cube of given size using a list of ellipsoid parameters.

    Parameters
    ----------
    size: tuple of int
        Size of the output cube.
    params: list of dict
        List of dictionaries with the parameters defining the ellipsoids
        to include in the cube.
    dtype: str, optional
        Data type of the output ndarray.

    Returns
    -------
    ndarray
        3D object filled with the specified ellipsoids.
    """
    # instantiate ndarray cube
    obj = np.zeros(size, dtype=dtype)

    # define coords
    coords = _define_coords(size)

    # recursively add ellipsoids to cube
    for param in params:
        _ellipsoid(param, out=obj, coords=coords)
    return obj


def _ellipsoid(params, shape=None, out=None, coords=None):
    """
    Generate a cube containing an ellipsoid defined by its parameters.
    If out is given, fills the given cube instead of creating a new one.
    """
    # handle inputs
    if shape is None and out is None:
        raise ValueError("You need to set shape or out")
    if out is None:
        out = np.zeros(shape)
    if shape is None:
        shape = out.shape
    if len(shape) == 1:
        shape = shape, shape, shape
    elif len(shape) == 2:
        shape = shape[0], shape[1], 1
    elif len(shape) > 3:
        raise ValueError("input shape must be lower or equal to 3")
    if coords is None:
        coords = _define_coords(shape)

    # rotate coords
    coords = _transform(coords, params)

    # recast as ndarray
    coords = np.asarray(coords)
    np.square(coords, out=coords)
    ellip_mask = coords.sum(axis=0) <= 1.
    ellip_mask.resize(shape)

    # fill ellipsoid with value
    out[ ellip_mask ] += params['A']
    return out


def _rotation_matrix(p):
    """
    Defines an Euler rotation matrix from angles phi, theta and psi.
    """
    cphi = np.cos(np.radians(p['phi']))
    sphi = np.sin(np.radians(p['phi']))
    ctheta = np.cos(np.radians(p['theta']))
    stheta = np.sin(np.radians(p['theta']))
    cpsi = np.cos(np.radians(p['psi']))
    spsi = np.sin(np.radians(p['psi']))
    alpha = [[cpsi * cphi - ctheta * sphi * spsi,
              cpsi * sphi + ctheta * cphi * spsi,
              spsi * stheta],
             [-spsi * cphi - ctheta * sphi * cpsi,
              -spsi * sphi + ctheta * cphi * cpsi,
              cpsi * stheta],
             [stheta * sphi,
              -stheta * cphi,
              ctheta]]
    return np.asarray(alpha)


def _define_coords(shape):
    """
    Generate a tuple of coords in 3D with a given shape.
    """
    mgrid = np.lib.index_tricks.nd_grid()
    cshape = np.asarray(1j) * shape
    x, y, z = mgrid[-1:1:cshape[0], -1:1:cshape[1], -1:1:cshape[2]]
    return x, y, z


def _transform(coords, p):
    """
    Apply rotation, translation and rescaling to a 3-tuple of coords.
    """
    alpha = _rotation_matrix(p)
    out_coords = np.tensordot(alpha, coords, axes=1)
    _shape = (3,) + (1,) * ( out_coords.ndim - 1 )
    _dt = out_coords.dtype
    M0 = np.array([p['x0'], p['y0'], p['z0']], dtype=_dt).reshape(_shape)
    sc = np.array([p['a'], p['b'], p['c']], dtype=_dt).reshape(_shape)
    out_coords -= M0
    out_coords /= sc
    return out_coords


def _get_shepp_array():
    """
    Returns the parameters for generating modified Shepp-Logan phantom.
    """
    shepp_array = [
        [1.,  .6900, .920, .810,   0.,     0.,   0.,   90.,   90.,   90.],
        [-.8, .6624, .874, .780,   0., -.0184,   0.,   90.,   90.,   90.],
        [-.2, .1100, .310, .220,  .22,     0.,   0., -108.,   90.,  100.],
        [-.2, .1600, .410, .280, -.22,     0.,   0.,  108.,   90.,  100.],
        [.1,  .2100, .250, .410,   0.,    .35, -.15,   90.,   90.,   90.],
        [.1,  .0460, .046, .050,   0.,     .1,  .25,   90.,   90.,   90.],
        [.1,  .0460, .046, .050,   0.,    -.1,  .25,   90.,   90.,   90.],
        [.1,  .0460, .023, .050, -.08,  -.605,   0.,   90.,   90.,   90.],
        [.1,  .0230, .023, .020,   0.,  -.606,   0.,   90.,   90.,   90.],
        [.1,  .0230, .046, .020,  .06,  -.605,   0.,   90.,   90.,   90.]]
    return shepp_array


def _array_to_params(array):
    """
    Converts list to a dictionary.
    """
    # mandatory parameters to define an ellipsoid
    params_tuple = [
        'A',
        'a', 'b', 'c',
        'x0', 'y0', 'z0',
        'phi', 'theta', 'psi']

    array = np.asarray(array)
    out = []
    for i in range(array.shape[0]):
        tmp = dict()
        for k, j in zip(params_tuple, list(range(array.shape[1]))):
            tmp[k] = array[i, j]
        out.append(tmp)
    return out
