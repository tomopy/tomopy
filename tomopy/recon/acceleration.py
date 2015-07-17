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
Module for reconstruction algorithms.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tomopy.util.mproc as mproc
import tomopy.util.extern as extern
import tomopy.util.dtype as dtype
from tomopy.sim.project import angles, get_center
import multiprocessing
import logging

logger = logging.getLogger(__name__)


__author__ = "Dake Feng"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['recon']


def recon_accelerated(
        tomo, theta, center=None, emission=True, algorithm=None, hardware=None,
        init_recon=None, **kwargs):
    """
    Reconstruct object from projection data using hardware acceleration. 

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    theta : array
        Projection angles in radian.
    center: array, optional
        Location of rotation axis.
    emission : bool, optional
        Determines whether data is emission or transmission type.
    algorithm : {str, function}
        One of the following string values.

        'tomoperi.ospml_hybrid'
            Ordered-subset penalized maximum likelihood algorithm with
            weighted linear and quadratic penalties.
        'tomoperi.ospml_quad'
            Ordered-subset penalized maximum likelihood algorithm with
            quadratic penalties.
        'tomoperi.pml_hybrid'
            Penalized maximum likelihood algorithm with weighted linear
            and quadratic penalties :cite:`Chang:04`.
        'tomoperi.pml_quad'
            Penalized maximum likelihood algorithm with quadratic penalty.
    hardware : str, optional
        One of the following supporting platforms.

        'Xeon_Phi'
            Intel Xeon Phi acceleration hardware platform.
        'nVidia_CUDA'
            nVidia CUDA GPGPU hardware platform.
    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    filter_name : str, optional
        Name of the filter for analytic reconstruction.
    num_iter : int, optional
        Number of algorithm iterations performed.
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.
    num_iter : int, optional
        Number of algorithm iterations performed.
    reg_par : float, optional
        Regularization parameter for smoothing.
    init_recon : ndarray, optional
        Initial guess of the reconstruction.

    Returns
    -------
    ndarray
        Reconstructed 3D object.

    Warning
    -------
    Filtering is not implemented for fbp.

    Example
    -------
    >>> import tomopy
    >>> obj = tomopy.shepp3d() # Generate an object.
    >>> ang = tomopy.angles(180) # Generate uniformly spaced tilt angles.
    >>> sim = tomopy.project(obj, ang) # Calculate projections.
    >>> rec = tomopy.recon_accelerated(sim, ang, algorithm='tomoperi.ospml_hybrid', hardware='Xeon_Phi') # Reconstruct object.
    >>>
    >>> # Show 64th slice of the reconstructed object.
    >>> import pylab
    >>> pylab.imshow(rec[64], cmap='gray')
    >>> pylab.show()
   """

    # Initialize tomography data.
    tomo = _init_tomo(tomo, emission)

    allowed_kwargs = {
        'ospml_hybrid': ['num_gridx', 'num_gridy', 'num_iter',
                         'reg_par', 'num_block', 'ind_block'],
        'ospml_quad': ['num_gridx', 'num_gridy', 'num_iter',
                       'reg_par', 'num_block', 'ind_block'],
        'pml_hybrid': ['num_gridx', 'num_gridy', 'num_iter', 'reg_par'],
        'pml_quad': ['num_gridx', 'num_gridy', 'num_iter', 'reg_par'],
    }

    generic_kwargs = ['num_gridx', 'num_gridy', 'options']

    # Generate kwargs for the algorithm.
    kwargs_defaults = _get_algorithm_kwargs(tomo.shape)
    if isinstance(algorithm, str):
        # Check whether we have an allowed method
        if not algorithm in allowed_kwargs:
            raise ValueError('Keyword "algorithm" must be one of %s, or a Python method.' %
                             (list(allowed_kwargs.keys()),))
        # Make sure have allowed kwargs appropriate for algorithm.
        for key in kwargs:
            if key not in allowed_kwargs[algorithm]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowed_kwargs[algorithm]))
        # Set kwarg defaults.
        for kw in allowed_kwargs[algorithm]:
            kwargs.setdefault(kw, kwargs_defaults[kw])
    elif hasattr(algorithm, '__call__'):
        # Set kwarg defaults.
        for kw in generic_kwargs:
            kwargs.setdefault(kw, kwargs_defaults[kw])
    else:
        raise ValueError('Keyword "algorithm" must be one of %s, or a Python method.' %
                         (list(allowed_kwargs.keys()),))

    # Generate args for the algorithm.
    args = _get_algorithm_args(tomo.shape, theta, center)

    # Initialize reconstruction.
    recon = _init_recon(
        (tomo.shape[1], kwargs['num_gridx'], kwargs['num_gridy']),
        init_recon)
    return _dist_recon(
        tomo, recon, _get_func(algorithm), args, kwargs, ncore, nchunk)


def _init_tomo(tomo, emission):
    tomo = dtype.as_float32(tomo)
    if not emission:
        tomo = -np.log(tomo)
    return tomo


def _init_recon(shape, init_recon, val=1e-6):
    if init_recon is None:
        recon = val * np.ones(shape, dtype='float32')
    else:
        recon = dtype.as_float32(recon)
    return recon


def _get_func(algorithm):
    if algorithm == 'ospml_hybrid':
        func = extern.c_ospml_hybrid
    elif algorithm == 'ospml_quad':
        func = extern.c_ospml_quad
    elif algorithm == 'pml_hybrid':
        func = extern.c_pml_hybrid
    elif algorithm == 'pml_quad':
        func = extern.c_pml_quad
    else:
        func = algorithm
    return func


def _dist_recon(tomo, recon, algorithm, args, kwargs, ncore, nchunk):
    mproc.init_tomo(tomo)
    return mproc.distribute_jobs(
        recon,
        func=algorithm,
        args=args,
        kwargs=kwargs,
        axis=0,
        ncore=ncore,
        nchunk=nchunk)


def _get_algorithm_args(shape, theta, center):
    dx, dy, dz = shape
    theta = dtype.as_float32(theta)
    center = get_center(shape, center)
    return (dx, dy, dz, center, theta)


def _get_algorithm_kwargs(shape):
    dx, dy, dz = shape
    return {
        'num_gridx': dz,
        'num_gridy': dz,
        'filter_name': np.array('shepp', dtype=(str, 16)),
        'num_iter': dtype.as_int32(1),
        'reg_par': np.ones(10, dtype='float32'),
        'num_block': dtype.as_int32(1),
        'ind_block': np.arange(0, dx, dtype='float32'),
        'options': {},
    }
