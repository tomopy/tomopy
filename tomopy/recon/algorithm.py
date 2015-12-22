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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import tomopy.util.mproc as mproc
import tomopy.util.extern as extern
import tomopy.util.dtype as dtype
from tomopy.sim.project import angles, get_center
import multiprocessing
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['recon']


def recon(
        tomo, theta, center=None, emission=True, algorithm=None,
        init_recon=None, ncore=None, nchunk=None, **kwargs):
    """
    Reconstruct object from projection data.

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

        'art'
            Algebraic reconstruction technique :cite:`Kak:98`.
        'bart'
            Block algebraic reconstruction technique.
        'fbp'
            Filtered back-projection algorithm.
        'gridrec'
            Fourier grid reconstruction algorithm :cite:`Dowd:99`,
            :cite:`Rivers:06`.
        'mlem'
            Maximum-likelihood expectation maximization algorithm
            :cite:`Dempster:77`.
        'osem'
            Ordered-subset expectation maximization algorithm
            :cite:`Hudson:94`.
        'ospml_hybrid'
            Ordered-subset penalized maximum likelihood algorithm with
            weighted linear and quadratic penalties.
        'ospml_quad'
            Ordered-subset penalized maximum likelihood algorithm with
            quadratic penalties.
        'pml_hybrid'
            Penalized maximum likelihood algorithm with weighted linear
            and quadratic penalties :cite:`Chang:04`.
        'pml_quad'
            Penalized maximum likelihood algorithm with quadratic penalty.
        'sirt'
            Simultaneous algebraic reconstruction technique.

    num_gridx, num_gridy : int, optional
        Number of pixels along x- and y-axes in the reconstruction grid.
    filter_name : str, optional
        Name of the filter for analytic reconstruction.

        'none'
            No filter.
        'shepp'
            Shepp-Logan filter (default).
        'cosine'
            Cosine filter.
        'hann'
            Cosine filter.
        'hamming'
            Hamming filter.
        'ramlak'
            Ram-Lak filter.
        'parzen'
            Parzen filter.
        'butterworth'
            Butterworth filter.

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
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

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
    >>> rec = tomopy.recon(sim, ang, algorithm='art') # Reconstruct object.
    >>>
    >>> # Show 64th slice of the reconstructed object.
    >>> import pylab
    >>> pylab.imshow(rec[64], cmap='gray')
    >>> pylab.show()

    Example using the ASTRA toolbox for recontruction

    For more information, see http://sourceforge.net/p/astra-toolbox/wiki/Home/
    and https://github.com/astra-toolbox/astra-toolbox. To install the ASTRA
    toolbox with conda, use:

    conda install -c https://conda.binstar.org/astra-toolbox astra-toolbox

    >>> import tomopy
    >>> obj = tomopy.shepp3d() # Generate an object.
    >>> ang = tomopy.angles(180) # Generate uniformly spaced tilt angles.
    >>> sim = tomopy.project(obj, ang) # Calculate projections.
    >>>
    >>> # Reconstruct object:
    >>> rec = tomopy.recon(sim, ang, algorithm=tomopy.astra,
    >>>       options={'method':'SART', 'num_iter':10*180,
    >>>       'proj_type':'linear',
    >>>       'extra_options':{'MinConstraint':0}})
    >>>
    >>> # Show 64th slice of the reconstructed object.
    >>> import pylab
    >>> pylab.imshow(rec[64], cmap='gray')
    >>> pylab.show()
    """

    # Initialize tomography data.
    tomo = _init_tomo(tomo, emission)

    allowed_kwargs = {
        'art': ['num_gridx', 'num_gridy', 'num_iter'],
        'bart': ['num_gridx', 'num_gridy', 'num_iter',
                 'num_block', 'ind_block'],
        'fbp': ['num_gridx', 'num_gridy', 'filter_name'],
        'gridrec': ['num_gridx', 'num_gridy', 'filter_name'],
        'mlem': ['num_gridx', 'num_gridy', 'num_iter'],
        'osem': ['num_gridx', 'num_gridy', 'num_iter',
                 'num_block', 'ind_block'],
        'ospml_hybrid': ['num_gridx', 'num_gridy', 'num_iter',
                         'reg_par', 'num_block', 'ind_block'],
        'ospml_quad': ['num_gridx', 'num_gridy', 'num_iter',
                       'reg_par', 'num_block', 'ind_block'],
        'pml_hybrid': ['num_gridx', 'num_gridy', 'num_iter', 'reg_par'],
        'pml_quad': ['num_gridx', 'num_gridy', 'num_iter', 'reg_par'],
        'sirt': ['num_gridx', 'num_gridy', 'num_iter'],
    }

    generic_kwargs = ['num_gridx', 'num_gridy', 'options']

    # Generate kwargs for the algorithm.
    kwargs_defaults = _get_algorithm_kwargs(tomo.shape)

    if isinstance(algorithm, six.string_types):

        # Check whether we have an allowed method
        if algorithm not in allowed_kwargs:
            raise ValueError(
                'Keyword "algorithm" must be one of %s, or a Python method.' %
                (list(allowed_kwargs.keys()),))

        # Make sure have allowed kwargs appropriate for algorithm.
        for key, value in list(kwargs.items()):
            if key not in allowed_kwargs[algorithm]:
                raise ValueError(
                    '%s keyword not in allowed keywords %s' %
                    (key, allowed_kwargs[algorithm]))
            else:
                # Make sure they are numpy arrays.
                if not isinstance(kwargs, (np.ndarray, np.generic)):
                    kwargs[key] = np.array(value)

                # Make sure reg_par is float32.
                if key == 'reg_par':
                    if not isinstance(kwargs['reg_par'], np.float32):
                        kwargs['reg_par'] = np.array(value, dtype='float32')

        # Set kwarg defaults.
        for kw in allowed_kwargs[algorithm]:
            kwargs.setdefault(kw, kwargs_defaults[kw])

    elif hasattr(algorithm, '__call__'):
        # Set kwarg defaults.
        for kw in generic_kwargs:
            kwargs.setdefault(kw, kwargs_defaults[kw])
    else:
        raise ValueError(
            'Keyword "algorithm" must be one of %s, or a Python method.' %
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
        tomo[tomo <= 0.] = 1.
        tomo = -np.log(tomo)
    return tomo


def _init_recon(shape, init_recon, val=1e-6):
    if init_recon is None:
        recon = val * np.ones(shape, dtype='float32')
    else:
        recon = dtype.as_float32(init_recon)
    return recon


def _get_func(algorithm):
    if algorithm == 'art':
        func = extern.c_art
    elif algorithm == 'bart':
        func = extern.c_bart
    elif algorithm == 'fbp':
        func = extern.c_fbp
    elif algorithm == 'gridrec':
        func = extern.c_gridrec
    elif algorithm == 'mlem':
        func = extern.c_mlem
    elif algorithm == 'osem':
        func = extern.c_osem
    elif algorithm == 'ospml_hybrid':
        func = extern.c_ospml_hybrid
    elif algorithm == 'ospml_quad':
        func = extern.c_ospml_quad
    elif algorithm == 'pml_hybrid':
        func = extern.c_pml_hybrid
    elif algorithm == 'pml_quad':
        func = extern.c_pml_quad
    elif algorithm == 'sirt':
        func = extern.c_sirt
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
