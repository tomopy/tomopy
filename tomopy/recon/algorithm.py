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
from tomopy.sim.project import get_center
import math
import logging
import concurrent.futures as cf

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['recon', 'init_tomo']


def recon(
        tomo, theta, center=None, sinogram_order=False, algorithm=None,
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
    sinogram_order: bool, optional
        Determins whether data is a stack of sinograms (True, y-axis first axis)
        or a stack of radiographs (False, theta first axis).
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
        'custom'
            A numpy array of size `next_power_of_2(num_detector_columns)/2`
            specifying a custom filter in Fourier domain. The first element
            of the filter should be the zero-frequency component.
        'custom2d'
            A numpy array of size `num_projections*next_power_of_2(num_detector_columns)/2`
            specifying a custom angle-dependent filter in Fourier domain. The first element
            of each filter should be the zero-frequency component.

    filter_par: list, optional
        Filter parameters as a list.
    num_iter : int, optional
        Number of algorithm iterations performed.
    num_block : int, optional
        Number of data blocks for intermediate updating the object.
    ind_block : array of int, optional
        Order of projections to be used for updating.
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
    tomo = init_tomo(tomo, sinogram_order, sharedmem=False)

    allowed_kwargs = {
        'art': ['num_gridx', 'num_gridy', 'num_iter'],
        'bart': ['num_gridx', 'num_gridy', 'num_iter',
                 'num_block', 'ind_block'],
        'fbp': ['num_gridx', 'num_gridy', 'filter_name', 'filter_par'],
        'gridrec': ['num_gridx', 'num_gridy', 'filter_name', 'filter_par'],
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
                if not isinstance(kwargs[key], (np.ndarray, np.generic)) and not isinstance(kwargs[key], six.string_types):
                    kwargs[key] = np.array(value)

                # Make sure reg_par and filter_par is float32.
                if key == 'reg_par' or key == 'filter_par':
                    if not isinstance(kwargs[key], np.float32):
                        kwargs[key] = np.array(value, dtype='float32')

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
    center_arr = get_center(tomo.shape, center)
    args = _get_algorithm_args(theta)

    # Initialize reconstruction.
    recon_shape = (tomo.shape[0], kwargs['num_gridx'], kwargs['num_gridy'])
    recon = _init_recon(recon_shape, init_recon, sharedmem=False)
    return _dist_recon(
        tomo, center_arr, recon, _get_func(algorithm), args, kwargs, ncore, nchunk)


# Convert data to sinogram order
# Also ensure contiguous data and set to sharedmem if parameter set to True
def init_tomo(tomo, sinogram_order, sharedmem=True):
    tomo = dtype.as_float32(tomo)
    if not sinogram_order:
        tomo = np.swapaxes(tomo, 0, 1)  # doesn't copy data
    if sharedmem:
        # copy data to sharedmem (if not already or not contiguous)
        tomo = dtype.as_sharedmem(tomo, copy=not dtype.is_contiguous(tomo))
    else:
        # ensure contiguous
        tomo = np.require(tomo, requirements="AC")
    return tomo


def _init_recon(shape, init_recon, val=1e-6, sharedmem=True):
    if init_recon is None:
        if sharedmem:
            recon = dtype.empty_shared_array(shape)
            recon[:] = val
        else:
            recon = np.full(shape, val, dtype=np.float32)
    else:
        recon = np.require(init_recon, dtype=np.float32, requirements="AC")
        if sharedmem:
            recon = dtype.as_sharedmem(recon)
    return recon


# TODO: replace with dict, then users could easily add their own functions
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


def _dist_recon(tomo, center, recon, algorithm, args, kwargs, ncore, nchunk):
    axis_size = recon.shape[0]
    ncore, slcs = mproc.get_ncore_slices(axis_size, ncore, nchunk)
    if ncore == 1:
        for slc in slcs:
            # run in this thread (useful for debugging)
            algorithm(tomo[slc], center[slc], recon[slc], *args, **kwargs)
    else:
        # execute recon on ncore threads
        with cf.ThreadPoolExecutor(ncore) as e:
            for slc in slcs:
                e.submit(algorithm, tomo[slc], center[slc], recon[slc], *args, **kwargs)
    return recon


def _get_algorithm_args(theta):
    theta = dtype.as_float32(theta)
    return (theta, )


def _get_algorithm_kwargs(shape):
    dy, dt, dx = shape
    return {
        'num_gridx': dx,
        'num_gridy': dx,
        'filter_name': 'shepp',
        'filter_par': np.array([0.5, 8], dtype='float32'),
        'num_iter': dtype.as_int32(1),
        'reg_par': np.ones(10, dtype='float32'),
        'num_block': dtype.as_int32(1),
        'ind_block': np.arange(0, dt, dtype=np.float32),  # TODO: I think this should be int
        'options': {},
    }
