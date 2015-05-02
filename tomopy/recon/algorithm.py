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
import tomopy.misc.mproc as mp
import tomopy.extern as ext
from tomopy.sim.project import angles, get_center
from tomopy.util import *
import multiprocessing
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['recon']


def recon(
        tomo, theta, center=None, emission=False,
        num_gridx=None, num_gridy=None, algorithm=None, **kwargs):

    allowed_kwargs = {
        'art': ['num_iter'],
        'bart': ['num_iter', 'num_block', 'ind_block'],
        'fbp': ['filter_name'],
        'gridrec': ['filter_name'],
        'mlem': ['num_iter'],
        'osem': ['num_iter', 'num_block', 'ind_block'],
        'ospml_hybrid': ['num_iter', 'reg_par', 'num_block', 'ind_block'],
        'ospml_quad': ['num_iter', 'reg_par', 'num_block', 'ind_block'],
        'pml_hybrid': ['num_iter', 'reg_par'],
        'pml_quad': ['num_iter', 'reg_par'],
        'sirt': ['num_iter'],
    }

    args = _get_algorithm_args(
        tomo.shape, theta, center, emission, num_gridx, num_gridy)
    kwargs_defaults = _get_algorithm_kwargs(tomo.shape)

    if isinstance(algorithm, str):
        # Make sure have allowed kwargs appropriate for algorithm
        for key in kwargs:
            if key not in allowed_kwargs[algorithm]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowed_kwargs[algorithm]))
        # Set kwarg defaults
        for kw in allowed_kwargs[algorithm]:
            kwargs.setdefault(kw, kwargs_defaults[kw])
    elif algorithm is None:
        raise ValueError('Keyword "algorithm" must be one of %s.' %
                         (list(allowed_kwargs.keys()),))

    recon = 1e-6 * np.ones((tomo.shape[1], args[5], args[6]), dtype='float32')

    if algorithm == 'art':
        arr = _dist_recon(tomo, recon, ext.c_art, args, kwargs)

    elif algorithm == 'bart':
        arr = _dist_recon(tomo, recon, ext.c_bart, args, kwargs)

    elif algorithm == 'fbp':
        arr = _dist_recon(tomo, recon, ext.c_fbp, args, kwargs)

    elif algorithm == 'gridrec':
        arr = _dist_recon(tomo, recon, ext.c_gridrec, args, kwargs)

    elif algorithm == 'mlem':
        arr = _dist_recon(tomo, recon, ext.c_mlem, args, kwargs)

    elif algorithm == 'osem':
        arr = _dist_recon(tomo, recon, ext.c_osem, args, kwargs)

    elif algorithm == 'ospml_hybrid':
        arr = _dist_recon(tomo, recon, ext.c_ospml_hybrid, args, kwargs)

    elif algorithm == 'ospml_quad':
        arr = _dist_recon(tomo, recon, ext.c_ospml_quad, args, kwargs)

    elif algorithm == 'pml_hybrid':
        arr = _dist_recon(tomo, recon, ext.c_pml_hybrid, args, kwargs)

    elif algorithm == 'pml_quad':
        arr = _dist_recon(tomo, recon, ext.c_pml_quad, args, kwargs)

    elif algorithm == 'sirt':
        arr = _dist_recon(tomo, recon, ext.c_sirt, args, kwargs)

    return arr


def _dist_recon(tomo, recon, algorithm, args, kwargs):
    mp.init_tomo(tomo)
    return mp.distribute_jobs(
        recon,
        func=algorithm,
        args=args,
        kwargs=kwargs,
        axis=0,
        ncore=None,
        nchunk=None)


def _get_algorithm_args(shape, theta, center, emission, num_gridx, num_gridy):
    dx, dy, dz = shape
    theta = as_float32(theta)
    center = get_center(shape, center)
    if num_gridx is None:
        num_gridx = shape[2]
    if num_gridy is None:
        num_gridy = shape[2]
    print(center)
    return (dx, dy, dz, center, theta, num_gridx, num_gridy)


def _get_algorithm_kwargs(shape):
    return {
        'filter_name': np.array('shepp', dtype=(str, 16)),
        'num_iter': as_int32(1),
        'reg_par': np.ones(10, dtype='float32'),
        'num_block': as_int32(1),
        'ind_block': np.arange(0, shape[0], dtype='float32'),
    }
