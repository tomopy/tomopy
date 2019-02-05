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
Module for hardware accelerated reconstruction algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import imp

logger = logging.getLogger(__name__)


__author__ = "Dake Feng"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['recon_accelerated']

known_implementations = {
    'tomoperi': 'tomopy_peri'
}


def recon_accelerated(
        tomo, theta, center=None, emission=True, algorithm=None, hardware=None,
        implementation=None, acc_option=None, init_recon=None, **kwargs):
    """
    Reconstruct object from projection data using hardware acceleration.
    A hardware acceleration implementation package is required. A free
    implementation can be downloaded from:
    https://github.com/PeriLLC/tomopy_peri_0.1.x

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
        One of the algorithms defined in non-accelerated recon function.

    hardware : str, optional
        One of the following supporting hardware platforms.

        'Xeon_Phi'
            Intel Xeon Phi hardware platform.
        'nVidia_GPU'
            nVidia GPU hardware platform.
    implementation : str, optional
        One of the following supporting packages,
        or a function providing accelerated recon.

        'tomo_peri'
            Tomopy_peri opensource packages,
            https://github.com/PeriLLC/tomopy_peri_0.1.x
    implementation : str, optional
        Options for hardware accelerated algorithms.
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
    >>> rec = tomopy.recon_accelerated(
    >>>           sim, ang, algorithm='ospml_hybrid',
    >>>           hardware='Xeon_Phi',
    >>>           implementation='tomoperi') # Reconstruct object.
    >>>
    >>> # Show 64th slice of the reconstructed object.
    >>> import pylab
    >>> pylab.imshow(rec[64], cmap='gray')
    >>> pylab.show()
   """

    if implementation is None:
        implementation = _search_implementation()
        logger.info('Implementation %s is chosen by default. ' %
                    implementation)
    else:

        if isinstance(implementation, str):
            # Check whether we have a known implementation
            if implementation not in known_implementations:
                raise ValueError(
                    'Keyword "implementation" must be one of %s, \
                    or a Python method.' %
                    (list(known_implementations.keys()),))

        elif not hasattr(implementation, '__call__'):
            raise ValueError(
                'Keyword "implementation" must be one of %s, \
                or a Python method.' %
                (list(known_implementations),))

    _impl_recon = _get_func(implementation)

    return _impl_recon(
        tomo, theta, center, emission, algorithm,
        hardware, acc_option, init_recon, **kwargs)


def _search_implementation():
    for key in known_implementations:
        try:
            imp.find_module(known_implementations[key])
            found = True
        except ImportError:
            found = False
        if found:
            return key

    raise ValueError('No known hardware accelerated reconstruction \
                     implementation found, try install one from %s!' %
                     (list(known_implementations.keys()),))


def _get_func(implementation):
    if implementation == 'tomoperi':
        try:
            import tomopy_peri.algorithm as alg
            func = alg.recon_accelerated
        except ImportError:
            raise ValueError(
                'Tomoperi hardware accelerated reconstruction \
                 implementation not found!')
    else:
        func = implementation
    return func
