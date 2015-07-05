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
Module for reconstruction software wrappers.
"""

from __future__ import absolute_import, division, print_function

import logging
import tomopy.util.mproc as mproc

import numpy as np

logger = logging.getLogger(__name__)


__author__ = "Daniel M. Pelt"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['astra']

default_options = {
    'astra': {
        'proj_type': 'line',
        'num_iter': 1,
    }
}

needed_options = {
    'astra': ['method']
}


def astra(*args):
    """
    Reconstruct object using the ASTRA toolbox

    Extra options
    ----------
    method : str
        ASTRA reconstruction method to use.
    num_iter : int, optional
        Number of algorithm iterations performed.
    proj_type : str, optional
        ASTRA projector type to use: 'line', 'linear', or 'strip' (CPU only).
    extra_options : dict, optional
        Extra options for the ASTRA config (i.e. those in cfg['option'])

    Warning
    -------
    When using CUDA, only 1 CPU core can be used.

    Example
    -------
    >>> import tomopy
    >>> obj = tomopy.shepp3d() # Generate an object.
    >>> ang = tomopy.angles(180) # Generate uniformly spaced tilt angles.
    >>> sim = tomopy.project(obj, ang) # Calculate projections.
    >>>
    >>> # Reconstruct object:
    >>> rec = tomopy.recon(sim, ang, algorithm=tomopy.astra,
    >>>       options={'method':'SART', 'num_iter':10*180, 'proj_type':'linear',
    >>>       'extra_options':{'MinConstraint':0}})
    >>>
    >>> # Show 64th slice of the reconstructed object.
    >>> import pylab
    >>> pylab.imshow(rec[64], cmap='gray')
    >>> pylab.show()
    """
    # Lazy import ASTRA
    import astra as astra_mod

    # Get shared arrays
    tomo = mproc.SHARED_TOMO
    recon = mproc.SHARED_ARRAY

    # Unpack arguments
    nang = args[0]
    nslices = args[1]
    ndet = args[2]
    centers = args[3]
    angles = args[4]
    num_gridx = args[5]['num_gridx']
    num_gridy = args[5]['num_gridy']
    opts = args[5]['options']
    istart = args[6]
    iend = args[7]

    # Check options
    for o in needed_options['astra']:
        if not o in opts:
            logger.error("Option %s needed for ASTRA reconstruction." % (o,))
            raise ValueError()
    for o in default_options['astra']:
        if not o in opts:
            opts[o] = default_options['astra'][o]

    # Create ASTRA geometries
    vol_geom = astra_mod.create_vol_geom((num_gridx,num_gridy))
    proj_geom = astra_mod.create_proj_geom('parallel',1.0,ndet,angles.astype(np.float64))

    # Create ASTRA data id
    sino = np.zeros((nang, ndet), dtype=np.float32)

    # Create ASTRA config
    cfg = astra_mod.astra_dict(opts['method'])

    if opts['proj_type']!='cuda':
        pi = astra_mod.create_projector(opts['proj_type'], proj_geom, vol_geom)
        sid = astra_mod.data2d.link('-sino', proj_geom, sino)
        cfg['ProjectorId'] = pi
        cfg['ProjectionDataId'] = sid
        use_cuda=False
    else:
        use_cuda=True

    if 'extra_options' in opts:
        cfg['option'] = opts['extra_options']


    # Perform reconstruction
    for i in xrange(istart, iend):
        sino[:] = tomo[:,i,:]

        # Fix center of rotation
        if use_cuda:
            proj_geom['option'] = {'ExtraDetectorOffset': (centers[i]-ndet/2.)*np.ones(nang)}
            sid = astra_mod.data2d.link('-sino', proj_geom, sino)
            cfg['ProjectionDataId'] = sid
            pi = astra_mod.create_projector(opts['proj_type'], proj_geom, vol_geom)
            cfg['ProjectorId'] = pi
        else:
            # Temporary workaround, will be fixed in later ASTRA version
            shft = int(np.round(ndet/2.-centers[i]))
            sino[:] = np.roll(sino,shft)
            l = shft
            r = sino.shape[1]+shft
            if l<0: l=0
            if r>sino.shape[1]: r=sino.shape[1]
            sino[:,0:l]=0
            sino[:,r:sino.shape[1]]=0

        vid = astra_mod.data2d.link('-vol', vol_geom, recon[i])
        cfg['ReconstructionDataId'] = vid
        alg_id = astra_mod.algorithm.create(cfg)
        astra_mod.algorithm.run(alg_id, opts['num_iter'])
        astra_mod.algorithm.delete(alg_id)
        astra_mod.data2d.delete(vid)
        if use_cuda:
            astra_mod.projector.delete(pi)
            astra_mod.data2d.delete(sid)

    # Clean up
    if not use_cuda:
        astra_mod.projector.delete(pi)
        astra_mod.data2d.delete(sid)

