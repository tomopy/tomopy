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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
        'proj_type': 'linear',
        'num_iter': 1,
        'gpu_list': None,
    }
}

needed_options = {
    'astra': ['method']
}


def astra(*args, **kwargs):
    """
    Reconstruct object using the ASTRA toolbox

    Extra options
    ----------
    method : str
        ASTRA reconstruction method to use.
    num_iter : int, optional
        Number of algorithm iterations performed.
    proj_type : str, optional
        ASTRA projector type to use (see ASTRA docs for more information):
            - 'cuda' (for GPU algorithms)
            - 'line', 'linear', or 'strip' (for CPU algorithms)
    gpu_list : list, optional
        List of GPU indices to use
    extra_options : dict, optional
        Extra options for the ASTRA config (i.e. those in cfg['option'])

    Example
    -------
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
    ret = [astra_run]
    ret.extend(args)
    ret.append(kwargs)
    return ret


def astra_run(tomo, center, recon, theta, kwargs):
    # Lazy import ASTRA
    import astra as astra_mod
    
    # Unpack arguments
    nang = tomo.shape[1]
    nslices = tomo.shape[0]
    ndet = tomo.shape[2]
    num_gridx = kwargs['num_gridx']
    num_gridy = kwargs['num_gridy']
    opts = kwargs['options']
    
    # Check options
    for o in needed_options['astra']:
        if o not in opts:
            logger.error("Option %s needed for ASTRA reconstruction." % (o,))
            raise ValueError()
    for o in default_options['astra']:
        if o not in opts:
            opts[o] = default_options['astra'][o]
    
    niter = opts['num_iter']
    proj_type = opts['proj_type']

    # Create ASTRA geometries
    vol_geom = astra_mod.create_vol_geom((num_gridx, num_gridy))
    
    # Number of GPUs to use
    if proj_type == 'cuda':
        if opts['gpu_list'] is not None:
            import concurrent.futures as cf
            gpu_list = opts['gpu_list']
            ngpu = len(gpu_list)
            executor = cf.ThreadPoolExecutor(max_workers=ngpu)
            chnks = np.round(np.linspace(0, nslices, ngpu+1)).astype(np.int)
            thrds = [executor.submit(astra_rec_cuda, tomo[chnks[i]:chnks[i+1]],
                                center[chnks[i]:chnks[i+1]],
                                recon[chnks[i]:chnks[i+1]], theta, vol_geom,
                                niter, proj_type, gpu_list[i], opts)
                                for i in range(ngpu)]
            for t in thrds:
                t.result()
        else:
            astra_rec_cuda(tomo, center, recon, theta, vol_geom, niter,
                           proj_type, None, opts)
    else:
        astra_rec_cpu(tomo, center, recon, theta, vol_geom, niter,
                      proj_type, opts)


def astra_rec_cuda(tomo, center, recon, theta, vol_geom, niter, proj_type, gpu_index, opts):
    # Lazy import ASTRA
    import astra as astra_mod
    nslices, nang, ndet = tomo.shape
    cfg = astra_mod.astra_dict(opts['method'])
    if 'extra_options' in opts:
        cfg['option'] = opts['extra_options']
    else:
        cfg['option'] = {}
    if gpu_index:
        cfg['option']['GPUindex'] = gpu_index
    oc = None
    const_theta = np.ones(nang)
    proj_geom = astra_mod.create_proj_geom('parallel', 1.0, ndet, theta.astype(np.float64))
    for i in range(nslices):
        if center[i] != oc:
            oc = center[i]
            proj_geom['option'] = {
                'ExtraDetectorOffset':
                (center[i] - ndet / 2.) * const_theta}
        pid = astra_mod.create_projector(proj_type, proj_geom, vol_geom)
        cfg['ProjectorId'] = pid
        sid = astra_mod.data2d.link('-sino', proj_geom, tomo[i])
        cfg['ProjectionDataId'] = sid
        vid = astra_mod.data2d.link('-vol', vol_geom, recon[i])
        cfg['ReconstructionDataId'] = vid
        alg_id = astra_mod.algorithm.create(cfg)
        astra_mod.algorithm.run(alg_id, niter)
        astra_mod.algorithm.delete(alg_id)
        astra_mod.data2d.delete(vid)
        astra_mod.data2d.delete(sid)
        astra_mod.projector.delete(pid)

def astra_rec_cpu(tomo, center, recon, theta, vol_geom, niter, proj_type, opts):
    # Lazy import ASTRA
    import astra as astra_mod
    nslices, nang, ndet = tomo.shape
    cfg = astra_mod.astra_dict(opts['method'])
    if 'extra_options' in opts:
        cfg['option'] = opts['extra_options']
    proj_geom = astra_mod.create_proj_geom('parallel', 1.0, ndet, theta.astype(np.float64))
    pid = astra_mod.create_projector(proj_type, proj_geom, vol_geom)
    sino = np.zeros((nang, ndet), dtype=np.float32)
    sid = astra_mod.data2d.link('-sino', proj_geom, sino)
    cfg['ProjectorId'] = pid
    cfg['ProjectionDataId'] = sid
    for i in range(nslices):
        shft = int(np.round(ndet / 2. - center[i]))
        if not shft == 0:
            sino[:] = np.roll(tomo[i], shft)
            l = shft
            r = ndet + shft
            if l < 0:
                l = 0
            if r > ndet:
                r = ndet
            sino[:, :l] = 0
            sino[:,  r:] = 0
        else:
            sino[:] = tomo[i]
        vid = astra_mod.data2d.link('-vol', vol_geom, recon[i])
        cfg['ReconstructionDataId'] = vid
        alg_id = astra_mod.algorithm.create(cfg)
        astra_mod.algorithm.run(alg_id, niter)
        astra_mod.algorithm.delete(alg_id)
        astra_mod.data2d.delete(vid)
    astra_mod.data2d.delete(sid)
    astra_mod.projector.delete(pid)
