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
Module for reconstruction software wrappers.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
from tomopy.util import mproc

import numpy as np
import copy
import threading

logger = logging.getLogger(__name__)


__author__ = "Daniel M. Pelt"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['astra', 'ufo_fbp', 'ufo_dfi', 'lprec']

default_options = {
    'astra': {
        'proj_type': 'linear',
        'num_iter': 1,
        'gpu_list': None,
    },
    'lprec': {
        'lpmethod': 'fbp',
        'interp_type': 'cubic',
        'filter_name': 'None',
        'num_iter': 1,
        'reg_par': 1,
        'gpu_list': [0],
    }
}

needed_options = {
    'astra': ['method']
}


def astra(tomo, center, recon, theta, **kwargs):
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
    # Lazy import ASTRA
    import astra as astra_mod

    # Unpack arguments
    nslices = tomo.shape[0]
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
    if proj_type == 'cuda' and opts['gpu_list'] is not None:
        import concurrent.futures as cf
        gpu_list = opts['gpu_list']
        ngpu = len(gpu_list)
        _, slcs = mproc.get_ncore_slices(nslices, ngpu)
        # execute recon on a thread per GPU
        with cf.ThreadPoolExecutor(ngpu) as e:
            for gpu, slc in zip(gpu_list, slcs):
                e.submit(astra_rec, tomo[slc], center[slc], recon[slc],
                            theta, vol_geom, niter, proj_type, gpu, opts)
    else:
        astra_rec(tomo, center, recon, theta, vol_geom, niter,
                           proj_type, None, opts)


def astra_rec(tomo, center, recon, theta, vol_geom, niter, proj_type, gpu_index, opts):
    # Lazy import ASTRA
    import astra as astra_mod
    nslices, nang, ndet = tomo.shape
    cfg = astra_mod.astra_dict(opts['method'])
    if 'extra_options' in opts:
        # NOTE: we are modifying 'extra_options' and so need to make a copy
        cfg['option'] = copy.deepcopy(opts['extra_options'])
    else:
        cfg['option'] = {}
    if gpu_index is not None:
        cfg['option']['GPUindex'] = gpu_index
    const_theta = np.ones(nang)
    proj_geom = astra_mod.create_proj_geom(
        'parallel', 1.0, ndet, theta.astype(np.float64))
    if hasattr(astra_mod, 'geom_postalignment'):
        proj_geom_orig = proj_geom
    for i in range(nslices):
        sino = tomo[i]
        if proj_type=='cuda':
            if hasattr(astra_mod, 'geom_postalignment'):
                proj_geom = astra_mod.geom_postalignment(proj_geom_orig, -(center[i] - ndet / 2.))
            else:
                proj_geom['option'] = {
                    'ExtraDetectorOffset':
                    (center[i] - ndet / 2.) * const_theta}
        else:
            shft = int(np.round(ndet / 2. - center[i]))
            if not shft == 0:
                sino = np.roll(tomo[i], shft)
                l = shft
                r = ndet + shft
                if l < 0:
                    l = 0
                if r > ndet:
                    r = ndet
                sino[:, :l] = 0
                sino[:,  r:] = 0
        pid = astra_mod.create_projector(proj_type, proj_geom, vol_geom)
        cfg['ProjectorId'] = pid
        sid = astra_mod.data2d.link('-sino', proj_geom, sino)
        cfg['ProjectionDataId'] = sid
        vid = astra_mod.data2d.link('-vol', vol_geom, recon[i])
        cfg['ReconstructionDataId'] = vid
        alg_id = astra_mod.algorithm.create(cfg)
        astra_mod.algorithm.run(alg_id, niter)
        astra_mod.algorithm.delete(alg_id)
        astra_mod.data2d.delete(vid)
        astra_mod.data2d.delete(sid)
        astra_mod.projector.delete(pid)


def _process_data(input_task, output_task, sinograms, slices):
    import ufo.numpy as unp
    num_sinograms, num_projections, width = sinograms.shape

    for i in range(num_sinograms):
        if i == 0:
            data = unp.empty_like(sinograms[i, :, :])
        else:
            data = input_task.get_input_buffer()

        # Set host array pointer and use that as first input
        data.set_host_array(
            sinograms[i, :, :].__array_interface__['data'][0], False)
        input_task.release_input_buffer(data)

        # Get last output and copy result back into NumPy buffer
        data = output_task.get_output_buffer()
        array = unp.asarray(data)
        frm = int(array.shape[0] / 2 - width / 2)
        to = int(array.shape[0] / 2 + width / 2)
        slices[i, :, :] = array[frm:to, frm:to]
        output_task.release_output_buffer(data)

    input_task.stop()


def ufo_fbp(tomo, center, recon, theta, **kwargs):
    """
    Reconstruct object using UFO's FBP pipeline
    """
    import gi
    gi.require_version('Ufo', '0.0')
    from gi.repository import Ufo

    width = tomo.shape[2]
    theta = theta[1] - theta[0]
    center = center[0]

    g = Ufo.TaskGraph()
    pm = Ufo.PluginManager()
    sched = Ufo.Scheduler()

    input_task = Ufo.InputTask()
    output_task = Ufo.OutputTask()
    fft = pm.get_task('fft')
    ifft = pm.get_task('ifft')
    fltr = pm.get_task('filter')
    backproject = pm.get_task('backproject')

    ifft.set_properties(crop_width=width)
    backproject.set_properties(
        axis_pos=center, angle_step=theta, angle_offset=np.pi)

    g.connect_nodes(input_task, fft)
    g.connect_nodes(fft, fltr)
    g.connect_nodes(fltr, ifft)
    g.connect_nodes(ifft, backproject)
    g.connect_nodes(backproject, output_task)

    args = (input_task, output_task, tomo, recon)
    thread = threading.Thread(target=_process_data, args=args)
    thread.start()
    sched.run(g)
    thread.join()

    logger.info("UFO+FBP run time: {}s".format(sched.props.time))


def ufo_dfi(tomo, center, recon, theta, **kwargs):
    """
    Reconstruct object using UFO's Direct Fourier pipeline
    """
    import gi
    gi.require_version('Ufo', '0.0')
    from gi.repository import Ufo

    theta = theta[1] - theta[0]
    center = center[0]

    g = Ufo.TaskGraph()
    pm = Ufo.PluginManager()
    sched = Ufo.Scheduler()

    input_task = Ufo.InputTask()
    output_task = Ufo.OutputTask()
    pad = pm.get_task('zeropad')
    fft = pm.get_task('fft')
    ifft = pm.get_task('ifft')
    dfi = pm.get_task('dfi-sinc')
    swap_forward = pm.get_task('swap-quadrants')
    swap_backward = pm.get_task('swap-quadrants')

    pad.set_properties(oversampling=1, center_of_rotation=center)
    fft.set_properties(dimensions=1, auto_zeropadding=False)
    ifft.set_properties(dimensions=2)
    dfi.set_properties(angle_step=theta)

    g.connect_nodes(input_task, pad)
    g.connect_nodes(pad, fft)
    g.connect_nodes(fft, dfi)
    g.connect_nodes(dfi, swap_forward)
    g.connect_nodes(swap_forward, ifft)
    g.connect_nodes(ifft, swap_backward)
    g.connect_nodes(swap_backward, output_task)

    args = (input_task, output_task, tomo, recon)
    thread = threading.Thread(target=_process_data, args=args)
    thread.start()
    sched.run(g)
    thread.join()

    logger.info("UFO+DFI run time: {}s".format(sched.props.time))


def lprec(tomo, center, recon, theta, **kwargs):
    """
    Reconstruct object using the Log-polar based method
    https://github.com/math-vrn/lprec

    Extra options
    ----------
    lpmethod : str
        LP reconsruction method to use
            - 'fbp'
            - 'grad'
            - 'cg'
            - 'tv'
            - 'em'
            - 'tve'
            - 'tvl1'
    filter_type:
        Filter for backprojection
            - 'ramp'
            - 'shepp-logan'
            - 'cosine'
            - 'cosine2'
            - 'hamming'
            - 'hann'
            - 'parzen'
    interp_type:
        Type of interpolation between Cartesian, polar and log-polar coordinates
            - 'linear'
            - 'cubic'
    Example
    -------
    >>> import tomopy
    >>> obj = tomopy.shepp3d() # Generate an object.
    >>> ang = tomopy.angles(180) # Generate uniformly spaced tilt angles.
    >>> sim = tomopy.project(obj, ang) # Calculate projections.
    >>>
    >>> # Reconstruct object:
    >>> rec = tomopy.recon(sim, ang, algorithm=tomopy.lprec,
    >>>       lpmethod='lpfbp', filter_name='parzen', interp_type='cubic', ncore=1)
    >>>
    >>> # Show 64th slice of the reconstructed object.
    >>> import pylab
    >>> pylab.imshow(rec[64], cmap='gray')
    >>> pylab.show()
    """
    from lprec import lpTransform
    from lprec import lpmethods
    import concurrent.futures as cf
    from functools import partial

    # set default options
    opts = kwargs
    for o in default_options['lprec']:
        if o not in kwargs:
            opts[o] = default_options['lprec'][o]

    filter_name = opts['filter_name']
    interp_type = opts['interp_type']
    lpmethod = opts['lpmethod']
    num_iter = opts['num_iter']
    reg_par = opts['reg_par']
    gpu_list = opts['gpu_list']

    # list of available methods for reconstruction
    lpmethods_list = {
        'fbp': lpmethods.fbp,
        'grad': lpmethods.grad,
        'cg': lpmethods.cg,
        'tv': lpmethods.tv,
        'em': lpmethods.em,
        'tve': lpmethods.tve,
        'tvl1': lpmethods.tvl1,
    }

    [Ns, Nproj, N] = tomo.shape
    ngpus = len(gpu_list)

    # number of slices for simultaneous processing by 1 gpu
    # (depends on gpu memory size, chosen for gpus with >= 4GB memory)
    Nssimgpu = min(int(pow(2, 24)/float(N*N)), int(np.ceil(Ns/float(ngpus))))

    # class lprec
    lp = lpTransform.lpTransform(
        N, Nproj, Nssimgpu, filter_name, int(center[0]), interp_type)
    # if not fbp, precompute for the forward transform
    lp.precompute(lpmethod != 'fbp')

    # list of slices sets for simultaneous processing b gpus
    ids_list = [None]*int(np.ceil(Ns/float(Nssimgpu)))
    for k in range(0, len(ids_list)):
        ids_list[k] = range(k*Nssimgpu, min(Ns, (k+1)*Nssimgpu))

    # init memory for each gpu
    for igpu in range(0, ngpus):
        gpu = gpu_list[igpu]
        # if not fbp, allocate memory for the forward transform arrays
        lp.initcmem(lpmethod != 'fbp', gpu)

    lock = threading.Lock()
    global BUSYGPUS
    BUSYGPUS = np.zeros(ngpus)
    # run reconstruciton on many gpus
    with cf.ThreadPoolExecutor(ngpus) as e:
        shift = 0
        for reconi in e.map(partial(lpmultigpu, lp, lpmethods_list[lpmethod], recon, tomo, num_iter, reg_par, gpu_list, lock), ids_list):
            recon[np.arange(0, reconi.shape[0])+shift] = reconi
            shift += reconi.shape[0]

    return recon


def lpmultigpu(lp, lpmethod, recon, tomo, num_iter, reg_par, gpu_list, lock, ids):
    """
    Reconstruction Nssimgpu slices simultaneously on 1 GPU
    """

    global BUSYGPUS
    lock.acquire()  # will block if lock is already held
    for k in range(len(gpu_list)):
        if BUSYGPUS[k] == 0:
            BUSYGPUS[k] = 1
            gpu_id = k
            break
    lock.release()
    gpu = gpu_list[gpu_id]

    # reconstruct
    recon[ids] = lpmethod(lp, recon[ids], tomo[ids], num_iter, reg_par, gpu)
    BUSYGPUS[gpu_id] = 0

    return recon[ids]
