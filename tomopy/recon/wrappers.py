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
        'lpmethod': 'lpfbp',
        'interp_type': 'cubic',
        'filter_name': 'None',
        'num_iter': '1',
        'reg_par': '1',
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
    if proj_type == 'cuda':
        if opts['gpu_list'] is not None:
            import concurrent.futures as cf
            gpu_list = opts['gpu_list']
            ngpu = len(gpu_list)
            _, slcs = mproc.get_ncore_slices(nslices, ngpu)
            # execute recon on a thread per GPU
            with cf.ThreadPoolExecutor(ngpu) as e:
                for gpu, slc in zip(gpu_list, slcs):
                    e.submit(astra_rec_cuda, tomo[slc], center[slc], recon[slc],
                             theta, vol_geom, niter, proj_type, gpu, opts)
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
        # NOTE: we are modifying 'extra_options' and so need to make a copy
        cfg['option'] = copy.deepcopy(opts['extra_options'])
    else:
        cfg['option'] = {}
    if gpu_index is not None:
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


def _process_data(input_task, output_task, sinograms, slices):
    import ufo.numpy as unp
    num_sinograms, num_projections, width = sinograms.shape

    for i in range(num_sinograms):
        if i == 0:
            data = unp.empty_like(sinograms[i,:,:])
        else:
            data = input_task.get_input_buffer()

        # Set host array pointer and use that as first input
        data.set_host_array(sinograms[i,:,:].__array_interface__['data'][0], False)
        input_task.release_input_buffer(data)

        # Get last output and copy result back into NumPy buffer
        data = output_task.get_output_buffer()
        array = unp.asarray(data)
        frm = int(array.shape[0] / 2 - width / 2)
        to = int(array.shape[0] / 2 + width / 2)
        slices[i,:,:] = array[frm:to, frm:to]
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
    backproject.set_properties(axis_pos=center, angle_step=theta, angle_offset=np.pi)

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
            - 'lpfbp'
            - 'lpgrad'
            - 'lptv'
            - 'lpem'
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
    lpmethods = {'lpfbp' : lpfbp,
                 'lpgrad' : lpgrad,
                 'lptv' : lptv,
                 'lpem' : lpem}

    from lprec import lpTransform 

    #set default options
    opts = kwargs
    for o in default_options['lprec']:
        if o not in kwargs:
            opts[o] = default_options['lprec'][o]

    filter_name = opts['filter_name']
    interp_type = opts['interp_type']
    lpmethod = opts['lpmethod']
    num_iter = opts['num_iter']
    reg_par = opts['reg_par']

    #Init lp method
    #number of slices for simultanious processing by 1 gpu, chosen for 4GB gpus
    Nslices, Nproj, N = tomo.shape
    Nslices0 = min(int(pow(2,25)/float(N*N)),Nslices)
    lphandle=lpTransform.lpTransform(N,Nproj,Nslices0,filter_name,int(center[0]+0.5),interp_type)

    if(lpmethod=='lpfbp'):
        #precompute only for the adj transform
        lphandle.precompute(0)
        lphandle.initcmem(0)   
        for k in range(0,int(np.ceil(Nslices/float(Nslices0)))):
            ids = range(k*Nslices0,min(Nslices,(k+1)*Nslices0))
            recon[ids] = lpmethods[lpmethod](lphandle,tomo[ids])
    else:
        #precompute for both fwd and adj transforms
        lphandle.precompute(1)
        lphandle.initcmem(1)   
        #run
        for k in range(0,int(np.ceil(Nslices/float(Nslices0)))):
            ids = range(k*Nslices0,min(Nslices,(k+1)*Nslices0))
            recon[ids] = lpmethods[lpmethod](lphandle,recon[ids],tomo[ids],num_iter,reg_par)

def lpfbp(lp,tomo):
    return lp.adj(tomo)

def lpgrad(lp,recon,tomo,num_iter,reg_par):

    recon0 = recon
    grad = recon*0
    grad0 = recon*0

    for i in range(0,num_iter):
        grad = 2*lp.adj(lp.fwd(recon)-tomo)
        if(reg_par<0):
            if(i==0):
                lam = np.float32(1e-3*np.ones(tomo.shape[0]))
            else:
                lam = np.sum(np.sum((recon-recon0)*(grad-grad0),1),1)/np.sum(np.sum((grad-grad0)*(grad-grad0),1),1)
        else:
            lam = np.float32(reg_par*np.ones(tomo.shape[0]))
        recon0 = recon
        grad0 = grad
        recon = recon - np.reshape(lam,[tomo.shape[0],1,1])*grad

    return recon

def lptv(lp,recon,tomo,num_iter,reg_par):

    lam = reg_par
    c = 0.35 #1/power_method(lp,tomo,num_iter)

    recon0 = recon
    prox0x = recon*0
    prox0y = recon*0
    div0 = recon*0
    prox1 = tomo*0


    for i in range(0,num_iter):
        #forward step
        #compute proximal prox0
        prox0x[:,:,:-1] +=  c*(recon[:,:,1:]-recon[:,:,:-1])
        prox0y[:,:-1,:] +=  c*(recon[:,1:,:]-recon[:,:-1,:])
        nprox = np.maximum(1,np.sqrt(prox0x*prox0x+prox0y*prox0y)/lam)
        prox0x = prox0x/nprox
        prox0y = prox0y/nprox
        #compute proximal prox1        
        prox1 = (prox1+c*lp.fwd(recon)-c*tomo)/(1+c)

        #backward step
        recon = recon0
        div0[:,:,1:] = (prox0x[:,:,1:]-prox0x[:,:,:-1])
        div0[:,:,0] = prox0x[:,:,0]
        div0[:,1:,:] += (prox0y[:,1:,:]-prox0y[:,:-1,:])
        div0[:,0,:] += prox0y[:,0,:]
        recon0 = recon0-c*lp.adj(prox1)+c*div0

        #update recon
        recon = 2*recon0 - recon

    return recon

def lpem(lp,recon,tomo,num_iter,reg_par):
    xi = lp.adj(tomo*0+np.float32(1))
    eps = reg_par
    xi = xi+np.float32(eps*np.max(xi))
    e = np.max(tomo)*eps
    for i in range(0,num_iter):
        g = lp.fwd(recon)
        upd = lp.adj(tomo/(g+e))
        recon = recon*(upd/xi)
    return recon


## power_method for estimate constant c in the lptv method
# def power_method(lp,tomo,num_iter):
#     x = lp.adj(tomo)
#     for k in range(0,num_iter):
#         prox0x = x*0
#         prox0y = x*0
#         div0 = x*0
#         prox0x[:,:,:-1] =  x[:,:,1:]-x[:,:,:-1]
#         prox0y[:,:-1,:] =  x[:,1:,:]-x[:,:-1,:]
#         div0[:,:,1:] = (prox0x[:,:,1:]-prox0x[:,:,:-1])
#         div0[:,:,0] = prox0x[:,:,0]
#         div0[:,1:,:] += (prox0y[:,1:,:]-prox0y[:,:-1,:])
#         div0[:,0,:] += prox0y[:,0,:]
#         x = lp.adj(lp.fwd(x)) - div0
#         s = np.linalg.norm(x)
#         x = x/s
#         return sqrt(s)
