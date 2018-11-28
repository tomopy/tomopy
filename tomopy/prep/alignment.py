#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016-17, UChicago Argonne, LLC. All rights reserved.      #
#                                                                         #
# Copyright 2017-17. UChicago Argonne, LLC. This software was produced    #
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

import numpy as np
import logging
from skimage import transform as tf
from skimage.feature import register_translation
from tomopy.recon.algorithm import recon
from tomopy.sim.project import project
import dxchange
import numpy as np

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016-17, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['align_seq',
           'align_joint',
           'scale',
           'tilt',
           'add_jitter',
           'add_noise',
           'blur_edges',
           'shift_images']


def align_seq(
        prj, ang, fdir='.', iters=10, pad=(0, 0),
        blur=True, center=None, algorithm='sirt',
        upsample_factor=10, rin=0.5, rout=0.8,
        save=False, debug=True):
    """
    Aligns the projection image stack using the sequential
    re-projection algorithm :cite:`Gursoy:17`.

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.
    ang : ndarray
        Projection angles in radians as an array.
    iters : scalar, optional
        Number of iterations of the algorithm.
    pad : list-like, optional
        Padding for projection images in x and y-axes.
    blur : bool, optional
        Blurs the edge of the image before registration.
    center: array, optional
        Location of rotation axis.
    algorithm : {str, function}
        One of the following string values.

        'art'
            Algebraic reconstruction technique :cite:`Kak:98`.
        'gridrec'
            Fourier grid reconstruction algorithm :cite:`Dowd:99`,
            :cite:`Rivers:06`.
        'mlem'
            Maximum-likelihood expectation maximization algorithm
            :cite:`Dempster:77`.
        'sirt'
            Simultaneous algebraic reconstruction technique.
        'tv'
            Total Variation reconstruction technique
            :cite:`Chambolle:11`.
        'grad'
            Gradient descent method with a constant step size

    upsample_factor : integer, optional
        The upsampling factor. Registration accuracy is
        inversely propotional to upsample_factor. 
    rin : scalar, optional
        The inner radius of blur function. Pixels inside
        rin is set to one.
    rout : scalar, optional
        The outer radius of blur function. Pixels outside
        rout is set to zero.
    save : bool, optional
        Saves projections and corresponding reconstruction
        for each algorithm iteration.
    debug : book, optional
        Provides debugging info such as iterations and error.

    Returns
    -------
    ndarray
        3D stack of projection images with jitter.
    ndarray
        Error array for each iteration.
    """

    # Needs scaling for skimage float operations.
    prj, scl = scale(prj)

    # Shift arrays
    sx = np.zeros((prj.shape[0]))
    sy = np.zeros((prj.shape[0]))

    conv = np.zeros((iters))

    # Pad images.
    npad = ((0, 0), (pad[1], pad[1]), (pad[0], pad[0]))
    prj = np.pad(prj, npad, mode='constant', constant_values=0)

    # Register each image frame-by-frame.
    for n in range(iters):
        # Reconstruct image.
        rec = recon(prj, ang, center=center, algorithm=algorithm)

        # Re-project data and obtain simulated data.
        sim = project(rec, ang, center=center, pad=False)

        # Blur edges.
        if blur:
            _prj = blur_edges(prj, rin, rout)
            _sim = blur_edges(sim, rin, rout)
        else:
            _prj = prj
            _sim = sim

        # Initialize error matrix per iteration.
        err = np.zeros((prj.shape[0]))

        # For each projection
        for m in range(prj.shape[0]):

            # Register current projection in sub-pixel precision
            shift, error, diffphase = register_translation(
                _prj[m], _sim[m], upsample_factor)
            err[m] = np.sqrt(shift[0]*shift[0] + shift[1]*shift[1])
            sx[m] += shift[0]
            sy[m] += shift[1]

            # Register current image with the simulated one
            tform = tf.SimilarityTransform(translation=(shift[1], shift[0]))
            prj[m] = tf.warp(prj[m], tform, order=5)

        if debug:
            print('iter=' + str(n) + ', err=' + str(np.linalg.norm(err)))
            conv[n] = np.linalg.norm(err)

        if save:
            dxchange.write_tiff(prj, fdir + '/tmp/iters/prj/prj')
            dxchange.write_tiff(sim, fdir + '/tmp/iters/sim/sim')
            dxchange.write_tiff(rec, fdir + '/tmp/iters/rec/rec')

    # Re-normalize data
    prj *= scl
    return prj, sx, sy, conv


def align_joint(
        prj, ang, fdir='.', iters=10, pad=(0, 0),
        blur=True, center=None, algorithm='sirt', 
        upsample_factor=10, rin=0.5, rout=0.8, 
        save=False, debug=True):
    """
    Aligns the projection image stack using the joint
    re-projection algorithm :cite:`Gursoy:17`.

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.
    ang : ndarray
        Projection angles in radians as an array.
    iters : scalar, optional
        Number of iterations of the algorithm.
    pad : list-like, optional
        Padding for projection images in x and y-axes.
    blur : bool, optional
        Blurs the edge of the image before registration.
    center: array, optional
        Location of rotation axis.
    algorithm : {str, function}
        One of the following string values.

        'art'
            Algebraic reconstruction technique :cite:`Kak:98`.
        'gridrec'
            Fourier grid reconstruction algorithm :cite:`Dowd:99`,
            :cite:`Rivers:06`.
        'mlem'
            Maximum-likelihood expectation maximization algorithm
            :cite:`Dempster:77`.
        'sirt'
            Simultaneous algebraic reconstruction technique.
        'tv'
            Total Variation reconstruction technique
            :cite:`Chambolle:11`.
        'grad'
            Gradient descent method with a constant step size

    upsample_factor : integer, optional
        The upsampling factor. Registration accuracy is
        inversely propotional to upsample_factor.
    rin : scalar, optional
        The inner radius of blur function. Pixels inside
        rin is set to one.
    rout : scalar, optional
        The outer radius of blur function. Pixels outside
        rout is set to zero.
    save : bool, optional
        Saves projections and corresponding reconstruction
        for each algorithm iteration.
    debug : book, optional
        Provides debugging info such as iterations and error.

    Returns
    -------
    ndarray
        3D stack of projection images with jitter.
    ndarray
        Error array for each iteration.
    """

    # Needs scaling for skimage float operations.
    prj, scl = scale(prj)

    # Shift arrays
    sx = np.zeros((prj.shape[0]))
    sy = np.zeros((prj.shape[0]))

    conv = np.zeros((iters))

    # Pad images.
    npad = ((0, 0), (pad[1], pad[1]), (pad[0], pad[0]))
    prj = np.pad(prj, npad, mode='constant', constant_values=0)

    # Initialization of reconstruction.
    rec = 1e-12 * np.ones((prj.shape[1], prj.shape[2], prj.shape[2]))

    # Register each image frame-by-frame.
    for n in range(iters):

        if np.mod(n, 1) == 0:
            _rec = rec

        # Reconstruct image.
        rec = recon(prj, ang, center=center, algorithm=algorithm,
                    num_iter=1, init_recon=_rec)

        # Re-project data and obtain simulated data.
        sim = project(rec, ang, center=center, pad=False)

        # Blur edges.
        if blur:
            _prj = blur_edges(prj, rin, rout)
            _sim = blur_edges(sim, rin, rout)
        else:
            _prj = prj
            _sim = sim

        # Initialize error matrix per iteration.
        err = np.zeros((prj.shape[0]))

        # For each projection
        for m in range(prj.shape[0]):

            # Register current projection in sub-pixel precision
            shift, error, diffphase = register_translation(
                _prj[m], _sim[m], upsample_factor)
            err[m] = np.sqrt(shift[0]*shift[0] + shift[1]*shift[1])
            sx[m] += shift[0]
            sy[m] += shift[1]

            # Register current image with the simulated one
            tform = tf.SimilarityTransform(translation=(shift[1], shift[0]))
            prj[m] = tf.warp(prj[m], tform, order=5)

        if debug:
            print('iter=' + str(n) + ', err=' + str(np.linalg.norm(err)))
            conv[n] = np.linalg.norm(err)

        if save:
            dxchange.write_tiff(prj, 'tmp/iters/prj/prj')
            dxchange.write_tiff(sim, 'tmp/iters/sim/sim')
            dxchange.write_tiff(rec, 'tmp/iters/rec/rec')

    # Re-normalize data
    prj *= scl
    return prj, sx, sy, conv


def tilt(obj, rad=0, phi=0):
    """
    Tilt object at a given angle from the rotation axis.

    Warning
    -------
    Not implemented yet.

    Parameters
    ----------
    obj : ndarray
        3D discrete object.
    rad : scalar, optional
        Radius in polar cordinates to define tilt angle.
        The value is between 0 and 1, where 0 means no tilt
        and 1 means a tilt of 90 degrees. The tilt angle
        can be obtained by arcsin(rad).
    phi : scalar, optional
        Angle in degrees to define tilt direction from the
        rotation axis. 0 degree means rotation in sagittal
        plane and 90 degree means rotation in coronal plane.

    Returns
    -------
    ndarray
        Tilted 3D object.
    """
    pass


def add_jitter(prj, low=0, high=1):
    """
    Simulates jitter in projection images. The jitter
    is simulated by drawing random samples from a uniform
    distribution over the half-open interval [low, high).

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.
    low : float, optional
        Lower boundary of the output interval. All values
        generated will be greater than or equal to low. The
        default value is 0.
    high : float
        Upper boundary of the output interval. All values
        generated will be less than high. The default value
        is 1.0.

    Returns
    -------
    ndarray
        3D stack of projection images with jitter.
    """
    from skimage import transform as tf

    # Needs scaling for skimage float operations.
    prj, scl = scale(prj)

    # Random jitter parameters are drawn from uniform distribution.
    jitter = np.random.uniform(low, high, size=(prj.shape[0], 2))

    for m in range(prj.shape[0]):
        tform = tf.SimilarityTransform(translation=jitter[m])
        prj[m] = tf.warp(prj[m], tform, order=0)

    # Re-scale back to original values.
    prj *= scl
    return prj, jitter[:, 0], jitter[:, 1]


def add_noise(prj, ratio=0.05):
    """
    Adds Gaussian noise with zero mean and a given standard
    deviation as a ratio of the maximum value in data.

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.
    ratio : float, optional
        Ratio of the standard deviation of the Gaussian noise
        distribution to the maximum value in data.

    Returns
    -------
    ndarray
        3D stack of projection images with added Gaussian noise.
    """
    std = prj.max() * ratio
    noise = np.random.normal(0, std, size=prj.shape)
    return prj + noise.astype('float32')


def scale(prj):
    """
    Linearly scales the projection images in the range
    between -1 and 1.

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.

    Returns
    -------
    ndarray
        Scaled 3D stack of projection images.
    """
    scl = max(abs(prj.max()), abs(prj.min()))
    prj /= scl
    return prj, scl


def blur_edges(prj, low=0, high=0.8):
    """
    Blurs the edge of the projection images.

    Parameters
    ----------
    prj : ndarray
        3D stack of projection images. The first dimension
        is projection axis, second and third dimensions are
        the x- and y-axes of the projection image, respectively.
    low : scalar, optional
        Min ratio of the blurring frame to the image size.
    high : scalar, optional
        Max ratio of the blurring frame to the image size.

    Returns
    -------
    ndarray
        Edge-blurred 3D stack of projection images.
    """
    _prj = prj.copy()
    dx, dy, dz = _prj.shape
    rows, cols = np.mgrid[:dy, :dz]
    rad = np.sqrt((rows - dy/2)**2 + (cols - dz/2)**2)
    mask = np.zeros((dy, dz))
    rmin, rmax = low * rad.max(), high * rad.max()
    mask[rad < rmin] = 1
    mask[rad > rmax] = 0
    zone = np.logical_and(rad >= rmin, rad <= rmax)
    mask[zone] = (rmax - rad[zone]) / (rmax - rmin)
    feathered = np.empty((dy, dz), dtype=np.uint8)
    _prj *= mask
    return _prj


def shift_images(prj, sx, sy):
    """
    Shift projections images for a given set of shift
    values in horizontal and vertical directions.
    """

    from skimage import transform as tf
    from skimage.feature import register_translation

    # Needs scaling for skimage float operations.
    prj, scl = scale(prj)

    # For each projection
    for m in range(prj.shape[0]):
        tform = tf.SimilarityTransform(translation=(sy[m], sx[m]))
        prj[m] = tf.warp(prj[m], tform, order=5)

    # Re-normalize data
    prj *= scl

    return prj
