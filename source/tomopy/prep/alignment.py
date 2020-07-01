#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2016-2019. UChicago Argonne, LLC. This software was produced  #
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
import concurrent.futures as cf
import tomopy.util.mproc as mproc
import logging

from skimage import transform as tf
from skimage.feature import register_translation
from tomopy.recon.algorithm import recon
from tomopy.sim.project import project
from tomopy.misc.npmath import gauss1d, calc_affine_transform
from tomopy.util.misc import write_tiff
from scipy.signal import medfilt, medfilt2d
from scipy.optimize import curve_fit
from scipy.ndimage import affine_transform
from scipy.ndimage import map_coordinates
from collections import namedtuple

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Chen Zhang, Nghia Vo"
__copyright__ = "Copyright (c) 2016-19, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['align_seq',
           'align_joint',
           'scale',
           'tilt',
           'add_jitter',
           'add_noise',
           'blur_edges',
           'shift_images',
           'find_slits_corners_aps_1id',
           'calc_slit_box_aps_1id',
           'remove_slits_aps_1id',
           'distortion_correction_proj',
           'distortion_correction_sino',
           'load_distortion_coefs',           
           ]


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
            write_tiff(prj, fdir + '/tmp/iters/prj', n)
            write_tiff(sim, fdir + '/tmp/iters/sim', n)
            write_tiff(rec, fdir + '/tmp/iters/rec', n)

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

    extra_kwargs = {}
    if algorithm != 'gridrec':
        extra_kwargs['num_iter'] = 1

    # Register each image frame-by-frame.
    for n in range(iters):

        if np.mod(n, 1) == 0:
            _rec = rec

        # Reconstruct image.
        rec = recon(prj, ang, center=center, algorithm=algorithm,
                    init_recon=_rec, **extra_kwargs)

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
            write_tiff(prj, 'tmp/iters/prj', n)
            write_tiff(sim, 'tmp/iters/sim', n)
            write_tiff(rec, 'tmp/iters/rec', n)

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
    rad = np.sqrt((rows - dy / 2)**2 + (cols - dz / 2)**2)
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

    # Needs scaling for skimage float operations.
    prj, scl = scale(prj)

    # For each projection
    for m in range(prj.shape[0]):
        tform = tf.SimilarityTransform(translation=(sy[m], sx[m]))
        prj[m] = tf.warp(prj[m], tform, order=5)

    # Re-normalize data
    prj *= scl

    return prj


def find_slits_corners_aps_1id(img,
                               method='quadrant+',
                               medfilt2_kernel_size=3,
                               medfilt_kernel_size=23,
                               ):
    """
    Automatically locate the slit box location by its four corners.

    NOTE:
    The four slits that form a binding box is the current setup at aps_1id,
    which reduce the illuminated region on the detector. Since the slits are
    stationary, they can serve as a reference to check detector drifting
    during the scan. Technically, the four slits should be used to find
    the transformation matrix (not necessarily affine) to correct the image.
    However, since we are dealing with 2D images with very little distortion,
    affine transformation matrices were used for approximation. Therefore
    the "four corners" are used instead of all four slits.

    Parameters
    ----------
    img : np.ndarray
        2D images
    method : str,  ['simple', 'quadrant', 'quadrant+'], optional
        method for auto detecting slit corners
            - simple    :: assume a rectange slit box, fast but less accurate
                           (1 pixel precision)
            - quadrant  :: subdivide the image into four quandrant, then use
                           an explicit method to find the corner
                           (1 pixel precision)
            - quadrant+ :: similar to quadrant, but use curve_fit (gauss1d) to
                           find the corner
                           (0.1 pixel precision)
    medfilt2_kernel_size : int, optional
        2D median filter kernel size for noise reduction
    medfilt_kernel_size : int, optional
        1D median filter kernel size for noise reduction

    Returns
    -------
    tuple
        autodetected slit corners (counter-clockwise order)
        (upperLeft, lowerLeft, lowerRight, upperRight)
    """
    img = medfilt2d(np.log(img.astype(np.float64)),
                    kernel_size=medfilt2_kernel_size,
                    )
    rows, cols = img.shape

    # simple method is simple, therefore it stands out
    if method.lower() == 'simple':
        # assuming a rectangle type slit box
        col_std = medfilt(np.std(img, axis=0), kernel_size=medfilt_kernel_size)
        row_std = medfilt(np.std(img, axis=1), kernel_size=medfilt_kernel_size)
        # NOTE: in the tiff img
        #  x is col index, y is the row index  ==> key point here !!!
        #  img slicing is doen with img[row_idx, col_idx]
        #  ==> so the image idx and corner position are FLIPPED!
        _left = np.argmax(np.gradient(col_std))
        _right = np.argmin(np.gradient(col_std))
        _top = np.argmax(np.gradient(row_std))
        _bottom = np.argmin(np.gradient(row_std))

        cnrs = np.array([[_left, _top],
                         [_left, _bottom],
                         [_right, _bottom],
                         [_right, _top],
                         ])
    else:
        # predefine all quadrants
        # Here let's assume that the four corners of the slit box are in the
        # four quadrant defined by the center of the image
        # i.e.
        #  uppper left quadrant: img[0     :cnt[1], 0     :cnt[0]]  => quadarnt origin =  (0,           0)
        #  lower  left quadrant: img[cnt[1]:      , 0     :cnt[0]]  => quadarnt origin =  (cnt[0],      0)
        #  lower right quadrant: img[cnt[1]:      , cnt[0]:      ]  => quadarnt origin =  (cnt[0], cnt[1])
        # upper right quadrant: img[0     :cnt[1], cnt[0]:      ]  => quadarnt
        # origin =  (0,      cnt[1])
        # center of image that defines FOUR quadrants
        cnt = [int(cols / 2), int(rows / 2)]
        Quadrant = namedtuple('Quadrant', 'img col_func, row_func')
        quadrants = [Quadrant(img=img[0:cnt[1], 0:cnt[0]], col_func=np.argmax, row_func=np.argmax),  # upper left,  1st quadrant
                     # lower left,  2nd quadrant
                     Quadrant(img=img[cnt[1]:, 0:cnt[0]],
                              col_func=np.argmax, row_func=np.argmin),
                     # lower right, 3rd quadrant
                     Quadrant(img=img[cnt[1]:, cnt[0]:],
                              col_func=np.argmin, row_func=np.argmin),
                     # upper right, 4th quadrant
                     Quadrant(img=img[0:cnt[0], cnt[1]:],
                              col_func=np.argmin, row_func=np.argmax),
                     ]
        # the origin in each quadrants ==> easier to set it here
        quadrantorigins = np.array([[0, 0],  # upper left,  1st quadrant
                                    [0, cnt[1]],  # lower left,  2nd quadrant
                                    # lower right, 3rd quadrant
                                    [cnt[0], cnt[1]],
                                    [cnt[1], 0],  # upper right, 4th quadrant
                                    ])
        # init four corners
        cnrs = np.zeros((4, 2))
        if method.lower() == 'quadrant':
            # the standard quadrant method
            for i, q in enumerate(quadrants):
                cnrs[i, :] = np.array([q.col_func(np.gradient(medfilt(np.std(q.img, axis=0), kernel_size=medfilt_kernel_size))),  # x is col_idx
                                       q.row_func(
                    np.gradient(
                        medfilt(
                            np.std(
                                q.img,
                                axis=1),
                            kernel_size=medfilt_kernel_size))),
                    # y is row_idx
                ])
            # add the origin offset back
            cnrs = cnrs + quadrantorigins
        elif method.lower() == 'quadrant+':
            # use Gaussian curve fitting to achive subpixel precision
            # TODO:
            # improve the curve fitting with Lorentz and Voigt fitting function
            for i, q in enumerate(quadrants):
                # -- find x subpixel position
                cnr_x_guess = q.col_func(
                    np.gradient(
                        medfilt(
                            np.std(
                                q.img,
                                axis=0),
                            kernel_size=medfilt_kernel_size)))
                # isolate the strongest peak to fit
                tmpx = np.arange(cnr_x_guess - 10, cnr_x_guess + 11)
                tmpy = np.gradient(np.std(q.img, axis=0))[tmpx]
                # tmpy[0] is the value from the highest/lowest pixle
                # tmpx[0] is basically cnr_x_guess
                # 5.0 is the guessted std,
                coeff, _ = curve_fit(gauss1d, tmpx, tmpy,
                                     p0=[tmpy[0], tmpx[0], 5.0],
                                     maxfev=int(1e6),
                                     )
                cnrs[i, 0] = coeff[1]  # x position
                # -- find y subpixel positoin
                cnr_y_guess = q.row_func(
                    np.gradient(
                        medfilt(
                            np.std(
                                q.img,
                                axis=1),
                            kernel_size=medfilt_kernel_size)))
                # isolate the peak (x, y here is only associated with the peak)
                tmpx = np.arange(cnr_y_guess - 10, cnr_y_guess + 11)
                tmpy = np.gradient(np.std(q.img, axis=1))[tmpx]
                coeff, _ = curve_fit(gauss1d, tmpx, tmpy,
                                     p0=[tmpy[0], tmpx[0], 5.0],
                                     maxfev=int(1e6),
                                     )
                cnrs[i, 1] = coeff[1]  # y posiiton
            # add the quadrant shift back
            cnrs = cnrs + quadrantorigins

        else:
            raise NotImplementedError(
                "Available methods are: simple, quadrant, quadrant+")

    # return the slit corner detected
    return cnrs


def calc_slit_box_aps_1id(slit_box_corners, inclip=(1, 10, 1, 10)):
    """
    Calculate the clip box based on given slip corners.

    Parameters
    ----------
    slit_box_corners : np.ndarray
        Four corners of the slit box as a 4x2 matrix
    inclip : tuple, optional
        Extra inclipping to avoid clipping artifacts

    Returns
    -------
    Tuple:
        Cliping indices as a tuple of four
        (clipFromTop, clipToBottom, clipFromLeft, clipToRight)

    """
    return (
        np.floor(slit_box_corners[:, 0].min()).astype(
            int) + inclip[0],  # clip top    row
        np.ceil(slit_box_corners[:, 0].max()).astype(
            int) - inclip[1],  # clip bottom row
        np.floor(slit_box_corners[:, 1].min()).astype(
            int) + inclip[2],  # clip left   col
        np.ceil(slit_box_corners[:, 1].max()).astype(
            int) - inclip[3],  # clip right  col
    )


def remove_slits_aps_1id(imgstacks, slit_box_corners, inclip=(1, 10, 1, 10)):
    """
    Remove the slits from still images

    Parameters
    ----------
    imgstacks : np.ndarray
        tomopy images stacks (axis_0 is the oemga direction)
    slit_box_corners : np.ndarray
        four corners of the slit box
    inclip : tuple, optional
        Extra inclipping to avoid clipping artifacts

    Returns
    -------
    np.ndarray
        tomopy images stacks without regions outside slits
    """
    xl, xu, yl, yu = calc_slit_box_aps_1id(slit_box_corners, inclip=inclip)
    return imgstacks[:, yl:yu, xl:xu]


def detector_drift_adjust_aps_1id(imgstacks,
                                  slit_cnr_ref,
                                  medfilt2_kernel_size=3,
                                  medfilt_kernel_size=3,
                                  ncore=None,
                                  ):
    """
    Adjust each still image based on the slit corners and generate report fig

    Parameters
    ----------
    imgstacks : np.ndarray
        tomopy images stacks (axis_0 is the oemga direction)
    slit_cnr_ref : np.ndarray
        reference slit corners from white field images
    medfilt2_kernel_size : int, optional
        2D median filter kernel size for slit conner detection
    medfilt_kernel_size  : int, optional
        1D median filter kernel size for slit conner detection
    ncore : int, optional
        number of cores used for speed up

    Returns
    -------
    np.ndarray
        adjusted imgstacks
    np.ndarray
        detected corners on each still image
    np.ndarray
        transformation matrices used to adjust each image
    """
    ncore = mproc.mp.cpu_count() - 1 if ncore is None else ncore

    def quick_diff(x): return np.amax(np.absolute(x))

    # -- find all projection corners (slow)
    # NOTE:
    #  Here we are using an iterative approach to find stable slit corners
    #  from each image
    #  1. calculate all slit corners with the given kernel size, preferably
    #     a small one for speed.
    #  2. double the kernel size and calculate again, but this time we are
    #     checking whether the slit corners are stable.
    #  3. find the ids (n_imgs) for those that are difficult, continue
    #     increasing the kernel size until all slit corners are found, or max
    #     number of iterations.
    #  4. move on to next step.
    nlist = range(imgstacks.shape[0])
    proj_cnrs = _calc_proj_cnrs(imgstacks, ncore, nlist,
                                'quadrant+',
                                medfilt2_kernel_size,
                                medfilt_kernel_size,
                                )
    cnrs_found = np.array([quick_diff(proj_cnrs[n, :, :] - slit_cnr_ref) < 15
                           for n in nlist])
    kernels = [(medfilt2_kernel_size+2*i, medfilt_kernel_size+2*j)
               for i in range(15)
               for j in range(15)]
    counter = 0

    while not cnrs_found.all():
        nlist = [idx for idx, cnr_found in enumerate(cnrs_found)
                 if not cnr_found]
        # NOTE:
        #   Check to see if we run out of candidate kernels:
        if counter > len(kernels):
            # we are giving up here...
            for idx, n_img in enumerate(nlist):
                proj_cnrs[n_img, :, :] = slit_cnr_ref
            break
        else:
            # test with differnt 2D and 1D kernels
            ks2d, ks1d = kernels[counter]

        _cnrs = _calc_proj_cnrs(imgstacks, ncore, nlist,
                                'quadrant+', ks2d, ks1d)
        for idx, _cnr in enumerate(_cnrs):
            n_img = nlist[idx]
            cnr = proj_cnrs[n_img, :, :]  # previous results
            # NOTE:
            #  The detector corner should not be far away from reference
            #  -> adiff < 15
            #  The detected corner should be stable
            #  -> rdiff < 0.1 (pixel)s
            adiff = quick_diff(_cnr - slit_cnr_ref)
            rdiff = quick_diff(_cnr - cnr)
            if rdiff < 0.1 and adiff < 15:
                cnrs_found[n_img] = True
            else:
                # update results
                proj_cnrs[n_img, :, :] = _cnr  # update results for next iter

        # next
        counter += 1

    # -- calculate affine transformation (fast)
    img_correct_F = np.ones((imgstacks.shape[0], 3, 3))
    for n_img in range(imgstacks.shape[0]):
        img_correct_F[n_img, :, :] = calc_affine_transform(
            proj_cnrs[n_img, :, :], slit_cnr_ref)

    # -- apply affine transformation (slow)
    tmp = []
    with cf.ProcessPoolExecutor(ncore) as e:
        for n_img in range(imgstacks.shape[0]):
            tmp.append(e.submit(affine_transform,
                                # input image
                                imgstacks[n_img, :, :],
                                # rotation matrix
                                img_correct_F[n_img, 0:2, 0:2],
                                # offset vector
                                offset=img_correct_F[n_img, 0:2,  2],
                                )
                       )
    imgstacks = np.stack([me.result() for me in tmp], axis=0)

    return imgstacks, proj_cnrs, img_correct_F


def _calc_proj_cnrs(imgs,
                    ncore,
                    nlist,
                    method,
                    medfilt2_kernel_size,
                    medfilt_kernel_size,
                    ):
    """
    Private function calculate slit corners concurrently

    Parameters
    ----------
    imgs : ndarray
        tomopy images stacks (axis_0 is the oemga direction)
    ncore : int
        number of cores to use
    nlist : list of int
        index of images to be processed
    method : str
        slit corner detection method name
    medfilt2_kernel_size : int
        2D median filter kernel size, must be odd
    medfilt_kernel_size : int
        1D median filter kernel size, must be odd

    Returns
    -------
    np.3darray
        detected corners on each still image
    """
    tmp = []
    with cf.ProcessPoolExecutor(ncore) as e:
        for n_img in nlist:
            tmp.append(e.submit(find_slits_corners_aps_1id,
                                imgs[n_img, :, :],
                                method=method,
                                medfilt2_kernel_size=medfilt2_kernel_size,
                                medfilt_kernel_size=medfilt_kernel_size,
                                )
                       )
    return np.stack([me.result() for me in tmp], axis=0)


def distortion_correction_proj(tomo, xcenter, ycenter, list_fact,
                                ncore=None, nchunk=None):
    """
    Apply distortion correction to projections using the polynomial model.
    Coefficients are calculated using Vounwarp package.:cite:`Vo:15`

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    xcenter : float
        Center of distortion in x-direction. From the left of the image.
    ycenter : float
        Center of distortion in y-direction. From the top of the image. 
    list_fact : list of floats
        Polynomial coefficients of the backward model.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    arr = mproc.distribute_jobs(
        tomo,
        func=_distortion_correction_proj,
        args=(xcenter, ycenter, list_fact),
        axis=0,
        ncore=ncore,
        nchunk=nchunk)
    return arr


def _unwarp_image_backward(mat, xcenter, ycenter, list_fact):
    """
    Unwarp an image using the polynomial model.
    
    Parameters
    ----------
    mat : 2D array.
    xcenter : float 
            Center of distortion in x-direction. From the left of the image.
    ycenter : float
            Center of distortion in y-direction. From the top of the image.
    list_fact : list of floats 
            Polynomial coefficients of the backward model.
    
    Returns
    -------
    2D array
        Corrected image.
    """
    (height, width) = mat.shape
    xu_list = np.arange(width) - xcenter
    yu_list = np.arange(height) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat**2 + yu_mat**2)
    fact_mat = np.sum(
        np.asarray([factor * ru_mat**i for i,
                    factor in enumerate(list_fact)]), axis=0)
    xd_mat = np.float32(np.clip(xcenter + fact_mat * xu_mat, 0, width - 1))
    yd_mat = np.float32(np.clip(ycenter + fact_mat * yu_mat, 0, height - 1))
    indices = np.reshape(yd_mat, (-1, 1)), np.reshape(xd_mat, (-1, 1))
    mat = map_coordinates(mat, indices, order=1, mode='reflect')
    return mat.reshape((height, width))


def _distortion_correction_proj(tomo, xcenter, ycenter, list_fact):
    for m in np.arange(tomo.shape[0]):
        proj = tomo[m, :, :]
        proj = _unwarp_image_backward(proj, xcenter, ycenter, list_fact)        
        tomo[m, :, :] = proj


def distortion_correction_sino(tomo, ind, xcenter, ycenter, list_fact):
    """
    Generate an unwarped sinogram of a 3D tomographic data using
    the polynomial model. Coefficients are calculated using Vounwarp
    package :cite:`Vo:15`

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int
        Index of the unwarped sinogram.
    xcenter : float
        Center of distortion in x-direction. From the left of the image.
    ycenter : float
        Center of distortion in y-direction. From the top of the image.         
    list_fact : list of floats
        Polynomial coefficients of the backward model.

    Returns
    -------
    2D array
        Corrected sinogram.
    """
    (depth, height, width) = tomo.shape
    xu_list = np.arange(0, width) - xcenter
    yu = ind - ycenter
    ru_list = np.sqrt(xu_list**2 + yu**2)
    flist = np.sum(
        np.asarray([factor * ru_list**i for i,
                    factor in enumerate(list_fact)]), axis=0)
    xd_list = np.clip(xcenter + flist * xu_list, 0, width - 1)
    yd_list = np.clip(ycenter + flist * yu, 0, height - 1)
    yd_min = np.int16(np.floor(np.amin(yd_list)))
    yd_max = np.int16(np.ceil(np.amax(yd_list))) + 1
    yd_list = yd_list - yd_min 
    sino = np.zeros((depth, width), dtype=np.float32)
    indices = yd_list, xd_list
    for i in np.arange(depth):
        sino[i] = map_coordinates(
            tomo[i, yd_min:yd_max, :], indices, order=1, mode='reflect')
    return sino


def load_distortion_coefs(file_path):
        """
        Load distortion coefficients from a text file.
        Order of the infor in the text file:
        xcenter
        ycenter
        factor_0
        factor_1
        factor_2
        ..

        Parameters
        ----------
        file_path: Path to the file.

        Returns
        -------
        Tuple of (xcenter, ycenter, list_fact).
        """
        with open(file_path, 'r') as f:
            x = f.read().splitlines()
            list_data = []
            for i in x:
                list_data.append(float(i.split()[-1]))
        xcenter = list_data[0]
        ycenter = list_data[1]
        list_fact = list_data[2:]
        return xcenter, ycenter, list_fact
