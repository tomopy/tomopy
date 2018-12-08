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
from tomopy.misc.npmath import gauss1d, calc_affine_transform
from scipy.signal import medfilt, medfilt2d
from scipy.optimize import curve_fit
from scipy.ndimage import affine_transform, shift
from collections import namedtuple

import dxchange


logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy, Chen Zhang"
__copyright__ = "Copyright (c) 2016-17, UChicago Argonne, LLC."
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


def detector_drift_adjust_aps_1id(imgstacks, slit_cnr_ref):
    """
    Adjust each still image based on the slit corners and generate report fig

    Parameters
    ----------
    imgstacks : np.ndarray
        tomopy images stacks (axis_0 is the oemga direction)
    slit_cnr_ref : np.ndarray
        reference slit corners from white field images

    Returns
    -------
    np.ndarray
        adjusted imgstacks
    np.ndarray
        detected corners on each still image
    np.ndarray
        transformation matrices used to adjust each image
    """
    # -- init slit corners for still images
    proj_cnrs = []

    # -- init affine transformation matrix
    img_correct_F = np.ones((imgstacks.shape[0], 3, 3))

    # -- work throgh each image
    for n_img in range(imgstacks.shape[0]):
        # detect the corners
        proj_cnr = find_slits_corners_aps_1id(imgstacks[n_img, :, :],
                                              method='quadrant+',
                                              )
        proj_cnrs.append(proj_cnr)

        # calcualte the affine transfomration required
        img_correct_F[n_img, :, :] = calc_affine_transform(proj_cnr,
                                                           slit_cnr_ref,
                                                           )

        # adjust the image
        if np.linalg.norm(img_correct_F[n_img, 0:2, 0:2] - np.eye(2)) < 1e-4:
            # shift only when rotation is small
            imgstacks[n_img, :, :] = shift(imgstacks[n_img, :, :],
                                           img_correct_F[n_img, 0:2, 2],
                                           )
        else:
            # perform full affine transformation
            # NOTE:
            #  This is particular slow, need optimization
            imgstacks[n_img, :, :] = affine_transform(imgstacks[n_img, :, :],                # input image
                                                      # rotation matrix
                                                      img_correct_F[n_img,
                                                                    0:2, 0:2],
                                                      # offset vector
                                                      offset=img_correct_F[n_img,
                                                                           0:2, 2],
                                                      )
    # convert proj_cnrs to np.array
    proj_cnrs = np.stack(proj_cnrs, axis=0)

    return imgstacks, proj_cnrs, img_correct_F
