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
Module for data correction and masking functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.ndimage import filters
import tomopy.util.mproc as mproc
import tomopy.util.dtype as dtype
import tomopy.util.extern as extern
import logging
import warnings
import numexpr as ne
import concurrent.futures as cf
from scipy.signal import medfilt2d

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__credits__ = "Mark Rivers, Xianghui Xiao"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['adjust_range',
           'circ_mask',
           'gaussian_filter',
           'median_filter',
           'median_filter_cuda',
           'sobel_filter',
           'remove_nan',
           'remove_neg',
           'remove_outlier',
           'remove_outlier1d',
           'remove_outlier_cuda',
           'remove_ring',
           'enhance_projs_aps_1id',
           ]


def adjust_range(arr, dmin=None, dmax=None):
    """
    Change dynamic range of values in an array.

    Parameters
    ----------
    arr : ndarray
        Input array.

    dmin, dmax : float, optional
        Mininum and maximum values to rescale data.

    Returns
    -------
    ndarray
        Output array.
    """
    if dmax is None:
        dmax = np.max(arr)
    if dmin is None:
        dmin = np.min(arr)
    if dmax < np.max(arr):
        arr[arr > dmax] = dmax
    if dmin > np.min(arr):
        arr[arr < dmin] = dmin
    return arr


def gaussian_filter(arr, sigma=3, order=0, axis=0, ncore=None):
    """
    Apply Gaussian filter to 3D array along specified axis.

    Parameters
    ----------
    arr : ndarray
        Input array.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard deviations
        of the Gaussian filter are given for each axis as a sequence, or
        as a single number, in which case it is equal for all axes.
    order : {0, 1, 2, 3} or sequence from same set, optional
        Order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. An order of 1, 2, or 3
        corresponds to convolution with the first, second or third
        derivatives of a Gaussian. Higher order derivatives are not
        implemented
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
        3D array of same shape as input.
    """
    arr = dtype.as_float32(arr)
    out = np.empty_like(arr)

    if ncore is None:
        ncore = mproc.mp.cpu_count()

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)]*arr.ndim
        for i in range(arr.shape[axis]):
            slc[axis] = i
            e.submit(filters.gaussian_filter, arr[tuple(slc)], sigma,
                     order=order, output=out[tuple(slc)])
    return out


def median_filter(arr, size=3, axis=0, ncore=None):
    """
    Apply median filter to 3D array along specified axis.

    Parameters
    ----------
    arr : ndarray
        Input array.
    size : int, optional
        The size of the filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """
    arr = dtype.as_float32(arr)
    out = np.empty_like(arr)

    if ncore is None:
        ncore = mproc.mp.cpu_count()

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)]*arr.ndim
        for i in range(arr.shape[axis]):
            slc[axis] = i
            e.submit(filters.median_filter, arr[tuple(slc)], size=(size, size),
                     output=out[tuple(slc)])
    return out


def median_filter_cuda(arr, size=3, axis=0):
    """
    Apply median filter to 3D array along 0 axis with GPU support.
    The winAllow is for A6000, Tian X support 3 to 8

    Parameters
    ----------
    arr : ndarray
        Input array.
    size : int, optional
        The size of the filter.
    axis : int, optional
        Axis along which median filtering is performed.

    Returns
    -------
    ndarray
        Median filtered 3D array.

    Example
    -------
    import tomocuda
    tomocuda.remove_outlier_cuda(arr, dif, 5)

    For more information regarding install and using tomocuda, check
    https://github.com/kyuepublic/tomocuda for more information
    """

    try:
        import tomocuda

        winAllow = range(2, 16)

        if(axis != 0):
            arr = np.swapaxes(arr, 0, axis)

        if size in winAllow:
            loffset = int(size/2)
            roffset = int((size-1)/2)
            prjsize = arr.shape[0]
            imsizex = arr.shape[2]
            imsizey = arr.shape[1]

            filter = tomocuda.mFilter(imsizex, imsizey, prjsize, size)
            out = np.zeros(
                shape=(prjsize, imsizey, imsizex), dtype=np.float32)

            for step in range(prjsize):
                # im_noisecu = arr[:][step][:].astype(np.float32)
                im_noisecu = arr[step].astype(np.float32)
                im_noisecu = np.lib.pad(im_noisecu, ((loffset, roffset),
                                                     (loffset, roffset)), 'symmetric')
                im_noisecu = im_noisecu.flatten()

                filter.setCuImage(im_noisecu)
                filter.run2DFilter(size)
                results = filter.retreive()
                results = results.reshape(imsizey, imsizex)
                out[step] = results

            if(axis != 0):
                out = np.swapaxes(out, 0, axis)
        else:
            warnings.warn("Window size not support, using cpu median filter")
            out = median_filter(arr, size, axis)

    except ImportError:
        warnings.warn("The tomocuda is not support, using cpu median filter")
        out = median_filter(arr, size, axis)

    return out


def sobel_filter(arr, axis=0, ncore=None):
    """
    Apply Sobel filter to 3D array along specified axis.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, optional
        Axis along which sobel filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
        3D array of same shape as input.
    """
    arr = dtype.as_float32(arr)
    out = np.empty_like(arr)

    if ncore is None:
        ncore = mproc.mp.cpu_count()

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)]*arr.ndim
        for i in range(arr.shape[axis]):
            slc[axis] = i
            e.submit(filters.sobel, arr[slc], output=out[slc])
    return out


def remove_nan(arr, val=0., ncore=None):
    """
    Replace NaN values in array with a given value.

    Parameters
    ----------
    arr : ndarray
        Input array.
    val : float, optional
        Values to be replaced with NaN values in array.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = dtype.as_float32(arr)
    val = np.float32(val)

    with mproc.set_numexpr_threads(ncore):
        ne.evaluate('where(arr!=arr, val, arr)', out=arr)

    return arr


def remove_neg(arr, val=0., ncore=None):
    """
    Replace negative values in array with a given value.

    Parameters
    ----------
    arr : ndarray
        Input array.
    val : float, optional
        Values to be replaced with negative values in array.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = dtype.as_float32(arr)
    val = np.float32(val)

    with mproc.set_numexpr_threads(ncore):
        ne.evaluate('where(arr<0, val, arr)', out=arr)
    return arr


def remove_outlier(arr, dif, size=3, axis=0, ncore=None, out=None):
    """
    Remove high intensity bright spots from a N-dimensional array by chunking
    along the specified dimension, and performing (N-1)-dimensional median
    filtering along the other dimensions.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which to chunk.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result. If same as arr, process
        will be done in-place.

    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = dtype.as_float32(arr)
    dif = np.float32(dif)

    tmp = np.empty_like(arr)

    ncore, chnk_slices = mproc.get_ncore_slices(arr.shape[axis], ncore=ncore)

    filt_size = [size]*arr.ndim
    filt_size[axis] = 1

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)]*arr.ndim
        for i in range(ncore):
            slc[axis] = chnk_slices[i]
            e.submit(filters.median_filter, arr[tuple(slc)], size=filt_size,
                     output=tmp[tuple(slc)])

    with mproc.set_numexpr_threads(ncore):
        out = ne.evaluate('where(arr-tmp>=dif,tmp,arr)', out=out)

    return out


def remove_outlier1d(arr, dif, size=3, axis=0, ncore=None, out=None):
    """
    Remove high intensity bright spots from an array, using a one-dimensional
    median filter along the specified axis.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result. If same as arr, process
        will be done in-place.

    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = dtype.as_float32(arr)
    dif = np.float32(dif)

    tmp = np.empty_like(arr)

    other_axes = [i for i in range(arr.ndim) if i != axis]
    largest = np.argmax([arr.shape[i] for i in other_axes])
    lar_axis = other_axes[largest]
    ncore, chnk_slices = mproc.get_ncore_slices(
        arr.shape[lar_axis], ncore=ncore)
    filt_size = [1]*arr.ndim
    filt_size[axis] = size

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)]*arr.ndim
        for i in range(ncore):
            slc[lar_axis] = chnk_slices[i]
            e.submit(filters.median_filter, arr[slc], size=filt_size,
                     output=tmp[slc], mode='mirror')

    with mproc.set_numexpr_threads(ncore):
        out = ne.evaluate('where(arr-tmp>=dif,tmp,arr)', out=out)

    return out


def remove_outlier_cuda(arr, dif, size=3, axis=0):
    """
    Remove high intensity bright spots from a 3D array along axis 0
    dimension using GPU.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which outlier removal is performed.

    Returns
    -------
    ndarray
       Corrected array.

    Example
    -------
    >>> import tomocuda
    >>> tomocuda.remove_outlier_cuda(arr, dif, 5)

    For more information regarding install and using tomocuda, check
    https://github.com/kyuepublic/tomocuda for more information

    """

    arr = dtype.as_float32(arr)
    dif = np.float32(dif)

    try:
        import tomocuda

        winAllow = range(2, 16)

        if(axis != 0):
            arr = np.swapaxes(arr, 0, axis)

        if size in winAllow:
            prjsize = arr.shape[0]
            loffset = int(size/2)
            roffset = int((size-1)/2)
            imsizex = arr.shape[2]
            imsizey = arr.shape[1]

            filter = tomocuda.mFilter(imsizex, imsizey, prjsize, size)
            out = np.zeros(shape=(prjsize, imsizey, imsizex), dtype=np.float32)

            for step in range(prjsize):
                im_noisecu = arr[step].astype(np.float32)
                im_noisecu = np.lib.pad(im_noisecu, ((loffset, roffset),
                                                     (loffset, roffset)), 'symmetric')
                im_noisecu = im_noisecu.flatten()

                filter.setCuImage(im_noisecu)
                filter.run2DRemoveOutliner(size, dif)
                results = filter.retreive()
                results = results.reshape(imsizey, imsizex)
                out[step] = results

            if(axis != 0):
                out = np.swapaxes(out, 0, axis)
        else:
            warnings.warn("Window size not support, using cpu outlier removal")
            out = remove_outlier(arr, dif, size)

    except ImportError:
        warnings.warn("The tomocuda is not support, using cpu outlier removal")
        out = remove_outlier(arr, dif, size)

    return out


def remove_ring(rec, center_x=None, center_y=None, thresh=300.0,
                thresh_max=300.0, thresh_min=-100.0, theta_min=30,
                rwidth=30, int_mode='WRAP', ncore=None, nchunk=None, out=None):
    """
    Remove ring artifacts from images in the reconstructed domain.
    Descriptions of parameters need to be more clear for sure.

    Parameters
    ----------
    arr : ndarray
        Array of reconstruction data
    center_x : float, optional
        abscissa location of center of rotation
    center_y : float, optional
        ordinate location of center of rotation
    thresh : float, optional
        maximum value of an offset due to a ring artifact
    thresh_max : float, optional
        max value for portion of image to filter
    thresh_min : float, optional
        min value for portion of image to filer
    theta_min : int, optional
        minimum angle in degrees (int) to be considered ring artifact
    rwidth : int, optional
        Maximum width of the rings to be filtered in pixels
    int_mode : str, optional
        'WRAP' for wrapping at 0 and 360 degrees, 'REFLECT' for reflective
        boundaries at 0 and 180 degrees.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    nchunk : int, optional
        Chunk size for each core.
    out : ndarray, optional
        Output array for result. If same as arr, process
        will be done in-place.

    Returns
    -------
    ndarray
        Corrected reconstruction data
    """

    rec = dtype.as_float32(rec)

    if out is None:
        out = rec.copy()
    else:
        out = dtype.as_float32(out)

    dz, dy, dx = rec.shape

    if center_x is None:
        center_x = (dx - 1.0)/2.0
    if center_y is None:
        center_y = (dy - 1.0)/2.0

    if int_mode.lower() == 'wrap':
        int_mode = 0
    elif int_mode.lower() == 'reflect':
        int_mode = 1
    else:
        raise ValueError("int_mode should be WRAP or REFLECT")

    args = (center_x, center_y, dx, dy, dz, thresh_max, thresh_min,
            thresh, theta_min, rwidth, int_mode)

    axis_size = rec.shape[0]
    ncore, nchunk = mproc.get_ncore_nchunk(axis_size, ncore, nchunk)
    with cf.ThreadPoolExecutor(ncore) as e:
        for offset in range(0, axis_size, nchunk):
            slc = np.s_[offset:offset+nchunk]
            e.submit(extern.c_remove_ring, out[slc], *args)
    return out


def circ_mask(arr, axis, ratio=1, val=0., ncore=None):
    """
    Apply circular mask to a 3D array.

    Parameters
    ----------
    arr : ndarray
            Arbitrary 3D array.
    axis : int
        Axis along which mask will be performed.
    ratio : int, optional
        Ratio of the mask's diameter in pixels to
        the smallest edge size along given axis.
    val : int, optional
        Value for the masked region.

    Returns
    -------
    ndarray
        Masked array.
    """
    arr = dtype.as_float32(arr)
    val = np.float32(val)
    _arr = arr.swapaxes(0, axis)
    dx, dy, dz = _arr.shape
    mask = _get_mask(dy, dz, ratio)

    with mproc.set_numexpr_threads(ncore):
        ne.evaluate('where(mask, _arr, val)', out=_arr)

    return _arr.swapaxes(0, axis)


def _get_mask(dx, dy, ratio):
    """
    Calculate 2D boolean circular mask.

    Parameters
    ----------
    dx, dy : int
        Dimensions of the 2D mask.

    ratio : int
        Ratio of the circle's diameter in pixels to
        the smallest mask dimension.

    Returns
    -------
    ndarray
        2D boolean array.
    """
    rad1 = dx / 2.
    rad2 = dy / 2.
    if dx < dy:
        r2 = rad1 * rad1
    else:
        r2 = rad2 * rad2
    y, x = np.ogrid[0.5 - rad1:0.5 + rad1, 0.5 - rad2:0.5 + rad2]
    return x * x + y * y < ratio * ratio * r2


def enhance_projs_aps_1id(imgstack, median_ks=5, ncore=None):
    """
    Enhance the projection images with weak contrast collected at APS 1ID

    This filter uses a median fileter (will be switched to enhanced recursive 
    median fileter, ERMF, in the future) for denoising, and a histogram 
    equalization for dynamic range adjustment to bring out the details.

    Parameters
    ----------
    imgstack : np.ndarray
        tomopy images stacks (axis_0 is the oemga direction)
    median_ks : int, optional
        2D median filter kernel size for local noise suppresion
    ncore : int, optional
        number of cores used for speed up

    Returns
    -------
    ndarray
        3D enhanced image stacks.
    """
    ncore = mproc.mp.cpu_count()-1 if ncore is None else ncore

    # need to use multiprocessing to speed up the process
    tmp = []
    with cf.ProcessPoolExecutor(ncore) as e:
        for n_img in range(imgstack.shape[0]):
            tmp.append(e.submit(_enhance_img,
                                imgstack[n_img, :, :],
                                median_ks,
                                )
                       )

    return np.stack([me.result() for me in tmp], axis=0)


def _enhance_img(img, median_ks, normalized=True):
    """
    Enhance the projection image from aps 1ID to counter its weak contrast 
    nature

    Parameters
    ----------
    img : ndarray
        original projection image collected at APS 1ID
    median_ks: int
        kernel size of the 2D median filter, must be odd
    normalized: bool, optional
        specify whether the enhanced image is normalized between 0 and 1,
        default is True

    Returns 
    -------
    ndarray
        enhanced projection image
    """
    wgt = _calc_histequal_wgt(img)
    img = medfilt2d(img, kernel_size=median_ks).astype(np.float64)
    img = ne.evaluate('(img**2)*wgt', out=img)
    return img/img.max() if normalized else img


def _calc_histequal_wgt(img):
    """
    Calculate the histogram equalization weight for a given image

    Parameters
    ----------
    img : ndarray
        2D images

    Returns
    -------
    ndarray
        histogram euqalization weights (0-1) in the same shape as original 
        image
    """
    return (np.sort(img.flatten()).searchsorted(img) + 1)/np.prod(img.shape)
