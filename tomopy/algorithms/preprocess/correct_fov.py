# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------


def correct_fov(data, num_overlap_pixels=None):
    """
    Stich 180-360 degree projections on 0-180 degree
    projections. This is usually the case when field
    of view of the detector is smaller than the
    object to be imaged.

    Parameters
    ----------
    data : ndarray
        Input data.

    num_overlap_pixels : scalar, optional
        The overlapping regions between 0-180 and 180-360
        degree projections.

    Returns
    -------
    data : ndarray
        Output processed data.
    """
    if num_overlap_pixels is None:
        num_overlap_pixels = _optimize_num_overlap_pixels(data)

    #num_projection, num_slices, num_pixels = data.shape
    #if num_projection % 2 != 0: # if odd
    #    img_first_half = data[1:num_projection/2 + 1, :, num_overlap_pixels:num_pixels]
    #    img_second_half = data[num_projection/2:num_projection - 1]
    #else:
    #    img_first_half = data[1:num_projection/2 + 1, :, num_overlap_pixels:num_pixels]
    #    img_second_half = data[num_projection/2:num_projection]

    num_projection, num_slices, num_pixels = data.shape
    if num_projection % 2 != 0:  # if odd
        img_first_half = data[1:num_projection / 2 + 1, :,
                              0:num_pixels - num_overlap_pixels]
        img_second_half = data[num_projection / 2:num_projection - 1]
    else:
        img_first_half = data[1:num_projection / 2 + 1, :,
                              0:num_pixels - num_overlap_pixels]
        img_second_half = data[num_projection / 2:num_projection]

    ind = range(0, num_pixels)[::-1]
    data = np.c_[img_first_half, img_second_half[:, :, ind]]
    return data

# --------------------------------------------------------------------


def _optimize_num_overlap_pixels(data):
    """
    """
    num_projection, num_slices, num_pixels = data.shape
    if num_projection % 2 != 0:  # if odd
        img_first_half = np.squeeze(data[1:num_projection / 2 + 1,
                                    num_slices / 2, :])
        img_second_half = np.squeeze(data[num_projection / 2:num_projection - 1,
                                     num_slices / 2, :])
    else:
        img_first_half = np.squeeze(data[1:num_projection / 2 + 1,
                                    num_slices / 2, :])
        img_second_half = np.squeeze(data[num_projection / 2:num_projection,
                                     num_slices / 2, :])
    ind = range(0, num_pixels)[::-1]
    img_second_half = img_second_half[:, ind]

    img_first_half = ndimage.filters.gaussian_filter(img_first_half, sigma=2)
    img_second_half = ndimage.filters.gaussian_filter(img_second_half, sigma=2)

    gx1, gy1 = np.gradient(img_first_half)
    gx2, gy2 = np.gradient(img_second_half)
    img_first_half = np.power(gx1, 2) + np.power(gy1, 2)
    img_second_half = np.power(gx2, 2) + np.power(gy2, 2)

    img1 = np.fft.fft(img_first_half)
    img2 = np.fft.fft(img_second_half)
    tmp = np.real(np.fft.ifft(np.multiply(np.conj(img2), img1)))
    return np.argmax(np.sum(np.abs(tmp), axis=0))
