# -*- coding: utf-8 -*-
# Filename: correct_alignment.py
import numpy as np
import shutil
import os
from scipy.optimize import minimize
from scipy import ndimage
from tomoRecon import tomoRecon
from dataio.data_read import Dataset
from dataio.file_types import Tiff

def optimize_center(data,
                    slice_no=None,
                    center_init=None,
                    hist_min=None,
                    hist_max=None,
                    tol=0.5,
                    sigma=2):
    """ Finds the best rotation center for tomographic reconstruction
    using the tomo_recon reconstruction code. This is done by
    ''Nelder-Mead'' routine of the scipy optimization module and
    the cost function is based on image entropy. The optimum
    rotation center is the one that produces the minimum image entropy.

    Parameters
    ----------
    data : ndarray
        Input data.

    slice_no : scalar, optional
        The index of the slice to be used for finding optimal center.
        Default is the central slice.

    center_init : scalar, optional
        The initial guess for the center. Default is half ot the number
        of pixels.

    hist_min : scalar, optional
        The minimum reconstructed value to be used when computing
        the histogram to compute the entropy. The default is the half
        minimum value of the central slice.

    hist_max : scalar, optional
        The maximum reconstructed value to be used when computing the
        histogram to compute the entropy. The default is the twice
        maximum value of the central slice.

    tol : scalar, optional
        Desired sub-pixel accuracy. Default is 1.

    sigma : scalar, optional
        Standard variation of the low pass filter. Default is ``1``.
        This is used for image denoising. Value can be higher for
        datasets having high frequency components
        (e.g., phase-contrast images). Higher values
        increase computation time.

    Returns
    -------
    optimal_center : scalar
        This function returns the index of the center position that
        results in the minimum entropy in the reconstructed image.
    """
    num_slices =  data.shape[1]
    num_pixels =  data.shape[2]

    if slice_no is None:
        slice_no = num_slices / 2
    elif slice_no > num_slices:
        raise ValueError('slice_no is higher than number of available slices.')

    if center_init is None:
        center_init = num_pixels / 2
    elif not np.isscalar(center_init) :
        raise ValueError('center_init must be a scalar.')

    #selectedSlice = np.expand_dims(selectedSlice, axis=1)
    dataset = Dataset(data=data)
    recon = tomoRecon.tomoRecon(dataset)
    recon.run(dataset, sliceNo=slice_no, printInfo=False)
    if hist_min is None:
        hist_min = np.min(recon.data)
        if hist_min < 0:
            hist_min = 2 * hist_min
        elif hist_min >= 0:
            hist_min = 0.5 * hist_min

    if hist_max is None:
        hist_max = np.max(recon.data)
        if hist_max < 0:
            hist_max = 0.5 * hist_max
        elif hist_max >= 0:
            hist_max = 2 * hist_max

    res = minimize(_costFunc,
                   center_init,
                   args=(dataset, recon, slice_no, hist_min, hist_max, sigma),
                   method='Nelder-Mead',
                   tol=tol,
                   options={'disp':True})

    print 'Calculated rotation center : ' + str(np.squeeze(res.x))
    return res.x

def _costFunc(center, data, recon, slice_no, hist_min, hist_max, sigma):
    """ Cost function of the ``optimize_center``.
    """
    data.center = center
    recon.run(data, sliceNo=slice_no, printInfo=False)
    histr, e = np.histogram(ndimage.filters.gaussian_filter(recon.data,
                                                            sigma=sigma),
                            bins=64, range=[hist_min, hist_max])
    histr = histr.astype('float64') / recon.data.size + 1e-12
    print 'Current center : ' + str(np.squeeze(center))
    return -np.dot(histr, np.log2(histr))

def diagnose_center(data,
                    slice_no=None,
                    center_start=None,
                    center_end=None,
                    center_step=None):
    """ Diagnostic tools to find rotation center.

    Helps finding the rotation center manually by
    visual inspection of the selected reconstructions
    with different centers. The outputs for different
    centers are put into ``data/diagnose`` directory
    and the corresponding center positions are printed
    so that one can skim through the images and
    select the best.

    Parameters
    ----------
    data : ndarray
        Input data.

    slice_no : scalar, optional
        The index of the slice to be used for diagnostics.
        Default is the central slice.

    center_start, center_end, center_step : scalar, optional
        Values of the start, end and step of the center values to
        be used for diagnostics.
    """
    num_projections =  data.shape[0]
    num_slices =  data.shape[1]
    num_pixels =  data.shape[2]

    if slice_no is None:
        slice_no = num_slices / 2
    if center_start is None:
        center_start = (num_pixels / 2) - 20
    if center_end is None:
        center_end = (num_pixels / 2) + 20
    if center_step is None:
        center_step = 1

    sliceData = data[:, slice_no, :]
    center = np.arange(center_start, center_end, center_step)
    num_center = center.size
    stacked_slices = np.zeros((num_projections, num_center, num_pixels),
                             dtype='float')
    for m in range(num_center):
        stacked_slices[:, m, :] = sliceData
    dataset = Dataset(data=stacked_slices, center=center)
    recon = tomoRecon.tomoRecon(dataset)
    recon.run(dataset, printInfo=False)
    f = Tiff()
    if os.path.isdir('data/diagnose'):
        shutil.rmtree('data/diagnose')
    f.write(recon.data, filename='data/diagnose/center_.tiff',)
    for m in range(num_center):
        print 'Center for data/diagnose/xxx' + str(m) + '.tiff: ' + str(center[m])

def register_translation(data1, data2, axis=0, num=0):
    """
    """
    if axis == 0:
        data1 = np.squeeze(data1[num, :, :])
        data2 = np.squeeze(data2[num, :, :])

    elif axis == 1:
        data1 = np.squeeze(data1[:, num, :])
        data2 = np.squeeze(data2[:, num, :])

    elif axis == 2:
        data1 = np.squeeze(data1[:, :, num])
        data2 = np.squeeze(data2[:, :, num])

    num_x1, num_y1 = data1.shape
    num_x2, num_y2 = data2.shape

    data1 = -np.log(data1)
    data2 = -np.log(data2)

    if num_x1 > num_x2:
        tmp = np.zeros((num_x1, num_y1))
        tmp[0:num_x2, 0:num_y2] = data2
        data2 = tmp
    elif num_x2 > num_x1:
        tmp = np.zeros((num_x2, num_y2))
        tmp[0:num_x1, 0:num_y1] = data1
        data1 = tmp

    # This stuff below is to filter out
    # slowly varying optics-related fluctuations
    # in data.
    tmp1 = np.fft.fftshift(np.fft.fft2(data1))
    tmp2 = np.fft.fftshift(np.fft.fft2(data2))
    a = 30
    tmp1[tmp1.shape[0]/2-a:tmp1.shape[0]/2+a, tmp1.shape[1]/2-a:tmp1.shape[1]/2+a] = 0
    tmp2[tmp2.shape[0]/2-a:tmp2.shape[0]/2+a, tmp2.shape[1]/2-a:tmp2.shape[1]/2+a] = 0

    data1 = np.abs(np.fft.ifft2(np.fft.ifftshift(tmp1)))
    data2 = np.abs(np.fft.ifft2(np.fft.ifftshift(tmp2)))

    data1 = ndimage.filters.gaussian_filter(data1, sigma=2)
    data2 = ndimage.filters.gaussian_filter(data2, sigma=2)
    # -------------

    data1 = np.fft.fft2(data1)
    data2 = np.fft.fft2(data2)
    tmp = np.abs(np.fft.ifft2(np.multiply(np.conjugate(data1), data2)))
    return np.unravel_index(tmp.argmax(), tmp.shape)
