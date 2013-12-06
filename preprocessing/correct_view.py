# -*- coding: utf-8 -*-
# Filename: correct_view.py
import numpy as np

def correct_view(data, num_overlap_pixels=None):
    """ Stich 180-360 degree projections on 0-180 degree
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
        num_overlap_pixels = optimize_num_overlap_pixels(data)

    num_projection, num_slices, num_pixels = data.shape
    if num_projection % 2 != 0: # if odd
        img_first_half = data[1:num_projection/2 + 1, :, num_overlap_pixels:num_pixels]
        img_second_half = data[num_projection/2:num_projection - 1]
    else:
        img_first_half = data[1:num_projection/2 + 1, :, num_overlap_pixels:num_pixels]
        img_second_half = data[num_projection/2:num_projection]

    ind = range(0, num_pixels)[::-1]
    data = np.c_[img_second_half[:, :, ind], img_first_half]
    return data

def optimize_num_overlap_pixels(data):
    num_projection, num_slices, num_pixels = data.shape
    if num_projection % 2 != 0: # if odd
        img_first_half = np.squeeze(data[1:num_projection/2 + 1, num_slices/2, :])
        img_second_half = np.squeeze(data[num_projection/2:num_projection - 1, num_slices/2, :])
    else:
        img_first_half = np.squeeze(data[1:num_projection/2 + 1, num_slices/2, :])
        img_second_half = np.squeeze(data[num_projection/2:num_projection, num_slices/2, :])
    ind = range(0, num_pixels)[::-1]
    img_second_half = img_second_half[:, ind]

    img1 = np.conj(np.fft.fft2(img_first_half))
    img2 = np.fft.fft2(img_second_half)
    tmp = np.abs(np.fft.ifft2(np.multiply(img1, img2)))
    print tmp.shape[1] - np.unravel_index(np.argmax(tmp), tmp.shape)[1]

    import pylab
    pylab.figure()
    pylab.imshow(tmp, interpolation='none', cmap='gray')
    pylab.show()


def optimize_nop(data,
                slice_no=None,
                center_init=None,
                hist_min=None,
                hist_max=None,
                tol=0.5,
                sigma=2):
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
