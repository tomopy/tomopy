# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.fftpack as spf
import ipdb
import imreg
from pylab import show, matshow

def align_projections(data, align_to_channel=None, method='cross-correlation', output_gifs=False, output_dir=None):
    """
    Align projections in rotation series.

    This function uses different approaches to align
    projections. This is typically required for data
    acquired from a scanning microscope where there
    can be significant relative motion between
    projections.

    Parameters
    ----------
    data : ndarray, float32
        4-D tomographic data with dimensions:
        [channels, projections, slices, pixels]

    align_to_channel: int32
        specify which channel to use when calculating (x,y) shifts for each projection.
        If None, sum channels.

    method: string
        Specify which method to use to align the projections.
        Available choices:
            - cross_correlation
            - phase_correlation
            - scale_and_rotation_invariant_phase_correlation

    output_gif: Boolean
        Whether to output a gif of the unaligned and aligned data

    output_dir: string
        Where to write the gif files.


    Returns
    -------
    output : ndarray
        Aligned data.

    output: ndarray
        Shifts that have been applied to data.
        Format [theta, x, y]

    References
    ----------
    ``An FFT-Based Technique for Translation, Rotation, and Scale-Invariant
    Image Registration``, B. Srinivasa Reddy and B. N. Chatterji,
    IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 5 , NO. 8, P. 1266, (1996)

    Examples
    --------
    - Auto-align projections

        >>> import tomopy
        >>>
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>>
        >>> # Save data before correction
        >>> output_file='tmp/before_correction_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>>
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>>
        >>> # Perform correction
        >>> d.correct_drift()
        >>>
        >>> # Save data after correction
        >>> output_file='tmp/after_correction_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """

    num_channels = data.shape[0]
    num_projections = data.shape[1]
    num_slices = data.shape[2]
    num_pixels = data.shape[3]

    if align_to_channel:
        unaligned = data[align_to_channel,:,:,:]
    else:
        unaligned = np.sum(data,axis=0)

    shifts = []
    ipdb.set_trace()
    if method == 'cross-correlation':
        for n in range(1, num_projections):
            t0, t1 = cross_correlate(unaligned[n,:,:], unaligned[n-1,:,:])
            unaligned[n,:,:] = translate(unaligned[n,:,:], -t0, -t1)
            #matshow(unaligned[n,:,:])
            #matshow(unaligned[n-1,:,:])
            #show()
            shifts.append([t0,t1])
    elif method == 'phase-correlation':
        for n in range(1, num_projections):
            t0, t1 = phase_correlate(unaligned[n,:,:], unaligned[n-1,:,:])
            unaligned[n,:,:] = translate(unaligned[n,:,:], t0, t1)
            shifts.append([t0,t1])
    elif method == 'rotation_and_scale_invariant_phase_correlation':
        for n in range(1, num_projections):
            img, scale, angle, (t0, t1) = imreg.similarity(unaligned[n,:,:], unaligned[n-1,:,:])
            unaligned[n,:,:] = img
            shifts.append([t0,t1])
    else:
        return

    aligned = unaligned

    return aligned, shifts

def translate(img, t0, t1):
    return np.roll(np.roll(img, t1, axis=1), t0, axis=0)

def cross_correlate(a, b):
    fa = spf.fft2(a)
    fb = spf.fft2(b)

    shape = a.shape
    c = abs(spf.ifft2(fa*fb.conjugate()))
    t0, t1 = np.unravel_index(np.argmax(c), a.shape)
    if t0 > shape[0]//2:
        t0 -= shape[0]
    if t1 > shape[1]//2:
        t1 -= shape[1]
    return [t0, t1]

def phase_correlate(a, b):
    fa = spf.fft2(a)
    fb = spf.fft2(b)

    shape = a.shape
    c = abs(spf.ifft2(fa*fb.conjugate()/(abs(fa)*abs(fb))))
    t0, t1 = np.unravel_index(np.argmax(c), a.shape)
    if t0 > shape[0]//2:
        t0 -= shape[0]
    if t1 > shape[1]//2:
        t1 -= shape[1]
    return [t0, t1]

def test_case():
    # Generate test data
    lena = sp.misc.lena()
    shape = lena.shape
    mask = np.where(sp.hypot(*sp.ogrid[-shape[0]/2:shape[0]/2, -shape[1]/2:shape[1]/2])<50, 1.0, 0.0)
    lena *= mask

    shift = [10,15]

    # Test cross correlation
    try:
        rec_shift = cross_correlate(lena, translate(lena, -shift[0], -shift[1]))
        assert rec_shift == shift
        print('Cross correlation test: PASSED')
    except AssertionError:
        print('Cross correlation test: FAILED')

    # Test phase correlation
    try:
        rec_shift = phase_correlate(lena, translate(lena, -shift[0], -shift[1]))
        assert rec_shift == shift
        print('Phase correlation test: PASSED')
    except AssertionError:
        print('Phase correlation test: FAILED')

    # Test invariant correlation
    try:
        img, scale, angle, rec_shift= imreg.similarity(lena, translate(lena, -shift[0], -shift[1]))
        rec_shift[0] *= -1
        rec_shift[1] *= -1
        assert rec_shift == shift
        print('Invariant correlation test: PASSED')
    except AssertionError:
        print('Invariant correlation test: FAILED')
        ipdb.set_trace()


if __name__ == '__main__':
    test_case()
