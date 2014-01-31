# -*- coding: utf-8 -*-
# Filename: phase_retrieval.py
import numpy as np


def phase_retrieval(data, pixel_size, dist, energy, alpha=1):
    """Perform single-material phase retrieval
    using projection data.

    Parameters
    ----------
    data : ndarray
        Projection data.

    pixel_size : scalar
        Detector pixel size in cm.

    dist : scalar
        Propagation distance of x-rays in cm.

    energy : scalar
        Energy of x-rays in keV.

    alpha : scalar
        Regularization parameter.

    Returns
    -------
    phase : ndarray
        Retrieved phase.

    References
    ----------
    - `J. of Microscopy, Vol 206(1), 33-40, 2001 \
    <http://onlinelibrary.wiley.com/doi/10.1046/j.1365-2818.2002.01010.x/abstract>`_
    """
    dx, dy = data.shape # dx:slices, dy:pixels
    wavelength = 2 * constants.PI * constants.PLANCK_CONSTANT * \
                 constants.SPEED_OF_LIGHT / energy

    # Fourier padding in powers of 2.
    pad_pixels = np.ceil(constants.PI * wavelength * dist / pixel_size ** 2)
    num_x = pow(2, np.ceil(np.log2(dx + pad_pixels)))
    num_y = pow(2, np.ceil(np.log2(dy + pad_pixels)))
    y_shift = int((num_x - dx) / 2.0)
    x_shift = int((num_y - dy) / 2.0)
    tmp_data = np.ones((num_x, num_y), dtype='complex')

    # Sampling in reciprocal space.
    indx = (1 / ((num_x-1) * pixel_size)) * np.arange(-(num_x-1)*0.5, num_x*0.5)
    indy = (1 / ((num_y-1) * pixel_size)) * np.arange(-(num_y-1)*0.5, num_y*0.5)
    du, dv = np.meshgrid(indy, indx)
    w2 = np.square(du) + np.square(dv)

    # Filter in Fourier space.
    H = 1 / (dist * wavelength * w2 / (4 * constants.PI)  + alpha)

    # Fourier transform of data.
    tmp_data[y_shift:dx+y_shift, x_shift:dy+x_shift] = data
    fft_data = np.fft.fftshift(np.fft.fft2(tmp_data))
    filtered_data = np.fft.ifftshift(np.multiply(H, fft_data))
    tmp = -np.real(np.fft.ifft2(filtered_data))
    data = tmp[y_shift:dx+y_shift, x_shift:dy+x_shift]
    return data
