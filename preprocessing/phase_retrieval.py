# -*- coding: utf-8 -*-
# Filename: phase_retrieval.py
import numpy as np
import constants
from tomoRecon import tomoRecon

def single_material(data, pixel_size, dist, energy, delta_over_mu=1e-8):
    """ Perform single-material phase retrieval
    according to Paganin's method.

    Parameters
    ----------
    data : ndarray
        Input data.

    pixel_size : scalar
        Detector pixel size in cm.

    dist : scalar
        Propagation distance of x-rays in cm.

    energy : scalar
        Energy of x-rays in keV.

    delta_over_mu : scalar
        Ratio of the imaginary part of the refractive index
        decrement to the attenuation of the material.

    Returns
    -------
    phase : ndarray
        Retrieved phase.

    References
    ----------
    - J. of Microscopy, Vol 206(1), 33-40(2001)
    """
    # Size of the detector
    num_projections, num_slices, num_pixels = data.shape

    # Sampling in reciprocal space.
    indx = (1 / ((num_slices - 1) * pixel_size)) * \
        np.arange(-(num_slices-1)*0.5, num_slices*0.5)
    indy = (1 / ((num_pixels - 1) * pixel_size)) * \
        np.arange(-(num_pixels-1)*0.5, num_pixels*0.5)
    du, dv = np.meshgrid(indy, indx)
    w2 = np.square(du) + np.square(dv)

    # Fourier transform of data.
    for m in range(num_projections):
        fft_data = np.fft.fftshift(tomoRecon.fftw2d(data[m, : ,:], direction='forward'))
        H = 1 / (4 * np.square(constants.PI) * dist * delta_over_mu * w2 + 1)
        filtered_data = np.fft.ifftshift(np.multiply(H, fft_data))
        data[m, : ,:] = -np.log(np.real(tomoRecon.fftw2d(filtered_data, direction='backward')))
    return data

def pure_phase(data, pixel_size, dist, energy, alpha=0.5):
    """ Perform Bronnikov-type phase retrieval.

    Parameters
    ----------
    data : ndarray
        Input data.

    pixel_size : scalar
        Detector pixel size in cm.

    dist : scalar
        Propagation distance of x-rays in cm.

    energy : scalar
        Energy of x-rays in keV.

    Returns
    -------
    phase : ndarray
        Retrieved phase.
    """
    # Size of the detector
    num_projections, num_slices, num_pixels = data.shape

    # Wavelength of x-rays.
    wavelength = (2 * constants.PI *
                constants.PLANCK_CONSTANT *
                constants.SPEED_OF_LIGHT) / energy

    # Sampling in reciprocal space.
    indx = (1 / ((num_slices - 1) * pixel_size)) * \
            np.arange(-(num_slices-1)*0.5, num_slices*0.5)
    indy = (1 / ((num_pixels - 1) * pixel_size)) * \
            np.arange(-(num_pixels-1)*0.5, num_pixels*0.5)
    du, dv = np.meshgrid(indy, indx)
    w2 = np.square(du) + np.square(dv)

    # Right-hand side term:
    data = 1 - data

    # Fourier transform of data.
    for m in range(num_projections):
        fft_data = np.fft.fftshift(tomoRecon.fftw2d(data[m, : ,:], direction='forward'))
        H = 1 / (2 * constants.PI * wavelength * dist * w2 + alpha)
        filtered_data = np.fft.ifftshift(np.multiply(H, fft_data))
        data[m, : ,:] = -np.real(tomoRecon.fftw2d(filtered_data, direction='backward'))
    return data
