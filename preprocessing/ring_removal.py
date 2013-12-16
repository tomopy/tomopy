# -*- coding: utf-8 -*-
# Filename: remove_rings.py
import numpy as np
import pywt

def dwtfft(data, level=6, wname='db10', sigma=2):
    """ Remove ring artifacts.

    Parameters
    ----------
    data : ndarray
        Input stack of projections.

    level : scalar, optional
        Number of DWT levels.

    wname : str, optional
        Type of the wavelet filter.

    sigma : scalar, optional
        Damping parameter in Fourier space.

    References
    ----------
    - Optics Express, Vol 17(10), 8567-8591(2009)
    """
    for n in range(data.shape[1]):
        # Wavelet decomposition.
        im = data[:, n, :]
        cH = []
        cV = []
        cD = []
        for m in range(level):
            im, (cHt, cVt, cDt) = pywt.dwt2(im, wname)
            cH.append(cHt)
            cV.append(cVt)
            cD.append(cDt)

        # FFT transform of horizontal frequency bands.
        for m in range(level):
            # FFT
            fcV = np.fft.fftshift(np.fft.fft(cV[m], axis=0))
            my, mx = fcV.shape

            # Damping of ring artifact information.
            y_hat = (np.arange(-my, my, 2, dtype='float')+1) / 2
            damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
            fcV = np.multiply(fcV, np.transpose(np.tile(damp, (mx, 1))))

            # Inverse FFT.
            cV[m] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

        # Wavelet reconstruction.
        nim = im
        for m in range(level)[::-1]:
            nim = nim[0:cH[m].shape[0], 0:cH[m].shape[1]]
            nim = pywt.idwt2((nim, (cH[m], cV[m], cD[m])), wname)
        nim = nim[0:data.shape[0], 0:data.shape[2]]
        data[:, n, :] = nim
    return data
