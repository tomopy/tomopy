# -*- coding: utf-8 -*-
import numpy as np
import pywt
from tomopy.tools.multiprocess import worker

@worker
def stripe_removal(args):
    """
    Remove stripes from sinogram data.

    Parameters
    ----------
    data : ndarray
        Projection data.

    level : scalar
        Number of DWT levels.

    wname : str
        Type of the wavelet filter.

    sigma : scalar
        Damping parameter in Fourier space.

    References
    ----------
    - `Optics Express, Vol 17(10), 8567-8591(2009) \
    <http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-17-10-8567>`_
    """
    data, level, wname, sigma = args
    
    dx, dy = data.shape
    # Wavelet decomposition.
    cH = []
    cV = []
    cD = []
    for m in range(level):
        data, (cHt, cVt, cDt) = pywt.dwt2(data, wname)
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
    for m in range(level)[::-1]:
        data = data[0:cH[m].shape[0], 0:cH[m].shape[1]]
        data = pywt.idwt2((data, (cH[m], cV[m], cD[m])), wname)
    return data[0:dx, 0:dy]
    