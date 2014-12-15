# -*- coding: utf-8 -*-
import numpy as np
import pywt
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------


def stripe_removal(args):
    """
    Remove stripes from sinogram data.

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    level : scalar
        Number of DWT levels.

    wname : str
        Type of the wavelet filter.

    sigma : scalar
        Damping parameter in Fourier space.

    Returns
    -------
    output : ndarray
        Corrected data.

    References
    ----------
    - `Optics Express, Vol 17(10), 8567-8591(2009) \
    <http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-17-10-8567>`_

    Examples
    --------
    - Remove sinogram stripes:

        >>> import tomopy
        >>>
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>>
        >>> # Save data before stripe removal
        >>> output_file='tmp/before_stripe_removal_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>>
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>>
        >>> # Perform stripe removal
        >>> d.stripe_removal(padding=True)
        >>>
        >>> # Save data after stripe removal
        >>> output_file='tmp/after_stripe_removal_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args

    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape)  # shared-array
    level, wname, sigma, padding = inputs

    dx, num_slices, dy = dshape

    # Padded temp image.
    num_x = dx
    if padding:
        num_x = dx + dx / 8

    x_shift = int((num_x - dx) / 2.)
    sli = np.zeros((num_x, dy), dtype='float32')

    for n in ind:
        sli[x_shift:dx + x_shift, :] = data[:, n, :]

        # Wavelet decomposition.
        cH = []
        cV = []
        cD = []
        for m in range(level):
            sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
            cH.append(cHt)
            cV.append(cVt)
            cD.append(cDt)

        # FFT transform of horizontal frequency bands.
        for m in range(level):
            # FFT
            fcV = np.fft.fftshift(np.fft.fft(cV[m], axis=0))
            my, mx = fcV.shape

            # Damping of ring artifact information.
            y_hat = (np.arange(-my, my, 2, dtype='float') + 1) / 2
            damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
            fcV = np.multiply(fcV, np.transpose(np.tile(damp, (mx, 1))))

            # Inverse FFT.
            cV[m] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

        # Wavelet reconstruction.
        for m in range(level)[::-1]:
            sli = sli[0:cH[m].shape[0], 0:cH[m].shape[1]]
            sli = pywt.idwt2((sli, (cH[m], cV[m], cD[m])), wname)

        data[:, n, :] = sli[x_shift:dx + x_shift, 0:dy]
