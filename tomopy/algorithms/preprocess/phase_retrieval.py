# -*- coding: utf-8 -*-
# Filename: phase_retrieval.py
import numpy as np
from tomopy.tools import constants
from tomopy.tools import fftw
import tomopy.tools.multiprocess_shared as mp
import time

# --------------------------------------------------------------------

def phase_retrieval(args):
    """
    Perform single-material phase retrieval
    using projection data.

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    pixel_size : scalar
        Detector pixel size in cm.

    dist : scalar
        Propagation distance of x-rays in cm.

    energy : scalar
        Energy of x-rays in keV.

    alpha : scalar, optional
        Regularization parameter.

    padding : bool, optional
        Applies padding for Fourier transform. For quick testing
        you can use False for faster results.

    Returns
    -------
    phase : ndarray
        Retrieved phase.

    References
    ----------
    - `J. of Microscopy, Vol 206(1), 33-40, 2001 \
    <http://onlinelibrary.wiley.com/doi/10.1046/j.1365-2818.2002.01010.x/abstract>`_
        
    Examples
    --------
    - Phase retrieval:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile)
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> d.normalize()
        >>> 
        >>> # Reconstruct data before phase retrieval
        >>> d.center=661.5
        >>> d.gridrec()
        >>> 
        >>> # Save reconstructed data before phase retrieval
        >>> output_file='tmp/before_phase_retrieval_'
        >>> tomopy.xtomo_writer(d.data_recon, output_file)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>> 
        >>> # Reconstruct data after phase retrieval
        >>> d.phase_retrieval(pixel_size=1e-4, dist=70, energy=20)
        >>> d.gridrec()
        >>> 
        >>> # Save reconstructed data after phase retrieval
        >>> output_file='tmp/after_phase_retrieval_'
        >>> tomopy.xtomo_writer(d.data_recon, output_file)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args
    
    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape) # shared-array
    pixel_size, dist, energy, alpha, padding = inputs

    # Compute the filter.
    H, x_shift, y_shift, tmp_proj = paganin_filter(data,
                                    pixel_size, dist, energy, alpha, padding)    

    num_proj, dx, dy = dshape # dx:slices, dy:pixels
    
    for m in ind:
        proj = data[m, :, :]
        
        if padding:
            tmp_proj[x_shift:dx+x_shift, y_shift:dy+y_shift] = proj
            fft_proj = fftw.fftw2(tmp_proj)
            filtered_proj = np.multiply(H, fft_proj)
            tmp = np.real(fftw.ifftw2(filtered_proj))/np.max(H)
            proj = tmp[x_shift:dx+x_shift, y_shift:dy+y_shift]   
                
        elif not padding:
            fft_proj = fftw.fftw2(proj)
            filtered_proj = np.multiply(H, fft_proj)
            proj = np.real(fftw.ifftw2(filtered_proj))/np.max(H)
        
        data[m, :, :] = proj
    
# --------------------------------------------------------------------

def paganin_filter(data, pixel_size, dist, energy, alpha, padding):
    """
    Calculates Paganin-type filter.
    
    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    pixel_size : scalar
        Detector pixel size in cm.

    dist : scalar
        Propagation distance of x-rays in cm.

    energy : scalar
        Energy of x-rays in keV.

    alpha : scalar, optional
        Regularization parameter.

    padding : bool, optional
        Applies padding for Fourier transform. For quick testing
        you can use False for faster results.

    Returns
    -------
    phase : ndarray
        Paganin filter.

    References
    ----------
    - `J. of Microscopy, Vol 206(1), 33-40, 2001 \
    <http://onlinelibrary.wiley.com/doi/10.1046/j.1365-2818.2002.01010.x/abstract>`_
    """
    num_proj, dx, dy = data.shape # dx:slices, dy:pixels
    wavelength = 2 * constants.PI * constants.PLANCK_CONSTANT * \
                constants.SPEED_OF_LIGHT / energy
                
    if padding:
        # Find padding values.
        pad_value = np.mean((data[:, :, 0] + data[:, :, dy-1]) / 2)
        
        # Fourier padding in powers of 2.
        pad_pixels = np.ceil(constants.PI * wavelength * dist / pixel_size ** 2)
        
        num_x = pow(2, np.ceil(np.log2(dx + pad_pixels)))
        num_y = pow(2, np.ceil(np.log2(dy + pad_pixels)))
        x_shift = int((num_x - dx) / 2.0)
        y_shift = int((num_y - dy) / 2.0)
        
        # Template padded image.
        tmp_proj = pad_value * np.ones((num_x, num_y), dtype='float32')
        
    elif not padding:
        num_x, num_y = dx, dy
        x_shift, y_shift, tmp_proj = None, None, None
        tmp_proj = np.ones((dx, dy), dtype='float32')
                
    # Sampling in reciprocal space.
    indx = (1 / ((num_x-1) * pixel_size)) * np.arange(-(num_x-1)*0.5, num_x*0.5)
    indy = (1 / ((num_y-1) * pixel_size)) * np.arange(-(num_y-1)*0.5, num_y*0.5)
    du, dv = np.meshgrid(indy, indx)
    w2 = np.square(du) + np.square(dv)

    # Filter in Fourier space.
    H = 1 / (wavelength * dist * w2 / (4 * constants.PI) + alpha)
    H = np.fft.fftshift(H)

    return H, x_shift, y_shift, tmp_proj