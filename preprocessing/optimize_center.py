# -*- coding: utf-8 -*-
# Filename: optimize_center.py
import numpy as np
from scipy.optimize import minimize

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
    print "Opimizing rotation center using Nelder-Mead method..."
    
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
    recon = tomoRecon.tomoRecon(self)
    recon.run(self, slice_no=slice_no, printInfo=False)
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
    
    res = minimize(read._costFunc,
                   center_init,
                   args=(self, recon, slice_no, hist_min, hist_max, sigma),
                   method='Nelder-Mead',
                   tol=tol,
                   options={'disp':True})
                   
    print 'Calculated rotation center : ' + str(np.squeeze(res.x))
    return res.x

@staticmethod
def _costFunc(center, data, recon, slice_no, hist_min, hist_max, sigma):
    """ Cost function of the ``optimize_center``.
    """
    data.center = center
    recon.run(data, slice_no=slice_no, printInfo=False)
    histr, e = np.histogram(ndimage.filters.gaussian_filter(recon.data,
                                                            sigma=sigma),
                            bins=64, range=[hist_min, hist_max])
    histr = histr.astype('float64') / recon.data.size + 1e-12
    print 'Current center : ' + str(np.squeeze(center))
    return -np.dot(histr, np.log2(histr))
