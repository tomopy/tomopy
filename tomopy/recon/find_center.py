# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy import ndimage
from gridrec import Gridrec
import logging
logger = logging.getLogger("tomopy")


def find_center(data,
                slice_no=None,
                center_init=None,
                hist_min=None,
                hist_max=None,
                tol=0.5,
                sigma=2):
    """ 
    Find the distance between the rotation axis and the middle
    of the detector field-of-view.
    
    The function exploits systematic artifacts in reconstructed images 
    due to shifts in the rotation center [1]. It uses image entropy
    as the error metric and ''Nelder-Mead'' routine (of the scipy 
    optimization module) as the optimizer.

    Parameters
    ----------
    data : ndarray
        Input data.

    slice_no : scalar, optional
        The index of the slice to be used for finding optimal center.

    center_init : scalar, optional
        The initial guess for the center.

    hist_min : scalar, optional
        The minimum reconstructed value to be used when computing
        the histogram to compute the entropy. The default is the half
        minimum value of the central slice.

    hist_max : scalar, optional
        The maximum reconstructed value to be used when computing the
        histogram to compute the entropy. The default is the twice
        maximum value of the central slice.

    tol : scalar, optional
        Desired sub-pixel accuracy.

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
        
    References
    ----------
    [1] `SPIE Proceedings, Vol 6318, 631818(2006) \
    <dx.doi.org/10.1117/12.679101>`_
    """
    num_slices =  data.shape[1]
    num_pixels =  data.shape[2]

    # Use middle slice is no slice is specified.
    if slice_no is None:
        slice_no = num_slices / 2
    
    # Use middle point of the detector area if the center is absent.
    if center_init is None:
        center_init = num_pixels / 2
       
    # Make an initial reconstruction to adjust histogram limits. 
    recon = Gridrec(data)
    recon.run(data, center=center_init, slice_no=slice_no)
    
    # Adjust histogram boundaries if given.
    if hist_min is None:
        hist_min = np.min(recon.data_recon)
        if hist_min < 0:
            hist_min = 2 * hist_min
        elif hist_min >= 0:
            hist_min = 0.5 * hist_min

    if hist_max is None:
        hist_max = np.max(recon.data_recon)
        if hist_max < 0:
            hist_max = 0.5 * hist_max
        elif hist_max >= 0:
            hist_max = 2 * hist_max

    # Magic is ready to happen...
    res = minimize(_costFunc,
                   center_init,
                   args=(data, recon, slice_no, hist_min, hist_max, sigma),
                   method='Nelder-Mead',
                   tol=tol)
    
    # Have a look at what I found:
    logger.info('calculated rotation center: ' + str(np.squeeze(res.x)))
    return res.x

def _costFunc(center, data, recon, slice_no, hist_min, hist_max, sigma):
    """ Cost function of the ``optimize_center``.
    """
    logger.info('trying center: ' + str(np.squeeze(center)))
    recon.run(data, center=center, slice_no=slice_no)
    histr, e = np.histogram(ndimage.filters.gaussian_filter(recon.data_recon,
                                                            sigma=sigma),
                            bins=64, range=[hist_min, hist_max])
    histr = histr.astype('float32') / recon.data_recon.size + 1e-12
    return -np.dot(histr, np.log2(histr))