# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy import ndimage

from gridrec import Gridrec

# --------------------------------------------------------------------

def _optimize_center(data, theta, slice_no, center_init, tol):
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

    slice_no : scalar
        The index of the slice to be used for finding optimal center.

    center_init : scalar
        The initial guess for the center.

    tol : scalar
        Desired sub-pixel accuracy.

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
    # Make an initial reconstruction to adjust histogram limits. 
    recon = Gridrec(data, airPixels=20, ringWidth=10)
    recon.reconstruct(data, theta=theta, center=center_init, slice_no=slice_no)
    
    # Adjust histogram boundaries according to reconstruction.
    hist_min = np.min(recon.data_recon)
    if hist_min < 0:
        hist_min = 2 * hist_min
    elif hist_min >= 0:
        hist_min = 0.5 * hist_min
        
    hist_max = np.max(recon.data_recon)
    if hist_max < 0:
        hist_max = 0.5 * hist_max
    elif hist_max >= 0:
        hist_max = 2 * hist_max

    # Magic is ready to happen...
    res = minimize(_costFunc, center_init,
                   args=(data, recon, theta, slice_no, hist_min, hist_max),
                   method='Nelder-Mead', tol=tol)
    
    # Have a look at what I found:
    print "calculated rotation center: " + str(np.squeeze(res.x))
    return res.x
    
# --------------------------------------------------------------------

def _costFunc(center, data, recon, theta, slice_no, hist_min, hist_max):
    """ 
    Cost function of the ``optimize_center``.
    """
    print 'trying center: ' + str(np.squeeze(center))
    recon.reconstruct(data, theta=theta, center=center, slice_no=slice_no)
    histr, e = np.histogram(ndimage.filters.gaussian_filter(recon.data_recon, sigma=2.), 
                            bins=64, range=[hist_min, hist_max])
    histr = histr.astype('float32') / recon.data_recon.size + 1e-12
    return -np.dot(histr, np.log2(histr))