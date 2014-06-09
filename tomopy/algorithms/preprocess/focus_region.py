# -*- coding: utf-8 -*-
import numpy as np
from tomopy.algorithms.preprocess.correct_drift import correct_drift

# --------------------------------------------------------------------

def focus_region(data, xcoord, ycoord, diameter, center, padded=False, correction=True):
    """
    Uses only a portion of the sinogram for reconstructing
    a circular region of interest (ROI). 

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    xcoord, ycoord : scalar
        X- and Y-coordinates of the center location of the circular ROI.
        
    diameter : scalar
        Diameter of the circular ROI.
        
    center : scalar
        Center of rotation of the original dataset.

    padded : bool, optional
        True if the original sinogram size is preserved.
        
    correction : bool, optional
        True if the correct_drift is applied after ROI selection.
                    
    Returns
    -------
    roidata : ndarray
        Modified ROI data.
    """
    num_projections = data.shape[0]
    num_pixels = data.shape[2]
    
    rad = np.sqrt(xcoord*xcoord+ycoord*ycoord)
    alpha = np.arctan2(xcoord, ycoord)
    
    l1 = center-diameter/2
    l2 = center-diameter/2+rad
    
    if padded: roidata = np.ones((data.shape[0], data.shape[1], data.shape[2]), dtype='float32')
    else: roidata = np.ones((data.shape[0], data.shape[1], diameter), dtype='float32')

    delphi = np.pi/num_projections
    for m in range(num_projections):
        ind1 = np.ceil(np.cos(alpha-m*delphi)*(l2-l1)+l1)
        ind2 = np.floor(np.cos(alpha-m*delphi)*(l2-l1)+l1+diameter)
        
        if ind1 < 0:
            ind1 = 0
        if ind2 < 0:
            ind2 = 0
        if ind1 > num_pixels:
            ind1 = num_pixels
        if ind2 > num_pixels:
            ind2 = num_pixels
            
        if padded: 
            if correction: roidata[m, :, ind1:ind2] = correct_drift(np.expand_dims(data[m, :, ind1:ind2], axis=1), air_pixels=5)
            else: roidata[m, :, ind1:ind2] = data[m, :, ind1:ind2]
        else: 
            if correction: roidata[m, :,0:(ind2-ind1)] = correct_drift(np.expand_dims(data[m, :, ind1:ind2], axis=1), air_pixels=5)
            else: roidata[m, :, 0:(ind2-ind1)] = data[m, :, ind1:ind2]
      
    return roidata
