# -*- coding: utf-8 -*-
# Filename: pyramids.py
""" Module for pyramid reduction and expansion.
""" 
import numpy as np
from scipy import ndimage
from registration import interpolate

def reduce(img, level, laplacian=False): #python has a built-in function called 'reduce', so it might be better to find another name for this.
    """ Image pyramid reduction.
    
    The function performs a low pass filtering (convolution with
    a 2-D gaussian function) to the input image then downscales it 
    by half up to a number of times determined by ``level``.
    
    Returns a Gaussian pyramid reduction of the image ``img`` 
    by the specified ``level``. The number of pixels of ``img``in both 
    X and Y directions should be even. If ``option`` is ``True`` 
    returns also the image Laplacian for all pyramidal levels.
    
    Parameters
    ----------
    img : ndarray
        Input image. The number of pixels of ``img``in both 
        X and Y directions should be even.
    
    level : int
        Number of pyramid reductions.
        
    laplacian : bool, optional
        If ``True`` returns the Laplacian pyramids of the image. 
        
    Returns
    ------
    img : ndarray, shape is half of ``img``
        Output image.
        
    laplacianImg : ndarray of ndarrays
        Laplacians from coarse to fine level. Returned if option is ``True``.
    """
    laplacianImg = np.empty(level, dtype=np.object)
    for m in range(level):
        imgSmooth = ndimage.filters.gaussian_filter(img, 2)
        laplacianImg[m] = img - imgSmooth
        img = interpolate.zoom(imgSmooth, 0.5)
    if laplacian is False:
        return img
    elif laplacian is True:
        return img, laplacianImg[::-1]
            
def expand(img, laplacianImg, level):
    """ Laplacian pyramid expansion of the image.
    
    Returns a Gaussian pyramid expansion of the coarse image ``img`` by the 
    specified level`` using ``imgLaplacian`` which consists of the Laplacian 
    images of the original image at the finer level.
    
    Parameters
    ----------
    img : ndarray
        Input image.
        
    laplacianImg : ndarray of ndarrays
         Laplacians from coarse to fine level.
        
    level : int
        Number of Laplacian pyramid expansions.
    
    Returns
    ------
    out : ndarray, size is double of ``img``
        Output image.
    """        
    for m in range(level):        
        img = interpolate.zoom(img, 2) + laplacianImg[m]
    return img
    
    
    
