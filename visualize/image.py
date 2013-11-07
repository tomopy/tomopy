# -*- coding: utf-8 -*-
# Filename: image.py
""" Module for visualization tools.
"""
import matplotlib.pylab as plt


def showSlice(reconOutput, sliceNo=0, clim=None):
    """ Visualize the reconstructed slice.
    
    Parameters
    -----------
    reconOutput : ndarray
        3-D matrix of stacked reconstructed slices.
        
    slices : scalar, optional
        The index of the slice to be imaged. By default the first
        slice is picked.
    """ 
    plt.figure(figsize=(7, 7))
    if len(reconOutput.shape) is 2:
        plt.imshow(reconOutput, 
                   interpolation='none',
                   cmap='gray')
        plt.colorbar()
    elif len(reconOutput.shape) is 3:
        plt.imshow(reconOutput[sliceNo, :, :], 
                   interpolation='none',
                   cmap='gray')
        plt.colorbar()
    if clim is not None:
        plt.clim(clim)
    plt.show()