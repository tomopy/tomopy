# -*- coding: utf-8 -*-
# Filename: image.py
""" Module for visualization tools.
"""
import matplotlib.pylab as plt

def show_slice(data, slice_no=0, clim=None):
    """ Visualize the reconstructed slice.

    Parameters
    -----------
    data : ndarray
        3-D matrix of stacked reconstructed slices.

    slice_no : scalar, optional
        The index of the slice to be imaged.
    """
    plt.figure(figsize=(7, 7))
    if len(data.shape) is 2:
        plt.imshow(data,
                   interpolation='none',
                   cmap='gray')
        plt.colorbar()
    elif len(data.shape) is 3:
        plt.imshow(data[slice_no, :, :],
                   interpolation='none',
                   cmap='gray')
        plt.colorbar()
    if clim is not None:
        plt.clim(clim)
    plt.show()
