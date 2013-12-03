# -*- coding: utf-8 -*-
# Filename: correct_view.py

def correct_view(data, num_overlap_pixels=0):
    """ Stich 180-360 degree projections on 0-180 degree
    projections. This is usually the case when field
    of view of the detector is smaller than the
    object to be imaged.
        
    Parameters
    ----------
    data : ndarray
        Input data.
        
    num_overlap_pixels : scalar, optional
        The overlapping regions between 0-180 and 180-360
        degree projections.
        
    Returns
    -------
    data : ndarray
        Output processed data.
    """
    print "Correcting field of view..."
    num_projection, num_slices, num_pixels = self.data.shape
    
    if num_projection % 2 != 0: # if odd
        img_first_half = self.data[1:num_projection/2 + 1, :, num_overlap_pixels:num_pixels]
        img_second_half = self.data[num_projection/2:num_projection - 1]
    else:
        img_first_half = self.data[1:num_projection/2 + 1, :, num_overlap_pixels:num_pixels]
        img_second_half = self.data[num_projection/2:num_projection]
    
    ind = range(0, num_pixels)[::-1]
    data = np.c_[img_second_half[:, :, ind], img_first_half]
    return data
