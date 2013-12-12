# -*- coding: utf-8 -*-
# Filename: zoom.py
import numpy as np
from registration import interpolate

def zoom(data, scale, axis=0, kind='bilinear'):
    """
    """
    num_projections, num_slices, num_pixels = data.shape
    if axis == 0:
        interp_data = np.empty((num_projections,
                             num_slices * scale,
                             num_pixels * scale))
        for m in range(num_projections):
            interp_data[m, :, :] = interpolate.zoom(data[m, :, :],
                                                 scale, kind=kind)
    if axis == 1:
        interp_data = np.empty((num_projections * scale,
                             num_slices,
                             num_pixels * scale))
        for m in range(num_slices):
            interp_data[:, m, :] = interpolate.zoom(data[:, m, :],
                                                 scale, kind=kind)
    if axis == 2:
        interp_data = np.empty((num_projections * scale,
                             num_slices * scale,
                             num_pixels))
        for m in range(num_pixels):
            interp_data[:, :, m] = interpolate.zoom(data[:, :, m],
                                                 scale, kind=kind)

    return interp_data
