# -*- coding: utf-8 -*-
import numpy as np
from skimage import morphology
from skimage.filter import sobel
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------


def region_segment(args):
    """
    Applies an region-based segementation to reconstructed data.

    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]

    low : scalar, int
       Lowest value for the marker.

    high : scalar, int
       Highest value for the marker.

    Returns
    -------
    output : ndarray
        Segmented data.
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args

    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape)  # shared-array
    low, high = inputs

    for m in ind:
        img = data[m, :, :]
        elevation_map = sobel(img)

        markers = np.zeros_like(img)
        markers[img < low] = 1
        markers[img > high] = 2

        img = morphology.watershed(elevation_map, markers)
        data[m, :, :] = img
