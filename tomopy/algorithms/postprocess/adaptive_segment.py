# -*- coding: utf-8 -*-
from scipy import ndimage
from skimage.filter import threshold_adaptive
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------


def adaptive_segment(args):
    """
    Applies an adaptive threshold to reconstructed data.

    Also known as local or dynamic thresholding
    where the threshold value is the weighted mean
    for the local neighborhood of a pixel subtracted
    by constant. Alternatively the threshold can be
    determined dynamically by a given function using
    the 'generic' method.

    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]

    block_size : scalar, int
        Uneven size of pixel neighborhood which is
        used to calculate the threshold value
        (e.g. 3, 5, 7, ..., 21, ...).

    offset : scalar, float
         Constant subtracted from weighted mean of
         neighborhood to calculate the local threshold
         value. Default offset is 0.

    Returns
    -------
    output : ndarray
        Thresholded data.

    References
    ----------
    - `http://scikit-image.org/docs/dev/auto_examples/plot_threshold_adaptive.html \
    <http://scikit-image.org/docs/dev/auto_examples/plot_threshold_adaptive.html>`_
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args

    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape)  # shared-array
    block_size, offset = inputs

    for m in ind:
        img = data[m, :, :]

        # Perform scikit adaptive thresholding.
        img = threshold_adaptive(img, block_size=block_size, offset=offset)

        # Remove small white regions
        img = ndimage.binary_opening(img)

        # Remove small black holes
        img = ndimage.binary_closing(img)

        data[m, :, :] = img
