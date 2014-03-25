# -*- coding: utf-8 -*-
from skimage.filter import threshold_otsu
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------

def threshold_segment(args):
    """
    Applies threshold based on Otsu's method to reconstructed data.
    
    Otsu’s method calculates an “optimal” threshold 
    (marked by a red line in the histogram below) by 
    maximizing the variance between two classes of pixels, 
    which are separated by the threshold. Equivalently, 
    this threshold minimizes the intra-class variance.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]
        
    cutoff : scalar, int
        Manually selected threshold value.
         
    Returns
    -------
    output : ndarray
        Thresholded data.
        
    References
    ----------
    - `http://en.wikipedia.org/wiki/Otsu’s_method \
    <http://en.wikipedia.org/wiki/Otsu’s_method>`_
    - `http://scikit-image.org/docs/dev/auto_examples/plot_otsu.html#example-plot-otsu-py \
    <http://scikit-image.org/docs/dev/auto_examples/plot_otsu.html#example-plot-otsu-py>`_
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args
    
    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape) # shared-array
    cutoff = inputs
    
    for m in ind:
        img = data[m, :, :]
        if cutoff == None:
            cutoff = threshold_otsu(img)
        img = img > cutoff
        data[m, :, :] = img