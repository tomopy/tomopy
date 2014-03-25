# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import reconstruction
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------

def remove_background(args):
    """
    Remove background from reconstructed data.
    
    We use morphological reconstruction to create 
    a background image, which we can subtract from 
    the original image to isolate bright features.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D reconstructed data with dimensions:
        [slices, pixels, pixels]
         
    Returns
    -------
    output : ndarray
        Background removed data.
        
    References
    ----------
    - `http://scikit-image.org/docs/dev/auto_examples/plot_regional_maxima.html \
    <http://scikit-image.org/docs/dev/auto_examples/plot_regional_maxima.html>`_
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args
    
    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape) # shared-array
    
    for m in ind:
        img = data[m, :, :]
        
        # first remove background.
        seed = np.copy(img)
        seed[1:-1, 1:-1] = img.min()
        img -= reconstruction(seed, img, method='dilation')
        
        data[m, :, :] = img