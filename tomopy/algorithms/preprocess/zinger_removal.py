# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import filters
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------

def zinger_removal(args):
    """
    Zinger removal.

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]
        
    zinger_level : scalar
        Threshold of counts to cut zingers.
        
    median_width : scalar
        Median filter width.
         
    Returns
    -------
    output : ndarray
        Zinger removed data.
        
    Examples
    --------
    - Remove zingers:
        
        >>> import tomopy
        >>> import numpy as np
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>> 
        >>> # Add some artificial zingers to data
        >>> ind = np.random.randint(low=0, high=1005, size=data.shape)
        >>> data[ind>1000] = 2000
        >>> 
        >>> # Save data before zinger removal
        >>> output_file='tmp/before_zinger_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> 
        >>> # Perform zinger removal
        >>> d.zinger_removal(median_width=10)
        >>> 
        >>> # Save data after zinger removal
        >>> output_file='tmp/after_zinger_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args
    
    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape) # shared-array
    zinger_level, median_width = inputs

    zinger_mask = np.zeros((1, data.shape[1], data.shape[2]))

    for m in ind:
        tmp_img = filters.median_filter(data[m, :, :],(median_width, median_width))
        zinger_mask = ((data[m, :, :]-tmp_img) >= zinger_level).astype(int)
        data[m,:,:] = tmp_img*zinger_mask + data[m, :, :]*(1-zinger_mask)