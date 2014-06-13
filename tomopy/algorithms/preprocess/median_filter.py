# -*- coding: utf-8 -*-
from scipy.ndimage import filters
import tomopy.tools.multiprocess_shared as mp

# --------------------------------------------------------------------

def median_filter(args):
    """
    Apply median filter to data.

    Parameters
    ----------
    data : ndarray
        3-D tomographic data with dimensions:
        [projections, slices, pixels]
        
    size : scalar
        The size of the filter. 
        
    axis : scalar
        Define the axis of data for filtering.
        0: projections, 1:sinograms, 2:pixels

    Returns
    -------
    output : ndarray
        Median filtered data.
        
    Examples
    --------
    - Apply median-filter to sinograms:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>> 
        >>> # Save data before filtering
        >>> output_file='tmp/before_filtering_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> 
        >>> # Perform filtering
        >>> d.median_filter()
        >>> 
        >>> # Save data after filtering
        >>> output_file='tmp/after_filtering_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    # Arguments passed by multi-processing wrapper
    ind, dshape, inputs = args
    
    # Function inputs
    data = mp.tonumpyarray(mp.shared_arr, dshape) # shared-array
    size, axis = inputs
    
    if axis == 0:
        for m in ind:
            data[m, :, :] = filters.median_filter(data[m, :, :], (size, size))
    elif axis == 1:
        for m in ind:
            data[:, m, :] = filters.median_filter(data[:, m, :], (size, size))
    elif axis == 2:
        for m in ind:
            data[:, :, m] = filters.median_filter(data[:, :, m], (size, size))





