# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '../..', 'lib/libprep.so'))
libprep = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def apply_padding(data, num_pad):
    """
    Applies zero padding to each projection data.
    
    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]
        
    num_pad : scalar, int32
        New dimension of the projections  
        after padding.
         
    Returns
    -------
    output : ndarray
        Padded data.
        
    Examples
    --------
    - Apply padding:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>> 
        >>> # Save data before padding
        >>> output_file='tmp/before_pad_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>> 
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> 
        >>> # Perform padding
        >>> d.apply_padding()
        >>> 
        >>> # Save data after padding
        >>> output_file='tmp/after_pad_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')
    
    if num_pad < num_pixels:
        return data

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    padded_data = np.ones((num_projections, num_slices, num_pad), 
                           dtype='float32')
    
    libprep.apply_padding.restype = ctypes.POINTER(ctypes.c_void_p)
    libprep.apply_padding(data.ctypes.data_as(c_float_p),
                          ctypes.c_int(num_projections),
                          ctypes.c_int(num_slices),
                          ctypes.c_int(num_pixels),
                          ctypes.c_int(num_pad),
                          padded_data.ctypes.data_as(c_float_p))
    return padded_data