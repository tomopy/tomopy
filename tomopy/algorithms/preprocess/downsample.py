# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
libpath = os.path.join(os.path.dirname(__file__), '../../lib/libprep.so')
libprep = ctypes.CDLL(os.path.abspath(libpath))

# --------------------------------------------------------------------


def downsample2d(data, level):
    """
    Downsample the slices by binning.

    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    level : scalar, int32
        Downsampling level. For example level=2
        means, the sinogram will be downsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    output : ndarray
        Downsampled 3-D tomographic data with dimensions:
        [projections, slices/level^2, pixels]

    Examples
    --------
    - Downsample data:

        >>> import tomopy
        >>>
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>>
        >>> # Save data before downsampling
        >>> output_file='tmp/before_downsampling_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>>
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>>
        >>> # Perform downsampling
        >>> d.downsample2d(level=2)
        >>>
        >>> # Save data after downsampling
        >>> output_file='tmp/after_downsampling_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    if level < 0:
        return data

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)

    binsize = np.power(2, level)
    downsampled_data = np.zeros((num_projections,
                                 num_slices,
                                 num_pixels / binsize),
                                dtype='float32')

    libprep.downsample2d.restype = ctypes.POINTER(ctypes.c_void_p)
    libprep.downsample2d(data.ctypes.data_as(c_float_p),
                         ctypes.c_int(num_projections),
                         ctypes.c_int(num_slices),
                         ctypes.c_int(num_pixels),
                         ctypes.c_int(level),
                         downsampled_data.ctypes.data_as(c_float_p))
    return downsampled_data

# --------------------------------------------------------------------


def downsample3d(data, level):
    """
    Downsample the slices and pixels by binning.

    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    level : scalar, int32
        Downsampling level. For example level=2
        means, the sinogram will be downsampled by 4,
        and level=3 means upsampled by 8.

    Returns
    -------
    downsampled_data : ndarray
        Downsampled 3-D tomographic data with dimensions:
        [projections, slices/level^2, pixels/level^2]

    Examples
    --------
    - Downsample data:

        >>> import tomopy
        >>>
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=4)
        >>>
        >>> # Save data before downsampling
        >>> output_file='tmp/before_downsampling_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>>
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>>
        >>> # Perform downsampling
        >>> d.downsample3d(level=2)
        >>>
        >>> # Save data after downsampling
        >>> output_file='tmp/after_downsampling_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    if level < 0:
        return data

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)

    binsize = np.power(2, level)
    downsampled_data = np.zeros((num_projections,
                                 num_slices / binsize,
                                 num_pixels / binsize),
                                dtype='float32')

    libprep.downsample3d.restype = ctypes.POINTER(ctypes.c_void_p)
    libprep.downsample3d(data.ctypes.data_as(c_float_p),
                         ctypes.c_int(num_projections),
                         ctypes.c_int(num_slices),
                         ctypes.c_int(num_pixels),
                         ctypes.c_int(level),
                         downsampled_data.ctypes.data_as(c_float_p))
    return downsampled_data
